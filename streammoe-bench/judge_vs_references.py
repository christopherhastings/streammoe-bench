#!/usr/bin/env python3
"""Judge locally-run model outputs against frontier reference answers.

Inputs:
    - quality-results/compare/responses_q4_resident.json
    - quality-results/compare/responses_q4_streaming.json
    - quality-results/compare/responses_bf16_streaming.json
      (each holds 80 full responses from quality_compare.py)

    - reference-answers/mtbench80_haiku.jsonl
    - reference-answers/mtbench80_sonnet.jsonl
      (each holds 80 responses from the Anthropic reference models)

Cross-pairs (default):
    q4_resident   vs haiku    q4_resident   vs sonnet
    q4_streaming  vs haiku    q4_streaming  vs sonnet
    bf16_streaming vs haiku   bf16_streaming vs sonnet

For every (local, reference) pair, prompt-by-prompt, calls the claude-
sonnet-4-6 tool-use judge (streammoe_bench.quality_gates.judge_pair) and
writes the verdict distribution — same / similar / different — per pair.

This gives the matrix our unified doc needs: do the three local configs
(stock-resident, Q4-streaming, BF16-streaming) produce answers
equivalent to Haiku and Sonnet on the 80-prompt MT-Bench suite, or do
they diverge, and where?
"""

from __future__ import annotations

import re

import argparse
import json
import os
import sys
import time
from pathlib import Path

STREAMMOE_PKG_ROOT = Path("/Users/claude/streammoe")
if str(STREAMMOE_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMMOE_PKG_ROOT))


LOCAL_CONFIGS = ["q4_resident", "q4_streaming", "bf16_streaming",
                 "qwen35_9b", "qwen35_122b"]
REFERENCES    = ["haiku", "sonnet"]


# --------------------------------------------------------------------------- #
# Qwen3 <think> handling
# --------------------------------------------------------------------------- #
#
# Qwen3 family models (including the 35B-A3B MoEs and the new 9B / 122B-A10B
# variants) produce reasoning inside <think>...</think> before the actual
# answer. The judge should evaluate the post-think answer, not the reasoning
# trace — otherwise a model that thinks carefully but answers tersely gets
# graded down on style of its scratchpad instead of its conclusion.
#
# Two helpers:
#   strip_think: remove closed <think>...</think> blocks AND any open
#                <think>...EOF tail (which happens when the response hits
#                n_predict mid-thought).
#   is_no_answer: the response was *all* thinking and produced no post-think
#                answer. These prompts are surfaced separately in the
#                verdict distribution rather than fed to the judge with an
#                empty string — the judge would mark them "different" which
#                is technically true but doesn't tell you the failure mode.

def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks and any unclosed <think>...EOF."""
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()


def is_no_answer(raw: str) -> bool:
    """True when the response is all thinking with no actual answer."""
    return bool(raw.strip()) and not strip_think(raw)


def load_local_responses(compare_dir: Path, name: str) -> dict[str, dict]:
    path = compare_dir / f"responses_{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing local responses: {path}")
    blob = json.loads(path.read_text())
    # keyed by prompt_id so we can line up against the references.
    return {r["prompt_id"]: r for r in blob["responses"] if r.get("content")}


def load_reference(ref_dir: Path, name: str) -> dict[str, dict]:
    path = ref_dir / f"mtbench80_{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing reference file: {path}")
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            if rec.get("content"):
                out[rec["id"]] = rec
    return out


def load_prompts(prompts_file: Path) -> dict[str, dict]:
    out = {}
    with prompts_file.open() as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                out[p["id"]] = p
    return out


def judge_pair_set(client, local_name, ref_name, local_responses,
                   ref_responses, prompts, judge_model, out_path) -> dict:
    """Judge every prompt where BOTH local and ref produced output.

    Resumable: if `out_path` already exists from a previous run, we load
    the partial results and skip prompts that have a verdict. The full
    result is re-written to disk after every prompt, so killing the
    process loses at most the verdict that was in-flight at the moment.
    """
    from streammoe_bench.quality_gates import judge_pair

    EMPTY_COUNTS = {"same": 0, "similar": 0, "different": 0,
                    "no_answer": 0, "parse_error": 0}

    common_ids = sorted(set(local_responses) & set(ref_responses))

    # Resume from a partial file if present. Skip any prompt_id we've
    # already judged; re-derive counts from the retained verdicts so a
    # partial rerun doesn't double-count.
    done_ids: set[str] = set()
    counts = dict(EMPTY_COUNTS)
    by_category: dict[str, dict] = {}
    verdicts = []
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text())
            for v in prior.get("verdicts", []):
                if v.get("prompt_id") and "verdict" in v:
                    done_ids.add(v["prompt_id"])
                    counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
                    cat = v.get("category") or "unknown"
                    bc = by_category.setdefault(cat, dict(EMPTY_COUNTS))
                    bc[v["verdict"]] = bc.get(v["verdict"], 0) + 1
                    verdicts.append(v)
            if done_ids:
                print(f"[resume] {local_name} vs {ref_name}: "
                      f"{len(done_ids)}/{len(common_ids)} already judged",
                      flush=True)
        except Exception as e:
            print(f"[warn] couldn't resume from {out_path}: {e}", file=sys.stderr)

    print(f"\n=== {local_name} vs {ref_name}  "
          f"({len(common_ids) - len(done_ids)} new, {len(common_ids)} total) ===",
          flush=True)

    def save():
        out_path.write_text(json.dumps({
            "local": local_name, "reference": ref_name,
            "judge_model": judge_model,
            "counts": counts, "by_category": by_category,
            "verdicts": verdicts,
            "n_compared": len(common_ids),
        }, indent=2))

    t0 = time.monotonic()
    for i, pid in enumerate(common_ids):
        if pid in done_ids:
            continue
        local = local_responses[pid]
        ref   = ref_responses[pid]
        prompt = prompts.get(pid, {}).get("prompt") or local.get("prompt") or ref.get("prompt")
        category = prompts.get(pid, {}).get("category") or ref.get("category")

        # Qwen3 produces <think>...</think> before the real answer. Strip
        # it before judging so the judge sees the answer, not the scratch-
        # pad. If the response is *all* thinking (n_predict cutoff hit
        # mid-think), surface "no_answer" instead of feeding an empty
        # string to the judge.
        raw_local = local["content"]
        if is_no_answer(raw_local):
            v = {"verdict": "no_answer", "score": 0,
                 "reason": "Model produced only <think> tokens; no actual answer."}
        else:
            stripped_local = strip_think(raw_local)
            # "stock" = reference model; "sidecar" = our local model.
            v = judge_pair(client, prompt, ref["content"], stripped_local,
                           model=judge_model)

        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
        bc = by_category.setdefault(category or "unknown", dict(EMPTY_COUNTS))
        bc[v["verdict"]] = bc.get(v["verdict"], 0) + 1
        verdicts.append({"prompt_id": pid, "category": category, **v})

        # Per-prompt checkpoint — a crash now loses at most one judge call.
        save()

        if (i + 1) % 10 == 0:
            dt = time.monotonic() - t0
            print(f"  {local_name} vs {ref_name} {i+1}/{len(common_ids)}  "
                  f"same={counts['same']:2d} sim={counts['similar']:2d} "
                  f"diff={counts['different']:2d} na={counts['no_answer']:2d}  "
                  f"({dt:.0f}s)", flush=True)

    save()
    return {
        "local": local_name, "reference": ref_name,
        "judge_model": judge_model,
        "counts": counts, "by_category": by_category,
        "verdicts": verdicts,
        "n_compared": len(common_ids),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-dir", default="./quality-results/compare",
                    help="directory containing responses_<name>.json from quality_compare.py")
    ap.add_argument("--ref-dir", default="./reference-answers",
                    help="directory containing mtbench80_<model>.jsonl")
    ap.add_argument("--prompts", default="mtbench80.jsonl")
    ap.add_argument("--output-dir", default="./quality-results/compare")
    ap.add_argument("--judge-model", default="claude-sonnet-4-6")
    ap.add_argument("--locals", default=",".join(LOCAL_CONFIGS))
    ap.add_argument("--refs",   default=",".join(REFERENCES))
    args = ap.parse_args()

    try:
        import anthropic
    except ImportError:
        print("[fatal] pip install anthropic", file=sys.stderr); return 2
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("[fatal] ANTHROPIC_API_KEY not set", file=sys.stderr); return 2
    client = anthropic.Anthropic(api_key=api_key)

    compare_dir = Path(args.compare_dir)
    ref_dir     = Path(args.ref_dir)
    out_dir     = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    prompts_file = Path(args.prompts)
    if not prompts_file.exists():
        prompts_file = Path(__file__).parent / args.prompts
    prompts = load_prompts(prompts_file)

    local_names = [s.strip() for s in args.locals.split(",") if s.strip()]
    ref_names   = [s.strip() for s in args.refs.split(",")   if s.strip()]

    # Gracefully skip locals that don't have a responses file yet (e.g.
    # qwen35_9b / qwen35_122b before they've been sampled).
    locals_data = {}
    for n in local_names:
        try:
            locals_data[n] = load_local_responses(compare_dir, n)
        except FileNotFoundError as e:
            print(f"[skip local] {n}: {e}", file=sys.stderr)

    refs_data   = {n: load_reference(ref_dir, n) for n in ref_names}

    grid = []
    for ln in local_names:
        if ln not in locals_data:
            continue
        for rn in ref_names:
            out_path = out_dir / f"judge_{ln}_vs_{rn}.json"
            result = judge_pair_set(
                client, ln, rn,
                locals_data[ln], refs_data[rn],
                prompts, args.judge_model,
                out_path,
            )
            grid.append(result)

    # Summary row per (local, reference) pair.
    summary_path = out_dir / f"judge_cross_reference_{int(time.time())}.json"
    summary_path.write_text(json.dumps({
        "generated_at": time.time(),
        "judge_model": args.judge_model,
        "grid": grid,
    }, indent=2))
    print(f"\nResults: {summary_path}")
    print(f"\n{'local':20s} {'reference':10s} {'n':>4s} "
          f"{'same':>5s} {'sim':>5s} {'diff':>5s} {'na':>4s} {'equiv%':>7s}")
    for r in grid:
        c = r["counts"]
        total = sum(c.values())
        equiv = c.get("same", 0) + c.get("similar", 0)
        print(f"{r['local']:20s} {r['reference']:10s} {r['n_compared']:>4d} "
              f"{c.get('same',0):>5d} {c.get('similar',0):>5d} "
              f"{c.get('different',0):>5d} {c.get('no_answer',0):>4d} "
              f"{100*equiv/max(1,total):>7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
