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

import argparse
import json
import os
import sys
import time
from pathlib import Path

STREAMMOE_PKG_ROOT = Path("/Users/claude/streammoe")
if str(STREAMMOE_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMMOE_PKG_ROOT))


LOCAL_CONFIGS = ["q4_resident", "q4_streaming", "bf16_streaming"]
REFERENCES    = ["haiku", "sonnet"]


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
    """Judge every prompt where BOTH local and ref produced output."""
    from streammoe_bench.quality_gates import judge_pair

    common_ids = sorted(set(local_responses) & set(ref_responses))
    counts = {"same": 0, "similar": 0, "different": 0, "parse_error": 0}
    by_category: dict[str, dict] = {}
    verdicts = []

    print(f"\n=== {local_name} vs {ref_name}  ({len(common_ids)} prompts) ===", flush=True)
    t0 = time.monotonic()
    for i, pid in enumerate(common_ids):
        local = local_responses[pid]
        ref   = ref_responses[pid]
        prompt = prompts.get(pid, {}).get("prompt") or local.get("prompt") or ref.get("prompt")
        category = prompts.get(pid, {}).get("category") or ref.get("category")
        # judge_pair is positional: (client, prompt, stock, sidecar).
        # "stock" = reference model; "sidecar" = our local model.
        v = judge_pair(client, prompt, ref["content"], local["content"], model=judge_model)
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
        bc = by_category.setdefault(category or "unknown",
                                    {"same": 0, "similar": 0, "different": 0, "parse_error": 0})
        bc[v["verdict"]] = bc.get(v["verdict"], 0) + 1
        verdicts.append({"prompt_id": pid, "category": category, **v})
        if (i + 1) % 10 == 0:
            dt = time.monotonic() - t0
            print(f"  {local_name} vs {ref_name} {i+1}/{len(common_ids)}  "
                  f"same={counts['same']:2d} sim={counts['similar']:2d} "
                  f"diff={counts['different']:2d}  ({dt:.0f}s)", flush=True)

    result = {
        "local": local_name, "reference": ref_name,
        "judge_model": judge_model,
        "counts": counts, "by_category": by_category,
        "verdicts": verdicts,
        "n_compared": len(common_ids),
    }
    out_path.write_text(json.dumps(result, indent=2))
    return result


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

    locals_data = {n: load_local_responses(compare_dir, n) for n in local_names}
    refs_data   = {n: load_reference(ref_dir, n) for n in ref_names}

    grid = []
    for ln in local_names:
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
    print(f"\n{'local':20s} {'reference':10s} {'n':>4s} {'same':>5s} {'sim':>5s} {'diff':>5s} {'equiv%':>7s}")
    for r in grid:
        c = r["counts"]
        total = sum(c.values())
        equiv = c["same"] + c["similar"]
        print(f"{r['local']:20s} {r['reference']:10s} {r['n_compared']:>4d} "
              f"{c['same']:>5d} {c['similar']:>5d} {c['different']:>5d} "
              f"{100*equiv/max(1,total):>7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
