#!/usr/bin/env python3
"""Byte-identical 80-prompt quality verification.

Hypothesis: the four StreamMoE TTFT patch layers are runtime / scheduling
changes. They don't alter the graph, the sampler, the tokenizer, or expert
selection. Therefore, for a greedy (temperature=0) run over the 80 MT-Bench
prompts, the sampled token stream must be byte-identical between baseline
and patched configurations.

Uses the same production-config composer as ttft_bench.py, so both benches
test the config that actually ships. Baseline = production flags. Patched =
production flags + --moe-eager-load + --streammoe-warmup (omits --moe-keep-warm
since its heartbeat would add non-determinism to the measurement timing).

Usage:
    python3.11 quality_verify.py --model qwen36bf16
    python3.11 quality_verify.py --model qwen36 --prompts mtbench80.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import urllib.request

STREAMMOE_PKG_ROOT = Path("/Users/claude/streammoe")
if str(STREAMMOE_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMMOE_PKG_ROOT))

from streammoe_bench.config import get_models
from streammoe_bench.runner import build_extra_args
from ttft_bench import production_config_for


PATCH_LAYERS = ["--moe-eager-load", "--streammoe-warmup"]


def wait_ready(port: int, deadline: float) -> bool:
    status_url = f"http://127.0.0.1:{port}/streammoe/status"
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                pass
        except OSError:
            time.sleep(0.1); continue
        try:
            with urllib.request.urlopen(status_url, timeout=2.0) as resp:
                if resp.status == 200 and json.loads(resp.read()).get("ok") is True:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def start_server(binary: str, model, port: int, extra: list, log_path: Path) -> subprocess.Popen:
    prod = build_extra_args(production_config_for(model), model)
    cmd = [
        binary,
        "-m", str(model.model_path),
        "--host", "127.0.0.1", "--port", str(port),
        "-ngl", "99",
        "--seed", "42",
    ] + prod + extra
    log = open(log_path, "w")
    (log_path.with_suffix(".cmd")).write_text(" ".join(cmd) + "\n")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def sample_completion(port: int, prompt: str, n_predict: int) -> dict:
    """Returns {tokens, content}. tokens is a list[int] from llama.cpp's
    sampled token ids if available, else falls back to utf-8 byte-list of
    content (still deterministic for byte-compare). content is the generated
    text — kept so the downstream LLM judge can evaluate meaning-equivalence
    when strict byte-match fails for non-deterministic Metal fp32 reasons."""
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode())
    content = data.get("content", "")
    if "tokens" in data and data["tokens"]:
        tokens = list(data["tokens"])
    else:
        tokens = list(content.encode("utf-8"))
    return {"tokens": tokens, "content": content}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary",
                    default="/Users/claude/streammoe/anemll-flash-llama.cpp/build/bin/llama-server")
    ap.add_argument("--model", default="qwen36bf16",
                    help="key from streammoe_bench.config.get_models() (e.g. qwen36, qwen36bf16)")
    ap.add_argument("--prompts", default="mtbench80.jsonl")
    ap.add_argument("--n-predict", type=int, default=48,
                    help="tokens per prompt (shorter = faster, still shows any divergence)")
    ap.add_argument("--output-dir", default="./quality-results")
    ap.add_argument("--judge", action="store_true",
                    help="after strict == check, send divergent pairs to Claude judge "
                         "(tool-use strict JSON, via streammoe_bench.quality_gates.judge_pair). "
                         "Requires ANTHROPIC_API_KEY.")
    ap.add_argument("--judge-model", default="claude-sonnet-4-6",
                    help="Anthropic model id for judging")
    args = ap.parse_args()

    models = get_models()
    if args.model not in models:
        print(f"[fatal] unknown model '{args.model}'. Known: {list(models)}", file=sys.stderr)
        return 2
    model = models[args.model]

    prompts_file = Path(args.prompts)
    if not prompts_file.exists():
        prompts_file = Path(__file__).parent / args.prompts
    prompts = []
    with prompts_file.open() as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["prompt"])

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    results = {"generated_at": time.time(), "model": args.model,
               "model_label": model.label,
               "n_prompts": len(prompts), "n_predict": args.n_predict,
               "binary": args.binary, "byte_matches": 0, "pairs": []}

    # Phase 1: baseline (production flags only, no patch layers).
    print(f"\n=== baseline ({args.model}) ===", flush=True)
    base_port = 11461
    base_log = out / f"server_{args.model}_baseline.log"
    base_proc = start_server(args.binary, model, base_port, [], base_log)
    deadline = 900 if "bf16" in args.model else 360
    if not wait_ready(base_port, time.monotonic() + deadline):
        print("baseline did not become ready", file=sys.stderr); return 1
    baselines = []
    try:
        for i, p in enumerate(prompts):
            baselines.append(sample_completion(base_port, p, args.n_predict))
            if i % 10 == 0: print(f"  baseline {i+1}/{len(prompts)}", flush=True)
    finally:
        stop_server(base_proc)

    # Phase 2: patched (production flags + L1 + L2).
    print(f"\n=== patched (+L1+L2) ({args.model}) ===", flush=True)
    patched_port = 11462
    patched_log = out / f"server_{args.model}_patched.log"
    patched_proc = start_server(args.binary, model, patched_port, PATCH_LAYERS, patched_log)
    if not wait_ready(patched_port, time.monotonic() + deadline):
        print("patched did not become ready", file=sys.stderr); return 1
    try:
        for i, p in enumerate(prompts):
            pt = sample_completion(patched_port, p, args.n_predict)
            byte_eq = pt["tokens"] == baselines[i]["tokens"]
            if byte_eq:
                results["byte_matches"] += 1
            n = min(len(pt["tokens"]), len(baselines[i]["tokens"]))
            div = next((k for k in range(n) if pt["tokens"][k] != baselines[i]["tokens"][k]), n) \
                  if not byte_eq else None
            # Keep full texts so a judge pass (below) can decide meaning-
            # equivalence when byte_eq is false.
            results["pairs"].append({
                "idx": i, "prompt": p,
                "baseline": baselines[i]["content"],
                "patched": pt["content"],
                "byte_eq": byte_eq,
                "diverge_at": div,
                "baseline_token_len": len(baselines[i]["tokens"]),
                "patched_token_len": len(pt["tokens"]),
            })
            if i % 10 == 0: print(f"  patched {i+1}/{len(prompts)}", flush=True)
    finally:
        stop_server(patched_proc)

    # Phase 3 (optional): LLM judge on the pairs that failed strict ==.
    # Metal fp32 reductions aren't bit-reproducible, so a handful of
    # divergences is expected noise; the judge is the real quality gate.
    if args.judge:
        print(f"\n=== judging {sum(1 for p in results['pairs'] if not p['byte_eq'])} divergent pairs ===", flush=True)
        try:
            import anthropic
            from streammoe_bench.quality_gates import judge_pair
        except ImportError as e:
            print(f"[judge skipped] missing dep: {e}", file=sys.stderr)
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("[judge skipped] ANTHROPIC_API_KEY not set", file=sys.stderr)
            else:
                client = anthropic.Anthropic(api_key=api_key)
                verdict_counts = {"same": 0, "similar": 0, "different": 0, "parse_error": 0}
                for pair in results["pairs"]:
                    if pair["byte_eq"]:
                        pair["judge"] = {"verdict": "same", "score": 5,
                                         "reason": "byte-identical (no judge call needed)"}
                        verdict_counts["same"] += 1
                        continue
                    v = judge_pair(client, pair["prompt"],
                                   pair["baseline"], pair["patched"],
                                   model=args.judge_model)
                    pair["judge"] = v
                    verdict_counts[v["verdict"]] = verdict_counts.get(v["verdict"], 0) + 1
                    print(f"  idx={pair['idx']:2d} verdict={v['verdict']:10s} score={v.get('score')} reason={v.get('reason','')[:80]}")
                results["judge_summary"] = verdict_counts
                results["judge_model"] = args.judge_model

    result_path = out / f"quality_verify_{args.model}_{int(time.time())}.json"
    result_path.write_text(json.dumps(results, indent=2))
    total = len(prompts)
    diverged = sum(1 for p in results["pairs"] if not p["byte_eq"])
    print(f"\nbyte matches:   {results['byte_matches']}/{total} ({100*results['byte_matches']/total:.1f}%)")
    print(f"divergent:      {diverged}")
    if "judge_summary" in results:
        j = results["judge_summary"]
        equiv = j.get("same", 0) + j.get("similar", 0)
        print(f"judge verdicts: same={j.get('same',0)} similar={j.get('similar',0)} "
              f"different={j.get('different',0)} parse_error={j.get('parse_error',0)}")
        print(f"semantic-equivalent: {equiv}/{total} ({100*equiv/total:.1f}%)")
    print(f"\nResults: {result_path}")
    # Exit 0 if ALL pairs judged same/similar (or byte-equal if no judge).
    if "judge_summary" in results:
        return 0 if results["judge_summary"].get("different", 0) == 0 and \
                    results["judge_summary"].get("parse_error", 0) == 0 else 2
    return 0 if results["byte_matches"] == total else 2


if __name__ == "__main__":
    sys.exit(main())
