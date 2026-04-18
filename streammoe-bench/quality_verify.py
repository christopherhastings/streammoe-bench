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


def sample_tokens(port: int, prompt: str, n_predict: int) -> list:
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
    # llama-server returns sampled token ids under "tokens" when populated;
    # fall back to content text if not (still deterministic for byte compare).
    if "tokens" in data and data["tokens"]:
        return list(data["tokens"])
    return list(data.get("content", "").encode("utf-8"))


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
               "binary": args.binary, "matches": 0, "mismatches": []}

    # Phase 1: baseline (production flags only, no patch layers).
    print(f"\n=== baseline ({args.model}) ===", flush=True)
    base_port = 11461
    base_log = out / f"server_{args.model}_baseline.log"
    base_proc = start_server(args.binary, model, base_port, [], base_log)
    deadline = 900 if "bf16" in args.model else 360
    if not wait_ready(base_port, time.monotonic() + deadline):
        print("baseline did not become ready", file=sys.stderr); return 1
    baseline_tokens = []
    try:
        for i, p in enumerate(prompts):
            baseline_tokens.append(sample_tokens(base_port, p, args.n_predict))
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
            pt = sample_tokens(patched_port, p, args.n_predict)
            if pt == baseline_tokens[i]:
                results["matches"] += 1
            else:
                n = min(len(pt), len(baseline_tokens[i]))
                div = next((k for k in range(n) if pt[k] != baseline_tokens[i][k]), n)
                results["mismatches"].append({
                    "idx": i, "diverge_at": div,
                    "baseline_len": len(baseline_tokens[i]),
                    "patched_len": len(pt),
                })
            if i % 10 == 0: print(f"  patched {i+1}/{len(prompts)}", flush=True)
    finally:
        stop_server(patched_proc)

    result_path = out / f"quality_verify_{args.model}_{int(time.time())}.json"
    result_path.write_text(json.dumps(results, indent=2))
    total = len(prompts)
    print(f"\nmatches:    {results['matches']}/{total}")
    print(f"mismatches: {len(results['mismatches'])}")
    for m in results["mismatches"][:5]:
        print(f"  idx={m['idx']} diverge_at={m['diverge_at']} "
              f"bl={m['baseline_len']} pt={m['patched_len']}")
    print(f"\nResults: {result_path}")
    return 0 if results["matches"] == total else 2


if __name__ == "__main__":
    sys.exit(main())
