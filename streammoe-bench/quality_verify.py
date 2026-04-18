#!/usr/bin/env python3
"""80-prompt byte-identical quality verification.

Hypothesis: The four TTFT layers are pure runtime / scheduling changes.
None of them alters the graph, the sampler, the tokenizer, or the expert
selection. Therefore, for a greedy (temperature=0) run over the MT-Bench
80 prompts, the sampled token stream must be byte-identical between
baseline and +L1+L2+L3+L4.

Usage:
    ./quality_verify.py --binary /path/to/llama-server --prompts mtbench.jsonl

Runs two llama-server instances (baseline and patched), sends the same 80
prompts with temperature=0, and asserts the token-id lists match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import urllib.request


BASELINE_FLAGS: List[str] = []
PATCHED_FLAGS: List[str] = [
    "--moe-eager-load",
    "--streammoe-warmup",
    # --moe-keep-warm deliberately omitted here; its heartbeats would race
    # the benchmark request timing and add non-determinism.
]


def wait_for_port(port: int, deadline: float) -> bool:
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def start_server(binary: str, model: str, sidecar: str, port: int,
                 extra: List[str], log_path: Path) -> subprocess.Popen:
    cmd = [
        binary,
        "-m", model, "--moe-sidecar", sidecar,
        "--moe-mode", "slot-bank", "--moe-slot-bank", "256",
        "--mlock", "-ngl", "99",
        "--host", "127.0.0.1", "--port", str(port),
        "-c", "8192", "--seed", "42",
    ] + extra
    log = open(log_path, "w")
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


def sample_tokens(port: int, prompt: str, n_predict: int) -> List[int]:
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
    # llama-server returns sampled token ids under "tokens" when that field
    # is populated; fall back to text-hash if tokens aren't returned.
    if "tokens" in data and data["tokens"]:
        return list(data["tokens"])
    return [ord(c) for c in data["content"]]  # coarse fallback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary",
                    default="/Users/claude/streammoe/anemll-flash-llama.cpp/build/bin/llama-server")
    ap.add_argument("--model",
                    default="/Users/christopherhastings/Downloads/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
    ap.add_argument("--sidecar",
                    default="/Users/claude/streammoe/models/qwen35-35b-sidecar")
    ap.add_argument("--prompts", default="mtbench80.jsonl",
                    help="JSONL file, one {\"prompt\": str} per line")
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--output-dir", default="./quality-results")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    prompts = []
    with open(args.prompts) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["prompt"])

    results = {"generated_at": time.time(), "n_prompts": len(prompts),
               "binary": args.binary, "matches": 0, "mismatches": []}

    # Phase 1: baseline
    print("\n=== baseline run ===", flush=True)
    base_port = 11461
    base_proc = start_server(args.binary, args.model, args.sidecar, base_port,
                             BASELINE_FLAGS, out / "server_baseline.log")
    if not wait_for_port(base_port, time.monotonic() + 300):
        print("baseline server failed to start", file=sys.stderr); return 1
    baseline_tokens = []
    try:
        for i, p in enumerate(prompts):
            baseline_tokens.append(sample_tokens(base_port, p, args.n_predict))
            if i % 10 == 0: print(f"  baseline {i+1}/{len(prompts)}", flush=True)
    finally:
        stop_server(base_proc)

    # Phase 2: patched
    print("\n=== patched run ===", flush=True)
    patched_port = 11462
    patched_proc = start_server(args.binary, args.model, args.sidecar, patched_port,
                                PATCHED_FLAGS, out / "server_patched.log")
    if not wait_for_port(patched_port, time.monotonic() + 300):
        print("patched server failed to start", file=sys.stderr); return 1
    try:
        for i, p in enumerate(prompts):
            pt = sample_tokens(patched_port, p, args.n_predict)
            if pt == baseline_tokens[i]:
                results["matches"] += 1
            else:
                # Find first divergence index for diagnostics.
                div = next((k for k in range(min(len(pt), len(baseline_tokens[i])))
                           if pt[k] != baseline_tokens[i][k]), min(len(pt), len(baseline_tokens[i])))
                results["mismatches"].append({"idx": i, "diverge_at": div,
                                             "baseline_len": len(baseline_tokens[i]),
                                             "patched_len": len(pt)})
            if i % 10 == 0: print(f"  patched {i+1}/{len(prompts)}", flush=True)
    finally:
        stop_server(patched_proc)

    result_path = out / f"quality_verify_{int(time.time())}.json"
    result_path.write_text(json.dumps(results, indent=2))
    total = len(prompts)
    print(f"\nmatches:  {results['matches']}/{total}")
    print(f"mismatches: {len(results['mismatches'])}")
    for m in results["mismatches"][:5]:
        print(f"  idx={m['idx']} diverge_at={m['diverge_at']} "
              f"baseline_len={m['baseline_len']} patched_len={m['patched_len']}")
    return 0 if results["matches"] == total else 2


if __name__ == "__main__":
    sys.exit(main())
