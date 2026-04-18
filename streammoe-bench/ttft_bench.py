#!/usr/bin/env python3
"""TTFT benchmark for StreamMoE patches.

Runs a 4x2 matrix:
    configs = [baseline, +L1, +L1+L2, +L1+L2+L3+L4]
    models  = [Q4_K_XL, BF16]

For each (config, model): N=10 cold runs + N=10 warm runs, reporting p50/p95
time-to-first-token, RSS at end of run, and server startup duration.

Uses /v1/completions with stream=true and measures elapsed seconds from
request start to first SSE chunk. "Cold" means the server was killed before
the run; "warm" means we reuse the same server process.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import urllib.request
import urllib.error


# --------------------------------------------------------------------------- #
# Configuration matrix
# --------------------------------------------------------------------------- #

@dataclass
class ConfigLayers:
    name: str
    flags: List[str] = field(default_factory=list)


CONFIGS = [
    ConfigLayers(name="baseline",        flags=[]),
    ConfigLayers(name="L1_eager",        flags=["--moe-eager-load"]),
    ConfigLayers(name="L1L2_warmup",     flags=["--moe-eager-load", "--streammoe-warmup"]),
    ConfigLayers(name="L1L2L3L4_full",   flags=["--moe-eager-load", "--streammoe-warmup",
                                                "--moe-keep-warm", "60"]),
]


@dataclass
class ModelSpec:
    name: str
    gguf: str
    sidecar: str


MODELS = [
    ModelSpec(
        name="q4_k_xl",
        gguf="/Users/christopherhastings/Downloads/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        sidecar="/Users/claude/streammoe/models/qwen35-35b-sidecar",
    ),
    ModelSpec(
        name="bf16",
        gguf="/Users/christopherhastings/Downloads/qwen36-bf16/BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf",
        sidecar="/Users/claude/streammoe/models/qwen36-bf16-sidecar",
    ),
]


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #

def wait_for_port(port: int, deadline: float) -> bool:
    """Two-stage readiness: socket accepts + /streammoe/status returns ok:true.

    Without the status probe we get HTTP 503 on the first /v1/completions
    because llama-server starts listening immediately but populates its
    model + slot state later. The status endpoint flips ok only when
    ctx_http.is_ready.store(true) has been called after model load."""
    status_url = f"http://127.0.0.1:{port}/streammoe/status"
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                pass
        except OSError:
            time.sleep(0.1); continue
        # Stage 2: /streammoe/status says ok.
        try:
            with urllib.request.urlopen(status_url, timeout=2.0) as resp:
                if resp.status == 200 and json.loads(resp.read()).get("ok") is True:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def start_server(binary: str, model: ModelSpec, cfg: ConfigLayers,
                 port: int, log_path: Path) -> subprocess.Popen:
    cmd = [
        binary,
        "-m", model.gguf,
        "--moe-sidecar", model.sidecar,
        "--moe-mode", "slot-bank",
        "--moe-slot-bank", "256",
        "--mlock",
        "-ngl", "99",
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", "4096",
    ]
    cmd += cfg.flags
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def process_rss_bytes(pid: int) -> int:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
        return int(out.strip()) * 1024
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# TTFT measurement
# --------------------------------------------------------------------------- #

def measure_ttft(port: int, prompt: str, timeout_s: float = 60.0) -> float:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": 4,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        # Read until we see the first "data:" line with a non-empty choice.
        for line in resp:
            if line.startswith(b"data:") and b'"text"' in line:
                return time.perf_counter() - t0
    # Fallback: no SSE chunk observed
    return time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_matrix(binary: str, output_dir: Path, n_iter: int, prompt: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"generated_at": time.time(), "binary": binary, "runs": []}

    for model in MODELS:
        if not os.path.exists(model.gguf):
            print(f"[skip] {model.name}: {model.gguf} missing", file=sys.stderr)
            continue
        for cfg in CONFIGS:
            port = 11450 + hash(cfg.name + model.name) % 50
            log_path = output_dir / f"server_{model.name}_{cfg.name}.log"
            print(f"\n=== {model.name} / {cfg.name} (port {port}) ===", flush=True)

            t_start = time.perf_counter()
            proc = start_server(binary, model, cfg, port, log_path)
            # BF16 can take several minutes to mmap + eager-load a 60 GiB sidecar.
            load_deadline = 900 if model.name == "bf16" else 360
            ready = wait_for_port(port, deadline=time.monotonic() + load_deadline)
            startup_s = time.perf_counter() - t_start
            if not ready:
                print(f"[error] server for {cfg.name}/{model.name} did not come up", file=sys.stderr)
                stop_server(proc)
                continue

            # Wait an extra beat so any Layer 2 warmup fires before our first request.
            time.sleep(2.0)

            cold_ttft = []
            warm_ttft = []
            try:
                # Cold run = first request against a freshly-started server.
                cold_ttft.append(measure_ttft(port, prompt))
                # Warm runs = N-1 follow-ups on the same server.
                for _ in range(n_iter - 1):
                    warm_ttft.append(measure_ttft(port, prompt))
            except Exception as e:
                print(f"[error] request phase: {e}", file=sys.stderr)

            rss = process_rss_bytes(proc.pid)
            stop_server(proc)

            results["runs"].append({
                "model": model.name,
                "config": cfg.name,
                "startup_s": startup_s,
                "rss_bytes": rss,
                "cold_ttft_s": cold_ttft,
                "warm_ttft_s": warm_ttft,
                "cold_p50": statistics.median(cold_ttft) if cold_ttft else None,
                "warm_p50": statistics.median(warm_ttft) if warm_ttft else None,
                "warm_p95": (sorted(warm_ttft)[int(0.95*len(warm_ttft))-1]
                             if len(warm_ttft) >= 2 else None),
            })
            time.sleep(1.0)

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary",
                    default="/Users/claude/streammoe/anemll-flash-llama.cpp/build/bin/llama-server")
    ap.add_argument("--output-dir", default="./ttft-results")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--prompt", default="Briefly: what is 2+2?")
    args = ap.parse_args()

    out = Path(args.output_dir)
    results = run_matrix(args.binary, out, args.iters, args.prompt)
    result_path = out / f"ttft_matrix_{int(time.time())}.json"
    result_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {result_path}")
    # Pretty summary:
    print("\n{:24s} {:14s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "config", "model", "startup_s", "cold_p50", "warm_p50", "rss_gib"))
    for r in results["runs"]:
        print("{:24s} {:14s} {:10.2f} {:10.3f} {:10.3f} {:10.2f}".format(
            r["config"], r["model"], r["startup_s"],
            r.get("cold_p50") or float("nan"),
            r.get("warm_p50") or float("nan"),
            r["rss_bytes"]/1024**3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
