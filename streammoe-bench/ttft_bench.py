#!/usr/bin/env python3
"""TTFT benchmark for StreamMoE patches — production-config aware.

This harness does NOT hand-roll llama-server flags. Instead it imports the
validated production config + flag builder from the existing
`streammoe_bench` package (at ~/streammoe/streammoe_bench) so that every
TTFT cell runs against the exact same flags used in the user-facing
production deployment. That way, if the production config is updated
later (new prefetch heuristic, new I/O flag, new default), this bench
picks it up without a code change here.

The only thing this bench adds on top of the production flags is the
StreamMoE TTFT patch layers (--moe-eager-load / --streammoe-warmup /
--moe-keep-warm).

Matrix:
    configs = [baseline, +L1, +L1+L2, +L1+L2+L3+L4]
    models  = [Q4_K_XL, BF16]

For each (config, model): N cold runs + (N-1) warm runs, reporting
TTFT (first-SSE-chunk latency), RSS, startup time.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import urllib.request

# --------------------------------------------------------------------------- #
# Wire up the production streammoe_bench package
# --------------------------------------------------------------------------- #

STREAMMOE_PKG_ROOT = Path("/Users/claude/streammoe")
if str(STREAMMOE_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMMOE_PKG_ROOT))

try:
    from streammoe_bench.config import (
        BenchConfig, ConfigParams, ModelDef, get_models,
    )
    from streammoe_bench.runner import build_extra_args
    _HAS_PRODUCTION_CONFIG = True
except ImportError as exc:
    print(f"[fatal] cannot import streammoe_bench from {STREAMMOE_PKG_ROOT}: {exc}",
          file=sys.stderr)
    print("[fatal] point STREAMMOE_PKG_ROOT at the directory containing "
          "streammoe_bench/, or install the package.", file=sys.stderr)
    sys.exit(2)


# --------------------------------------------------------------------------- #
# Configuration matrix — Layers composed on top of validated prod flags
# --------------------------------------------------------------------------- #

@dataclass
class ConfigLayers:
    name: str
    extra_flags: list = field(default_factory=list)


LAYERS = [
    ConfigLayers(name="baseline",        extra_flags=[]),
    ConfigLayers(name="L1_eager",        extra_flags=["--moe-eager-load"]),
    ConfigLayers(name="L1L2_warmup",     extra_flags=["--moe-eager-load", "--streammoe-warmup"]),
    ConfigLayers(name="L1L2L3L4_full",   extra_flags=["--moe-eager-load", "--streammoe-warmup",
                                                      "--moe-keep-warm", "60"]),
]


def production_config_for(model: ModelDef) -> BenchConfig:
    """The canonical 'best-throughput' config from streammoe_bench phase2c.

    Matches the PROJECT_OVERVIEW 'Validated production config' block:
    slot-bank streaming + temporal prefetch + io_split=4 + f16 KV cache.
    slot_bank gets model-specific default (128 for BF16 to keep Metal
    allocation inside 48 GiB, 256 for Q4 where it fits comfortably).
    """
    # Slot-bank ceiling depends on how much the routed-expert footprint
    # weighs. BF16 experts are 4x Q4 experts per slot; 128 slots of BF16
    # fit on a 48 GiB Mac, 256 do not. This mirrors what runner.py's
    # phase2c config uses in production.
    default_slot_bank = 128 if model.key in ("qwen36bf16",) else 256
    return BenchConfig(
        name="prod-best-tps",
        label=f"production best-tps ({model.label})",
        description="Phase-2C validated best-throughput config",
        params=ConfigParams(
            slot_bank=default_slot_bank,
            prefetch="temporal",   # THE important one — keeps slot-bank warm
            io_split=4,
            kv_quant="f16",
            ctx=4096,              # TTFT harness uses short prompts
            topk=8,
        ),
        extra_server_args=["--flash-attn", "on"],
    )


# Models to run from the production registry. User can override via --models.
DEFAULT_MODEL_KEYS = ["qwen36", "qwen36bf16"]


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #

def wait_ready(port: int, deadline: float) -> bool:
    """Two-stage readiness: socket + /streammoe/status ok:true.

    The status endpoint flips ok only when ctx_http.is_ready.store(true)
    has been called after the model finished loading. Without the second
    stage we'd hit HTTP 503 on the first completion call because the port
    accepts before the model is ready."""
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


def start_server(binary: str, model: ModelDef, config: BenchConfig, layer: ConfigLayers,
                 port: int, log_path: Path) -> subprocess.Popen:
    # `build_extra_args` returns the validated production flag list; we
    # add model + host/port + the specific patch layers under test.
    prod_flags = build_extra_args(config, model)
    cmd = [
        binary,
        "-m", str(model.model_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-ngl", "99",
    ] + prod_flags + layer.extra_flags
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    # Record the exact argv for auditability.
    (log_path.with_suffix(".cmd")).write_text(" ".join(cmd) + "\n")
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def process_rss_bytes(pid: int) -> int:
    try:
        return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])) * 1024
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# TTFT + decode-throughput measurement
# --------------------------------------------------------------------------- #

def measure_request(port: int, prompt: str, n_predict: int = 4,
                    timeout_s: float = 60.0) -> dict:
    """Returns {ttft_s, decode_tokps, total_s}.

    ttft_s  = wall-clock to first SSE chunk with a non-empty text field
    decode_tokps = tokens-per-second during the decode phase
                   (tokens after the first divided by elapsed-after-first)
    """
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    t_first = None
    tokens_seen = 0
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for line in resp:
            if line.startswith(b"data:") and b'"text"' in line:
                if t_first is None:
                    t_first = time.perf_counter()
                tokens_seen += 1
    t_end = time.perf_counter()
    ttft = (t_first - t0) if t_first is not None else (t_end - t0)
    decode_s = max(0.0, t_end - (t_first or t_end))
    decode_tokps = (tokens_seen - 1) / decode_s if decode_s > 0 and tokens_seen > 1 else None
    return {"ttft_s": ttft, "decode_tokps": decode_tokps, "total_s": t_end - t0,
            "tokens": tokens_seen}


# --------------------------------------------------------------------------- #
# Matrix driver
# --------------------------------------------------------------------------- #

def run_matrix(binary: str, output_dir: Path, n_iter: int, prompt: str,
               n_predict: int, model_keys: list, layer_names: list) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"generated_at": time.time(), "binary": binary, "runs": []}
    checkpoint = output_dir / "ttft_matrix_in_progress.json"

    def save_checkpoint():
        checkpoint.write_text(json.dumps(results, indent=2))

    all_models = get_models()
    selected_models = [all_models[k] for k in model_keys if k in all_models]
    if not selected_models:
        raise SystemExit(f"no matching models in {list(all_models.keys())}")
    selected_layers = [l for l in LAYERS if (not layer_names or l.name in layer_names)]

    for model in selected_models:
        config = production_config_for(model)
        if not model.model_path.exists():
            print(f"[skip] {model.key}: {model.model_path} missing", file=sys.stderr)
            continue
        for layer in selected_layers:
            port = 11500 + (abs(hash(layer.name + model.key)) % 80)
            log_path = output_dir / f"server_{model.key}_{layer.name}.log"
            print(f"\n=== {model.key} / {layer.name} (port {port}) ===", flush=True)

            t_start = time.perf_counter()
            proc = start_server(binary, model, config, layer, port, log_path)
            deadline = 900 if "bf16" in model.key else 360
            if not wait_ready(port, time.monotonic() + deadline):
                print(f"[error] {model.key}/{layer.name} did not become ready", file=sys.stderr)
                stop_server(proc); continue
            startup_s = time.perf_counter() - t_start
            time.sleep(2.0)  # let warmup thread finish if any

            cold, warm = [], []
            try:
                cold.append(measure_request(port, prompt, n_predict=n_predict))
                for _ in range(n_iter - 1):
                    warm.append(measure_request(port, prompt, n_predict=n_predict))
            except Exception as e:
                print(f"[error] request phase: {e}", file=sys.stderr)

            rss = process_rss_bytes(proc.pid)
            stop_server(proc)

            def stat(xs, key):
                vs = [x[key] for x in xs if x.get(key) is not None]
                return statistics.median(vs) if vs else None

            results["runs"].append({
                "model": model.key,
                "model_label": model.label,
                "config": layer.name,
                "production_flags": build_extra_args(config, model),
                "layer_flags": layer.extra_flags,
                "startup_s": startup_s,
                "rss_bytes": rss,
                "cold": cold,
                "warm": warm,
                "cold_ttft_p50": stat(cold, "ttft_s"),
                "warm_ttft_p50": stat(warm, "ttft_s"),
                "cold_decode_tokps": stat(cold, "decode_tokps"),
                "warm_decode_tokps": stat(warm, "decode_tokps"),
            })
            save_checkpoint()
            time.sleep(1.0)

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary",
                    default="/Users/claude/streammoe/anemll-flash-llama.cpp/build/bin/llama-server")
    ap.add_argument("--output-dir", default="./ttft-results")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=8,
                    help="tokens to generate per request (need >1 to measure decode tok/s)")
    ap.add_argument("--prompt", default="Briefly: what is 2+2?")
    ap.add_argument("--models", default=",".join(DEFAULT_MODEL_KEYS),
                    help="comma-separated keys from streammoe_bench.config.get_models()")
    ap.add_argument("--layers", default="",
                    help="comma-separated subset from {baseline,L1_eager,L1L2_warmup,L1L2L3L4_full}")
    args = ap.parse_args()

    out = Path(args.output_dir)
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    layer_names = [l.strip() for l in args.layers.split(",") if l.strip()]
    results = run_matrix(args.binary, out, args.iters, args.prompt,
                         args.n_predict, model_keys, layer_names)
    path = out / f"ttft_matrix_{int(time.time())}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {path}")

    print(f"\n{'config':20s} {'model':14s} {'startup':>9s} {'cold_ttft':>10s} "
          f"{'warm_ttft':>10s} {'warm_tokps':>12s} {'rss_gib':>9s}")
    for r in results["runs"]:
        print(f"{r['config']:20s} {r['model']:14s} "
              f"{r['startup_s']:>9.2f} {r.get('cold_ttft_p50') or float('nan'):>10.3f} "
              f"{r.get('warm_ttft_p50') or float('nan'):>10.3f} "
              f"{(r.get('warm_decode_tokps') or 0):>12.2f} "
              f"{r['rss_bytes']/1024**3:>9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
