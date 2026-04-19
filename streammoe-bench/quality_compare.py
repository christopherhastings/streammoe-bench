#!/usr/bin/env python3
"""Multi-config quality comparison on 80 MT-Bench prompts.

Runs each configured server back-to-back against the same 80-prompt
MT-Bench suite (greedy decoding, seed=42, ctx=8192, n_predict=1024 so
nothing truncates), saves every full response, then feeds the requested
pairwise comparisons to the claude-sonnet-4-6 judge.

Default config set (over Qwen3.6-35B-A3B on the fork at ~/streammoe/...):
    q4_resident    GGUF Q4_K_XL, mode stock, all experts resident in RAM.
    q4_streaming   GGUF Q4_K_XL, slot-bank sb=256, temporal prefetch.
    bf16_streaming GGUF BF16,   slot-bank sb=128, temporal prefetch.

Default pairs judged:
    streaming_vs_resident   q4_streaming vs q4_resident
                            → does slot-bank streaming degrade quality?
    q4_vs_bf16_streaming    q4_streaming vs bf16_streaming
                            → does Q4 quant vs BF16 reference differ in practice?
    resident_vs_bf16        q4_resident vs bf16_streaming
                            → the full stock-vs-streaming-reference contrast.

Designed to be killable: intermediate outputs and per-prompt judge verdicts
are written to disk after each step, so a crash or Ctrl-C preserves work.

Usage:
    export ANTHROPIC_API_KEY=...
    python3.11 quality_compare.py                       # full default run
    python3.11 quality_compare.py --skip-sampling       # only re-judge saved outputs
    python3.11 quality_compare.py --only q4_streaming   # sample just one config
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


# --------------------------------------------------------------------------- #
# Configs + judge pairs
# --------------------------------------------------------------------------- #

QWEN36_Q4  = "qwen36"      # Qwen3.6-35B-A3B-UD-Q4_K_XL
QWEN36_BF16 = "qwen36bf16" # Qwen3.6-35B-A3B-BF16

# Each config is a (name, model_key, flag_builder). The flag_builder runs
# AFTER the base -m / --host / --port / --ctx-size / -ngl so you can add
# streaming / sidecar flags without re-specifying the basics.
def q4_resident_flags(model):
    # Stock all-resident: no sidecar, no slot-bank. Fits in 48 GiB for Q4.
    return []

def q4_streaming_flags(model):
    return [
        "--moe-sidecar", str(model.sidecar_dir),
        "--moe-mode", "slot-bank", "--moe-slot-bank", "256",
        "--moe-prefetch-temporal",
        "--flash-attn", "on",
    ]

def bf16_streaming_flags(model):
    return [
        "--moe-sidecar", str(model.sidecar_dir),
        "--moe-mode", "slot-bank", "--moe-slot-bank", "128",
        "--moe-prefetch-temporal",
        "--flash-attn", "on",
    ]

CONFIGS = {
    "q4_resident":    {"model_key": QWEN36_Q4,   "flags": q4_resident_flags,
                       "label": "GGUF Q4_K_XL resident (stock)"},
    "q4_streaming":   {"model_key": QWEN36_Q4,   "flags": q4_streaming_flags,
                       "label": "GGUF Q4_K_XL slot-bank streaming (sb=256)"},
    "bf16_streaming": {"model_key": QWEN36_BF16, "flags": bf16_streaming_flags,
                       "label": "GGUF BF16 slot-bank streaming (sb=128)"},
}

# Judge pairs — each (a_name, b_name) sends A as "stock" and B as "sidecar"
# to the claude-sonnet-4-6 judge. The judge is neutral; the labels are
# just for its prompt template. verdict "different" means they diverged,
# not which is better.
JUDGE_PAIRS = [
    ("q4_streaming",   "q4_resident",     "streaming_vs_resident"),
    ("q4_streaming",   "bf16_streaming",  "q4_vs_bf16_streaming"),
    ("q4_resident",    "bf16_streaming",  "resident_vs_bf16"),
]


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #

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


def start_server(binary: str, model, flags: list, port: int, log_path: Path,
                 ctx_size: int) -> subprocess.Popen:
    cmd = [
        binary,
        "-m", str(model.model_path),
        "--host", "127.0.0.1", "--port", str(port),
        "--ctx-size", str(ctx_size),
        "-ngl", "99",
        "--seed", "42",
    ] + flags
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
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def sample(port: int, prompt: str, n_predict: int, timeout_s: float = 600.0) -> dict:
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
        data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode())
    t_elapsed = time.perf_counter() - t0
    timings = data.get("timings", {})
    return {
        "content": data.get("content", ""),
        "n_tokens": timings.get("predicted_n", 0),
        "decode_tps": timings.get("predicted_per_second"),
        "ttft_s": (timings.get("prompt_ms", 0) / 1000.0) if timings.get("prompt_ms") else None,
        "wall_s": t_elapsed,
    }


def run_config(name: str, binary: str, output_dir: Path, prompts: list,
               n_predict: int, ctx_size: int) -> dict:
    cfg = CONFIGS[name]
    model = get_models()[cfg["model_key"]]
    flags = cfg["flags"](model)
    port = 11500 + (abs(hash(name)) % 80)
    log_path = output_dir / f"server_{name}.log"

    print(f"\n=== sampling {name} ({cfg['label']}) on port {port} ===", flush=True)
    t_load = time.perf_counter()
    proc = start_server(binary, model, flags, port, log_path, ctx_size)
    deadline = 900 if "bf16" in name else 360
    if not wait_ready(port, time.monotonic() + deadline):
        stop_server(proc)
        return {"name": name, "error": "server did not become ready", "label": cfg["label"]}
    load_s = time.perf_counter() - t_load
    time.sleep(2.0)

    responses = []
    try:
        for i, p in enumerate(prompts):
            try:
                r = sample(port, p["prompt"], n_predict)
                r.update({"prompt_id": p["id"], "category": p.get("category"),
                          "prompt": p["prompt"]})
                responses.append(r)
            except Exception as e:
                print(f"  [err] prompt {i} ({p.get('id')}): {e}", file=sys.stderr)
                responses.append({"prompt_id": p["id"], "category": p.get("category"),
                                  "prompt": p["prompt"], "error": str(e),
                                  "content": "", "n_tokens": 0})
            if i % 10 == 0:
                print(f"  {name} {i+1}/{len(prompts)}  "
                      f"last: n_tok={responses[-1].get('n_tokens')} "
                      f"wall={responses[-1].get('wall_s',0):.1f}s", flush=True)
            # Incremental save so we don't lose progress on crash.
            (output_dir / f"responses_{name}.json").write_text(
                json.dumps({"name": name, "label": cfg["label"],
                            "load_s": load_s, "flags": flags,
                            "responses": responses}, indent=2))
    finally:
        stop_server(proc)
    return {"name": name, "label": cfg["label"], "load_s": load_s,
            "flags": flags, "responses": responses}


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #

def judge(pair_name: str, a_data: dict, b_data: dict, model: str) -> dict:
    try:
        import anthropic
        from streammoe_bench.quality_gates import judge_pair
    except ImportError as e:
        return {"pair": pair_name, "error": f"import: {e}"}
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"pair": pair_name, "error": "ANTHROPIC_API_KEY not set"}
    client = anthropic.Anthropic(api_key=api_key)

    print(f"\n=== judging pair: {pair_name}  ({a_data['label']}  vs  {b_data['label']}) ===",
          flush=True)
    a_by_id = {r["prompt_id"]: r for r in a_data["responses"]}
    b_by_id = {r["prompt_id"]: r for r in b_data["responses"]}
    common = [pid for pid in a_by_id if pid in b_by_id]
    verdicts = []
    counts = {"same": 0, "similar": 0, "different": 0, "parse_error": 0}
    for i, pid in enumerate(common):
        a = a_by_id[pid]
        b = b_by_id[pid]
        if a.get("error") or b.get("error"):
            verdicts.append({"prompt_id": pid, "skipped": True})
            continue
        v = judge_pair(client, a["prompt"], a["content"], b["content"], model=model)
        verdicts.append({"prompt_id": pid, "category": a.get("category"), **v})
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
        if i % 10 == 0:
            print(f"  {pair_name} {i+1}/{len(common)}: {v['verdict']} "
                  f"(same={counts['same']} sim={counts['similar']} diff={counts['different']})",
                  flush=True)
    return {"pair": pair_name, "counts": counts, "verdicts": verdicts,
            "a": a_data["name"], "b": b_data["name"]}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary",
                    default="/Users/claude/streammoe/anemll-flash-llama.cpp/build/bin/llama-server")
    ap.add_argument("--output-dir", default="./quality-results/compare")
    ap.add_argument("--prompts", default="mtbench80.jsonl")
    ap.add_argument("--ctx-size", type=int, default=8192)
    ap.add_argument("--n-predict", type=int, default=1024)
    ap.add_argument("--only", default="", help="only run this config (comma list ok)")
    ap.add_argument("--skip-sampling", action="store_true",
                    help="don't spawn servers — re-judge existing responses_*.json")
    ap.add_argument("--judge-model", default="claude-sonnet-4-6")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    prompts_file = Path(args.prompts)
    if not prompts_file.exists():
        prompts_file = Path(__file__).parent / args.prompts
    prompts = []
    with prompts_file.open() as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    only = {s.strip() for s in args.only.split(",") if s.strip()}

    sampled = {}
    for name in CONFIGS:
        if only and name not in only: continue
        path = out / f"responses_{name}.json"
        if args.skip_sampling:
            if path.exists():
                sampled[name] = json.loads(path.read_text())
                print(f"[reuse] {name} — {len(sampled[name]['responses'])} responses")
            else:
                print(f"[skip] {name} — no cached file at {path}", file=sys.stderr)
            continue
        sampled[name] = run_config(name, args.binary, out, prompts,
                                    args.n_predict, args.ctx_size)

    # Judge each declared pair for which both sides are available.
    judged = []
    for a, b, pair_name in JUDGE_PAIRS:
        if a not in sampled or b not in sampled: continue
        result = judge(pair_name, sampled[a], sampled[b], args.judge_model)
        judged.append(result)
        (out / f"judge_{pair_name}.json").write_text(json.dumps(result, indent=2))

    final = {"generated_at": time.time(),
             "ctx_size": args.ctx_size, "n_predict": args.n_predict,
             "configs": {name: {"label": s.get("label"),
                                "flags": s.get("flags"),
                                "load_s": s.get("load_s"),
                                "n_responses": len(s.get("responses", []))}
                         for name, s in sampled.items()},
             "judged": judged}
    out_path = out / f"compare_{int(time.time())}.json"
    out_path.write_text(json.dumps(final, indent=2))
    print(f"\nResults: {out_path}")
    for j in judged:
        if "counts" in j:
            c = j["counts"]
            total = sum(c.values())
            equiv = c.get("same", 0) + c.get("similar", 0)
            print(f"  {j['pair']:28s} same={c.get('same',0):>2}  "
                  f"similar={c.get('similar',0):>2}  "
                  f"different={c.get('different',0):>2}  "
                  f"equivalent={equiv}/{total}  ({100*equiv/max(1,total):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
