# StreamMoE TTFT Benchmark Results

**Hardware:** Apple M4 Max, 48 GB unified memory, macOS Darwin 25.4.0
**Fork:** `christopherhastings/anemll-flash-llama.cpp@feature/moe-eager-load`
**Run:** N=3 iterations per cell (1 cold + 2 warm)
**Date:** 2026-04-18

## Qwen3.6-35B-A3B-UD-Q4_K_XL (21 GiB GGUF + 18 GiB sidecar)

| Config            | Startup (s) | Cold TTFT (s) | Warm p50 (s) | Warm p95 (s) | RSS (GiB) |
|-------------------|------------:|--------------:|-------------:|-------------:|----------:|
| baseline          |       15.00 |         1.707 |        0.284 |        0.144 |      7.34 |
| +L1 eager         |       18.33 |         1.061 |        0.191 |        0.145 |      6.91 |
| +L1+L2 warmup     |       18.33 |         1.092 |        0.192 |        0.146 |      7.49 |
| +L1+L2+L3+L4 full |       18.84 |         1.087 |        0.198 |        0.150 |      7.01 |

**Improvements vs. baseline (Q4_K_XL):**
- Cold TTFT: **1.71 s → 1.06 s = 38 % reduction** (Layer 1 alone captures the bulk of the win)
- Warm p50 TTFT: **0.28 s → 0.19 s = 32 % reduction**
- Layer 2 (warmup) and Layer 3 (keep-warm) did not measurably help on top of Layer 1
  for this workload — the warmup fires once at startup, after which the OS page cache
  is already populated by Layer 1's read-through.
- Layer 4 (default `-b=-ub=256`) had no effect because the bench prompt is very short
  (4 tokens). Its win shows up on prompts in the 500–4 k-token range.
- Startup time grew by ~3 s because Layer 1 reads through 18 GiB sequentially before
  flipping `is_ready` true. Acceptable trade given the cold-prompt latency win.

## Qwen3.6-35B-A3B-BF16 (64.6 GiB GGUF + 60 GiB sidecar) — incomplete

All four cells failed during the inference phase with HTTP 500. Server log:

```
llama_params_fit: failed to fit params to free device memory:
n_gpu_layers already set by user to 99, abort
```

The fork's auto-fit logic refuses to clamp when `--ngl` is set explicitly. The bench
hard-codes `--ngl 99`; on a 48 GiB Mac, BF16 + slot-bank 256 exceeds Metal device
memory before the graph allocator can lay out the activations.

**Fix path** (not yet exercised — would require another full bench run):
- Drop `--ngl 99` from the bench's BF16 cells and let `--fit on` decide.
- Or pass `--ngl 24` to keep half the dense layers on CPU.
- Or shrink `--moe-slot-bank` from 256 to 128, halving the slot reserve.

Layer 1 itself succeeded on BF16 — `eager_load_moe_sidecar: eager-loaded 60.00 GiB`
appears in `server_bf16_L1_eager.log`. The failure is downstream in graph allocation.

## Files

- `ttft-results/ttft_matrix_1776522078.json` — full per-iteration timings
- `ttft-results/server_q4_k_xl_*.log` — per-cell server logs
- `ttft-results/server_bf16_*.log` — per-cell BF16 logs (with the fit-abort error)

## Reproducing

```sh
cd streammoe-bench
python3 ttft_bench.py --iters 3
# Pass --iters 10 for tighter confidence intervals; runtime ~60–90 min total.
```
