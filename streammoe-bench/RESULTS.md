# StreamMoE TTFT Benchmark Results

**Hardware:** Apple M4 Max, 48 GB unified memory, macOS Darwin 25.4.0
**Fork:** `christopherhastings/anemll-flash-llama.cpp@feature/moe-eager-load`
**Run:** N=3 iterations per cell (1 cold + 2 warm)
**Dates:** Q4 2026-04-17 · BF16 2026-04-18

## Qwen3.6-35B-A3B-UD-Q4_K_XL (21 GiB GGUF + 18 GiB sidecar, sb=256)

| Config            | Startup (s) | Cold TTFT (s) | Warm p50 (s) | Warm p95 (s) | RSS (GiB) |
|-------------------|------------:|--------------:|-------------:|-------------:|----------:|
| baseline          |       15.00 |         1.707 |        0.284 |        0.144 |      7.34 |
| +L1 eager         |       18.33 |         1.061 |        0.191 |        0.145 |      6.91 |
| +L1+L2 warmup     |       18.33 |         1.092 |        0.192 |        0.146 |      7.49 |
| +L1+L2+L3+L4 full |       18.84 |         1.087 |        0.198 |        0.150 |      7.01 |

**Wins on Q4_K_XL:**
- Cold TTFT **1.71 s → 1.06 s = 38 % reduction** (Layer 1 captures the win).
- Warm p50 **0.28 s → 0.19 s = 32 % reduction**.
- 18 GiB sidecar fits comfortably in 48 GiB RAM, so Layer 1's page-cache
  warmth survives between prompts.

## Qwen3.6-35B-A3B-BF16 (64.6 GiB GGUF + 60 GiB sidecar, sb=128)

| Config            | Startup (s) | Cold TTFT (s) | Warm p50 (s) | Warm p95 (s) | RSS (GiB) |
|-------------------|------------:|--------------:|-------------:|-------------:|----------:|
| baseline          |       42.73 |         3.633 |        0.672 |        0.531 |     18.63 |
| +L1 eager         |       58.39 |         3.657 |        0.675 |        0.538 |     18.63 |
| +L1+L2 warmup     |       59.36 |         3.571 |        0.605 |        0.424 |     18.90 |
| +L1+L2+L3+L4 full |       61.14 |         3.506 |        0.998 |        0.839 |     18.88 |

**Findings on BF16:**
- **Layer 1 does nothing** (cold 3.63 → 3.66 s, warm 0.67 → 0.68 s). The
  60 GiB sidecar exceeds 48 GiB physical RAM, so the OS evicts the
  read-through bytes before the first prompt arrives. The safeguard
  correctly refuses to mlock (sidecar > 60 % of RAM).
- **Layer 2 helps** (warm p50 0.67 → 0.60 s, ~10 % win; warm p95 0.53 → 0.42 s,
  ~20 % win). Graph / shader / sampler warmup pays for itself even when
  the page cache can't be kept warm.
- **Layer 3 regresses warm p50** (0.60 → 0.998 s). The keep-warm heartbeat
  fires during the measurement window and steals Metal command-queue time
  from the user's warm request. Needs a pause-on-activity gate — see
  Known issues.
- Startup grows +15.7 s under Layer 1 (60 GiB sequential read) — the
  page-cache investment that doesn't pay back on this hardware.

**Recommended config on BF16 for this class of hardware (sidecar > RAM):**
```
--moe-mode slot-bank --moe-slot-bank 128
--streammoe-warmup             # Layer 2 only
--ngl 99 -c 4096
```
Skip `--moe-eager-load` and `--moe-keep-warm` — they either cost startup
for no TTFT gain (L1) or actively regress warm TTFT (L3).

## Known issues captured by this run

1. **Layer 3 keep-warm races with user requests.** The 60 s heartbeat fires
   on a detached thread and blocks the Metal queue. Fix: skip the heartbeat
   if a real request completed within the last `interval_s`.
2. **Layer 4 (auto `-ub=b=256`) must stay opt-in** on large-sidecar BF16
   configs — initial matrix run (before safeguard) OOM'd Metal on `--ngl 99`.
3. **Layer 1 is hardware-dependent.** On machines where sidecar ≤ RAM
   it's a 38 % cold-TTFT win; on machines where sidecar > RAM it's a net
   startup-latency loss with no TTFT payback.

## Files

- `ttft-results/ttft_matrix_1776522078.json` — Q4 matrix (2026-04-17)
- `ttft-results/ttft_matrix_1776524320.json` — BF16 matrix with sb=128 (2026-04-18)
- `ttft-results/server_*.log` — per-cell server logs

## Reproducing

```sh
cd streammoe-bench
python3 ttft_bench.py --iters 3                    # full 4×2 matrix (~90 min)
python3 ttft_bench.py --iters 3 --models bf16      # just BF16 (~40 min)
python3 ttft_bench.py --iters 3 --models q4_k_xl   # just Q4 (~20 min)
```
