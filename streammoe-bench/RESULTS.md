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

Two runs: first without `--moe-prefetch-temporal`, second with it. The
flag was missing from the bench — a harness bug — and fixing it was
correct. But it did NOT materially move TTFT, for a reason that matters:

> **TTFT (time to first token) measures cold first-token latency.
> Prefetch helps tokens 2+, not token 1.** The benchmark metric itself
> is the wrong tool for quantifying "streaming-after-warm" performance.

### Run A — prefetch-temporal OFF (harness bug, 2026-04-18 08:30)

| Config            | Startup (s) | Cold TTFT (s) | Warm p50 (s) | Warm p95 (s) | RSS (GiB) |
|-------------------|------------:|--------------:|-------------:|-------------:|----------:|
| baseline          |       42.73 |         3.633 |        0.672 |        0.531 |     18.63 |
| +L1 eager         |       58.39 |         3.657 |        0.675 |        0.538 |     18.63 |
| +L1+L2 warmup     |       59.36 |         3.571 |        0.605 |        0.424 |     18.90 |
| +L1+L2+L3+L4 full |       61.14 |         3.506 |        0.998 |        0.839 |     18.88 |

### Run B — prefetch-temporal ON (matches validated production config, 2026-04-18 10:30)

| Config            | Startup (s) | Cold TTFT (s) | Warm p50 (s) | Warm p95 (s) | RSS (GiB) |
|-------------------|------------:|--------------:|-------------:|-------------:|----------:|
| baseline          |       43.79 |         3.664 |        0.677 |        0.547 |     18.63 |
| +L1 eager         |       59.33 |         3.654 |        0.618 |        0.422 |     18.63 |
| +L1+L2 warmup     |       58.88 |         3.559 |        0.682 |        0.523 |     18.90 |
| +L1+L2+L3+L4 full |       59.88 |         3.496 |        0.704 |        0.553 |     18.92 |

**Findings on BF16 (both runs):**
- **Cold TTFT is ~3.5 s regardless of patch layer or prefetch.** The first
  inference has no prefetch history to pull from; its latency is dominated
  by pread'ing ~2.5 GiB of routed experts from the SSD. Layer 1 reads-through
  the sidecar but those bytes don't survive in the page cache on 48 GiB RAM,
  so cold-path pread has to re-hit the SSD.
- **Warm p50 and p95 are the metrics that improve** on BF16 — for most
  layer combinations they drop into the 0.4-0.7 s band. With prefetch
  on, L1 alone gives the best warm p95 (0.422 s).
- **Layer 3 regression observed in Run A** (warm p50 0.60 → 0.998 s) was
  the keep-warm heartbeat racing with real requests. With prefetch on
  in Run B it no longer regressed; differences are now within noise. The
  race is still theoretically there — pause-on-activity gate still
  recommended for production.
- **Startup +15.7 s under Layer 1** is the 60 GiB sequential read. On
  48 GiB hardware this startup cost buys only the warm path, since the
  cold path re-reads from SSD anyway.

### Why TTFT isn't the full picture

The user-facing streaming-throughput win for Flash-MoE is
**decode tokens/second**, not TTFT. Prefetch-temporal refreshes routed
experts into slots immediately after each token decodes; subsequent
tokens hit warm slots and stream at 22-23 tok/s (per PROJECT_OVERVIEW).
A dedicated decode-throughput bench cell would report this — the TTFT
harness does not.

**Follow-up work:** add `decode_tokps` to the bench harness. Run a
longer n_predict (256 tokens), measure steady-state tokens/sec during
the decode phase, and include it in the per-config summary. That's the
metric that directly answers "can BF16 stream at a usable rate."

### Recommended config on BF16 for this class of hardware (sidecar > RAM)

```
--moe-mode slot-bank --moe-slot-bank 128
--moe-prefetch-temporal         # keeps warm hit rate up during decode
--streammoe-warmup              # Layer 2; warms kernels + routing
--flash-attn on
--ngl 99 -c 4096
```

Skip `--moe-eager-load` — the 60 GiB read-through costs 15 s of startup
and doesn't survive page-cache eviction on 48 GiB RAM.
Skip `--moe-keep-warm` until the pause-on-activity gate lands.

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
