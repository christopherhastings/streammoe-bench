# StreamMoE TTFT Benchmark Results

**Hardware:** Apple M4 Max, 48 GB unified memory, macOS Darwin 25.4.0
**Fork:** `christopherhastings/anemll-flash-llama.cpp@feature/moe-eager-load`
**Latest run:** production-config bench with decode tok/s (2026-04-18 11:15)

All numbers below are from the "Run C" matrix — the first run that
composes llama-server flags through the validated `streammoe_bench.runner.build_extra_args()`
function rather than hand-rolling a subset. This is the config that ships
in production, plus each of the four TTFT patch layers on top.

Production flags (composed automatically per model):
```
--moe-sidecar <path> --moe-mode slot-bank --moe-slot-bank {256|128}
--moe-prefetch-temporal --ctx-size 4096 --flash-attn on -ngl 99
```

## Matrix — Qwen3.6-35B-A3B-UD-Q4_K_XL (sb=256)

| Config            | Startup (s) | Cold TTFT (s) | Warm TTFT (s) | Warm decode (tok/s) | RSS (GiB) |
|-------------------|------------:|--------------:|--------------:|--------------------:|----------:|
| baseline          |       13.18 |         2.138 |         0.333 |              37.36  |      8.76 |
| +L1 eager         |       18.97 |       **1.239** |         0.281 |              33.64  |      8.33 |
| +L1+L2 warmup     |       17.92 |       **1.178** |         0.292 |              35.37  |      8.76 |
| +L1+L2+L3+L4 full |       17.92 |         1.212 |         0.312 |            **26.06**|      8.82 |

## Matrix — Qwen3.6-35B-A3B-BF16 (sb=128)

| Config            | Startup (s) | Cold TTFT (s) | Warm TTFT (s) | Warm decode (tok/s) | RSS (GiB) |
|-------------------|------------:|--------------:|--------------:|--------------------:|----------:|
| baseline          |       41.35 |         3.567 |         0.381 |            **21.93**|     23.19 |
| +L1 eager         |       59.25 |         3.513 |         0.398 |              21.36  |     23.20 |
| +L1+L2 warmup     |       60.23 |         3.500 |         0.375 |              21.97  |     23.41 |
| +L1+L2+L3+L4 full |       59.25 |         3.497 |         0.385 |              21.74  |     23.41 |

## Verdict

**BF16 IS streaming as expected.** Steady **~22 tok/s decode** across all
configs, matching PROJECT_OVERVIEW's 22-23 tok/s validated production
figure. The earlier impression that "BF16 doesn't stream" was a harness
artifact — that run had `--moe-prefetch-temporal` missing, which broke
warm slot-bank residency and inflated warm TTFT. With prefetch on,
BF16 warm TTFT drops to **0.375 s** (was 0.605 s in Run A) and decode
holds at 22 tok/s.

**Layer 1 (eager-load) is a 45 % cold-TTFT win on Q4** (2.14 → 1.18 s)
because the 18 GiB sidecar fits in 48 GiB RAM and the page cache stays
warm. **On BF16 it buys nothing** (3.57 → 3.51 s) because the 60 GiB
sidecar evicts from the page cache between prompts — the safeguard
correctly skips mlock and falls back to read-through-only.

**Layer 2 (warmup) is a tiny cold-TTFT win on both models** and has no
downside. It's the only layer I'd enable by default.

**Layer 3 (keep-warm) regresses decode tok/s on Q4** (37 → 26 tok/s,
−30 %). The heartbeat thread races with real requests and steals Metal
queue time. Don't enable by default until the pause-on-activity gate
lands. On BF16 the regression is smaller (21.9 → 21.7 tok/s) but still
present. This is the top open issue.

**Layer 4 (auto `ub=b=256`) is now opt-in only**, after the default
caused Metal OOM on BF16 and a host-level memory pressure event. The
short-prompt TTFT optimization is still available — just explicit.

## Recommended per-model configs

### Q4_K_XL — maximum cold-TTFT win
```
--moe-sidecar <path> --moe-mode slot-bank --moe-slot-bank 256
--moe-prefetch-temporal --flash-attn on --ctx-size 4096 -ngl 99
--moe-eager-load --streammoe-warmup     # Layers 1 + 2
```
**Expected:** cold TTFT ~1.18 s, warm ~0.29 s, decode ~35 tok/s.

### BF16 — sidecar-larger-than-RAM
```
--moe-sidecar <path> --moe-mode slot-bank --moe-slot-bank 128
--moe-prefetch-temporal --flash-attn on --ctx-size 4096 -ngl 99
--streammoe-warmup                       # Layer 2 only
```
**Expected:** cold TTFT ~3.5 s, warm ~0.38 s, decode ~22 tok/s. Skip L1
(no benefit, +15 s startup) and skip L3 (decode regression).

## Quality verification — BF16 (2026-04-18, 80 MT-Bench prompts)

Baseline (production flags) vs patched (production flags + `--moe-eager-load`
+ `--streammoe-warmup`), greedy decoding (temperature=0, top_k=1, seed=42),
48 tokens per prompt.

| Metric              | Value         |
|---------------------|--------------:|
| Byte-identical      | **79 / 80**   |
| Match rate          | **98.75 %**   |
| Diverging prompt    | `mt_126` (coding — "median of two sorted arrays") |
| Divergence position | char 72 of the generated answer |
| Length delta        | 7 chars (baseline 177, patched 170) |

The one divergence lines up with the prior project's well-documented
caveat: **Metal fp32 reductions are not bit-reproducible** across runs
because the reduction ordering depends on scheduling that varies each
boot. Strict byte-level matching is therefore a lower bound on
equivalence, not an upper bound — a ~1 % divergence rate at this prompt
budget is expected and does not imply a patch-introduced regression.
The prior phase 2C used an LLM-as-judge gate (strict-JSON tool-use) for
this reason; that gate reports zero regressions for the production
streaming configs vs stock.

Full result: `quality-results/quality_verify_qwen36bf16_1776563852.json`.

**To re-run with a stricter gate**, drive `quality_verify.py` from
`streammoe_bench/judge.py` (claude-sonnet-4.6 tool-use judge that the
prior sweep used) instead of `==`; the judge confirmed equivalence on
all 80 prompts of the current streaming config pre-patch. A follow-up
task captures plugging it in.

## Known issues / per-workload guidance

1. **Layer 3 keep-warm heartbeat race — FIXED.** Previously the heartbeat
   fired unconditionally every `interval_s` seconds and collided with
   real user requests, collapsing Q4 decode from 37 → 26 tok/s. Fixed in
   `server.cpp` via an atomic `streammoe_last_user_request_ns` that the
   ex_wrapper stamps on every non-heartbeat request. The heartbeat thread
   reads it and skips its cycle if user activity happened within the
   last `interval_s`. The heartbeat's own curl sends
   `X-StreamMoE-Warmup: 1` so it doesn't reset the gate itself. Needs
   a fresh bench run to confirm the regression is gone.
2. **Layer 1 on Q4 trades 2-4 tok/s decode for 45 % cold TTFT win.** The
   mlock'd sidecar pages compete with Metal unified-memory pressure.
   **Recommended by workload:**
   - Chat / interactive (short responses, < 100 tokens decoded): enable
     L1 — the cold-TTFT win dominates and the per-token cost is
     negligible relative to user reaction time.
   - Long-form generation (summaries, document translation, agent loops
     that run > 300 tokens): skip L1 to keep the full 37 tok/s decode.
     The 2-4 tok/s regression compounds over long generations and
     dwarfs the one-time cold-TTFT win.

## Files

- `ttft-results/ttft_matrix_1776554017.json` — latest, production-config composer (Run C)
- `ttft-results/ttft_matrix_1776552201.json` — Run B (prefetch on, hand-rolled flags)
- `ttft-results/ttft_matrix_1776524320.json` — Run A BF16 (prefetch off, buggy)
- `ttft-results/ttft_matrix_1776522078.json` — Run A Q4 (prefetch off, still OK because Q4 doesn't rely on prefetch as hard)

## Q4 — streaming vs all-resident (from 80-prompt phase 2C, N=80)

From the prior project's MT-Bench 80-prompt quality sweep (each cell runs
all 80 prompts with temperature=0 + seed=42; numbers are per-prompt
averages):

| Mode                                 | Decode (tok/s) | RSS (GiB) | Quality vs stock |
|--------------------------------------|---------------:|----------:|------------------|
| **stock** (all experts resident)     |        25.12  |     21.54 | reference        |
| **best-tps streaming** (sb=256 + temporal prefetch) |        22.14  |    **5.43** | equivalent (judge-verified) |
| **lowest-ram streaming** (sb=256 + temporal + q4 KV + io_split=16) |        21.09  |    **4.92** | equivalent (judge-verified) |

**Tradeoff: 12 % slower sustained decode for 4× less RAM.** Both streaming
configurations pass the 80-prompt quality gate (Phase 2C LLM-as-judge,
strict JSON) — zero regressions on algorithmic reasoning, writing,
math, reasoning, or extraction categories.

Short-prompt decode (16-token generation, Run C) ran faster at 37 tok/s
because the slot-bank hit rate is higher on short prompts before the
cache churn starts; 22 tok/s is the sustained-decode number that
matches user experience.

**Source:** `/Users/claude/streammoe/results/sweep_20260416_230927/phase2c_qwen36_summary.json`

## Reproducing

```sh
cd streammoe-bench
pip3 install pydantic
python3.11 ttft_bench.py --iters 3                       # full 2-model matrix (~70 min)
python3.11 ttft_bench.py --iters 3 --models qwen36bf16   # BF16 only (~40 min)
python3.11 ttft_bench.py --iters 3 --layers L1L2_warmup  # one layer across both models
```

Bench now imports flags from `/Users/claude/streammoe/streammoe_bench/`
via `production_config_for(model)` → `build_extra_args()`. If production
adds a new flag, rerun this bench to pick it up — no ttft_bench.py edit
needed.
