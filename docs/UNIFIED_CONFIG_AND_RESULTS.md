# StreamMoE — Unified Configuration & Results Matrix

**Hardware:** Apple M4 Max, 48 GB unified memory, macOS Darwin 25.4.0
**Model under test:** Qwen3.6-35B-A3B (any quantization; 35B-class, 3B-active MoE)
**Last updated:** 2026-04-18

This document is the one place to look when deciding *how* to run a model.
For every combination of engine × quantization × loading strategy we've
measured, you get: what flags to set, what you'll get in exchange, and
whether it passes our quality bar.

---

## Headline recommendation

| Role                       | Config                                                  | Why |
|----------------------------|---------------------------------------------------------|-----|
| **Production / ship**      | GGUF Q4_K_XL, slot-bank streaming sb=256 + `--streammoe-warmup` | 25 tok/s sustained, ~5 GB RAM, 1.2s cold TTFT, judge-verified quality parity |
| **Maximum chat UX**        | Same + `--moe-eager-load`                               | 45 % cold-TTFT reduction (2.14 s → 1.18 s) on any-RAM-fits workload |
| **Reference quality**      | GGUF BF16, slot-bank streaming sb=128                   | 13 tok/s sustained, byte-identical quality baseline |
| **Long-form generation**   | GGUF Q4_K_XL, slot-bank *without* L1                    | 37 tok/s peak decode; L1's 2-4 tok/s cost compounds over 300+ tokens |
| **DO NOT SHIP**            | MLX 4-bit resident                                      | 36 % "different" verdicts vs GGUF stock on STEM/coding/math |

---

## Full matrix

All numbers are 80-prompt MT-Bench decode averages where available,
otherwise the sample size is noted. TTFT is cold-first-token unless
marked "warm". Quality is the pairwise Claude-sonnet-4.6 tool-use judge
verdict distribution (`same / similar / different`) vs GGUF stock.

| Engine            | Quant     | Loading              | Decode tok/s | TTFT    | RSS     | Quality (same/sim/diff, n) |
|-------------------|-----------|----------------------|-------------:|---------|--------:|----------------------------|
| **GGUF (Flash-MoE fork)** | Q4_K_XL | **slot-bank stream** (sb=256, temporal prefetch) | **25.7**  (sustained) / 35-37 (16-tok) | 1.18 s (+L1+L2) / 2.14 s (baseline) | **5.3 GB** | **judge-verified equivalent, 0/80 regressions** |
| GGUF (Flash-MoE fork) | Q4_K_XL | stock (all-resident) | 25.12           | —        | 21.54 GB | baseline (identity)        |
| GGUF (Flash-MoE fork) | Q4_K_XL | slot-bank lowest-ram (sb=256 + q4 KV + io_split=16) | 21.09 | — | 4.92 GB | judge-verified equivalent |
| GGUF (Flash-MoE fork) | **BF16**   | **slot-bank stream** (sb=128, temporal) | **13.5** (sustained) / 22 (16-tok) | 3.50 s | **7.9 GB** (24 GB steady-state) | 15/5/0 same/sim/diff (n=20) — judge-equivalent to Q4 |
| GGUF (Flash-MoE fork) | BF16   | stock                | —             | —        | —        | doesn't fit on 48 GB       |
| GGUF (Flash-MoE fork) | Q5_K_M | slot-bank stream     | pending rerun345 | —      | ~6 GB   | pending                    |
| **MLX (SwiftLM)** | 4-bit   | resident (full GPU)  | 19.96–24.29    | **0.18-0.20 s** | 13.4 GB (8.3 GB peak req) | **13/38/29 same/sim/diff (n=80)** — 36 % different on STEM |
| MLX (SwiftLM)     | 4-bit   | `--stream-experts`   | 8.34           | 1.74 s   | 4.5 GB   | not judged (n=5)           |
| MLX (SwiftLM)     | BF16    | streaming            | 0.14 tok/s prefill | 22+ min | MEM_DEMAND 129 GB | not viable on 48 GB |
| Ollama (MLX)      | BF16    | memory               | refuses to load | —       | —        | "requires 65 GiB > 37 avail" |
| GGUF (fork)       | 397B Q4 | slot-bank stream     | 0.37           | —        | 6 GB     | not viable (SSD-bound)     |

Sources: GGUF numbers from `streammoe-bench/RESULTS.md` + prior project
phase 2C (`phase2c_qwen36_summary.json`). MLX numbers from
`sweep_mlx_20260417_184225` (80-prompt) and prior project's three-way
comparison. Note the two different decode numbers for slot-bank Q4 /
BF16 — sustained 300-token decoding hits a steady-state cache-churn
rate (25.7 / 13.5 tok/s); 16-token short-prompt decode is higher because
the slot bank hasn't saturated yet (35 / 22 tok/s).

---

## Configuration recipes

All examples target Qwen3.6-35B-A3B on M4 Max 48 GB. Adjust model paths.

### Recipe A — production streaming (Q4_K_XL)

```bash
~/streammoe/anemll-flash-llama.cpp/build/bin/llama-server \
  -m ~/Downloads/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --moe-sidecar ~/streammoe/models/qwen35-35b-sidecar \
  --moe-mode slot-bank --moe-slot-bank 256 \
  --moe-prefetch-temporal \
  --flash-attn on --ctx-size 4096 -ngl 99 \
  --streammoe-warmup                        # Layer 2
  --host 127.0.0.1 --port 11434
```
**Expected:** ~25 tok/s sustained decode, ~5 GB RSS, 1.2 s cold TTFT.

Optional additions:
- `--moe-eager-load` — 45 % cold TTFT reduction for chat workloads; trades 2-4 tok/s decode.
- `--moe-keep-warm 60` — heartbeat every 60 s to keep cache hot on idle bursty workloads (fixed in fork `de95d1f3` — no longer collides with real requests).

### Recipe B — reference quality (BF16)

```bash
~/streammoe/anemll-flash-llama.cpp/build/bin/llama-server \
  -m ~/Downloads/qwen36-bf16/BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf \
  --moe-sidecar ~/streammoe/models/qwen36-bf16-sidecar \
  --moe-mode slot-bank --moe-slot-bank 128 \
  --moe-prefetch-temporal \
  --flash-attn on --ctx-size 4096 -ngl 99 \
  --streammoe-warmup                        # Layer 2 only — L1 no-op on sidecar>RAM
```
**Expected:** ~13-22 tok/s decode, ~24 GB RSS, 3.5 s cold TTFT.

Do NOT pass `--moe-eager-load` — 60 GB sidecar exceeds 48 GB physical
RAM, so the read-through can't stay in page cache. Layer 1 would cost
15 s of startup for no TTFT gain. The fork safeguard will log and skip
mlock automatically if you do pass it.

### Recipe C — MLX 4-bit (fast but lower quality)

```bash
# SwiftLM running at :5413
swiftlm serve --model mlx-qwen36-4bit
```
**Expected:** ~20 tok/s decode, 0.2 s TTFT, 13 GB RSS.
**DO NOT SHIP** for analytical workloads — 36 % of 80 MT-Bench answers were
judged "different" from GGUF stock (mean_judge_score 3.25/5, strict
pass rate 2.5 %). Acceptable for casual chat; not for STEM, coding, math.

### Recipe D — Ollama compatibility layer

Ollama on macOS uses its own llama.cpp fork. For streaming, StreamMoE
binds `:11434` directly and replaces Ollama's runtime. See the StreamMoE
macOS app's ServerSupervisor + OllamaController for the handoff.

---

## Decision tree

```
Do you need reference quality?
├─ YES → Recipe B (BF16 streaming) — 13 tok/s, 24 GB RSS, judge-parity
└─ NO → Do you need short-prompt chat TTFT < 1.5s?
        ├─ YES → Recipe A + --moe-eager-load — 1.18 s cold, 5 GB RSS
        └─ NO  → Does the workload generate > 300 tokens per request?
                  ├─ YES → Recipe A without --moe-eager-load — 35 tok/s peak
                  └─ NO  → Recipe A — 25 tok/s, 5 GB RSS, balanced
```

---

## Five cross-engine findings

1. **GGUF slot-bank beats MLX streaming 3×** (25.7 vs 8.3 tok/s). The
   Flash-MoE fork's purpose-built per-layer slot-bank + Metal kernels
   outperform generic MLX+mmap streaming by a wide margin.
2. **MLX resident wins TTFT 14×** (0.18 s vs 2.5 s baseline). MLX
   pre-compiles kernels and eagerly allocates Metal buffers. The
   StreamMoE TTFT patch closes this gap to ~6× (1.18 s vs 0.18 s)
   without giving up GGUF's throughput or quality.
3. **MLX 4-bit quantization diverges from GGUF K-quants** on technical
   tasks — 80 % "different" on STEM, 50 % on coding, 40 % on math. This
   is a quality problem, not a speed problem — shipping MLX 4-bit is
   objectively worse on workloads that matter.
4. **BF16 35B needs ≥64 GB Mac.** No 48 GB configuration runs MLX BF16
   at usable speed; only GGUF BF16 via slot-bank streaming is viable at
   48 GB, and it's 2× slower than Q4.
5. **Layer 1 (eager-load) only helps when sidecar fits in RAM.** On Q4
   (18 GB sidecar) it's a 45 % cold-TTFT win. On BF16 (60 GB sidecar)
   it's a 15 s startup tax for zero TTFT gain — the fork's safeguard
   skips it automatically.

---

## Source files

- `streammoe-bench/RESULTS.md` — full TTFT matrix with all four layers on/off
- `streammoe-bench/ttft-results/ttft_matrix_*.json` — raw per-cell JSON
- `streammoe-bench/quality-results/quality_verify_qwen36bf16_*.json` — BF16 quality pairs
- `/Users/claude/streammoe/README.md` — authoritative prior-project findings (pre-TTFT-patch)
- `/Users/claude/streammoe/results/sweep_20260416_230927/phase2c_qwen36_summary.json` — 80-prompt Q4 sustained decode
- `/Users/claude/streammoe/results/sweep_mlx_20260417_184225/mlx_4bit_80_gates.json` — MLX 4-bit 80-prompt judge verdicts
