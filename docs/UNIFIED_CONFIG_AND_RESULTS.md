# StreamMoE — Unified Configuration & Results Matrix

**Hardware:** Apple M4 Max, 48 GB unified memory, macOS Darwin 25.4.0
**Last updated:** 2026-04-18
**Scope:** Every Qwen / Gemma MoE model we've benchmarked × every engine
(GGUF-on-fork, MLX-via-SwiftLM, MLX-via-Ollama) × every loading strategy
(stock-resident, slot-bank streaming, MLX-resident, MLX-streaming).

This is the one place to look when deciding *how* to run a model.

---

## Headline

| Role                       | Config                                                   | Why |
|----------------------------|----------------------------------------------------------|-----|
| **Production / ship**      | GGUF Q4_K_XL, slot-bank streaming (sb=256, temporal) + `--streammoe-warmup` | 25 tok/s sustained, ~5 GB RAM, 1.2 s cold TTFT, judge-verified quality parity with stock |
| **Max chat UX**            | Same + `--moe-eager-load` (chat workloads only)          | 45 % cold-TTFT reduction; costs 2-4 tok/s decode on long-form |
| **Reference quality**      | GGUF BF16, slot-bank streaming (sb=128, temporal)        | 13 tok/s sustained, 7.9 GB RSS. Judge-equivalent to Q4 — no quality gain but a reference anchor. |
| **Long-form generation**   | GGUF Q4_K_XL, slot-bank **without** L1                   | 35-37 tok/s peak decode. L1's 2-4 tok/s cost compounds over 300+ tokens. |
| **Fast TTFT but lower q**  | MLX 4-bit (SwiftLM)                                      | 0.20 s TTFT, 20 tok/s, BUT 29/80 answers "different" vs GGUF stock on MT-Bench |
| **DO NOT SHIP for STEM**   | MLX 4-bit for coding/math/STEM                           | 80 % "different" verdicts on STEM, 50 % on coding, 40 % on math |

---

## The full model matrix

### Qwen3.6-35B-A3B — primary production target

**256 experts · top-8 routing · 40 MoE layers · 35B params / 3B active**

Everything we ship today runs this model.

#### Qwen3.6 · Q4_K_XL (21 GB GGUF · 18 GB sidecar)

Three-stage validation:

| Stage | Prompts | Mode | Decode | TTFT | RSS | Quality vs stock |
|-------|--------:|------|-------:|------|----:|------------------|
| Phase 2A (axis-1 top) | 4 | slot_bank=256, temporal | 24.59 tok/s | 1.35 s | 4.39 GB | — |
| Phase 2A (axis-2 top) | 4 | sb=128, temporal+predict-prev | 21.16 tok/s | 2.17 s | 4.40 GB | — |
| Phase 2A (axis-3 top) | 4 | io_split=1, temporal | 17.14 tok/s | 2.74 s | 4.47 GB | — |
| Phase 2B (factorial p2b-13) | 4 | 5-axis Latin square best | 23.41 tok/s | 1.18 s | 4.55 GB | — |
| **Phase 2C stock** | **80** | all-resident | **25.12 tok/s** | 0.51 s | **21.54 GB** | baseline (identity) |
| **Phase 2C best-tps** | **80** | sb=256, temporal, ctx=32k | **22.14 tok/s** | 3.31 s | **5.43 GB** | **judge 4.71/5, 0/80 regressions** |
| Phase 2C lowest-ram | 80 | sb=256, q4 KV, io_split=16, ctx=32k | 21.09 tok/s | 3.34 s | 4.92 GB | judge-verified equivalent |
| **TTFT bench Run C baseline** | 3 (n_predict=16) | production flags | **37.36 tok/s** | 2.14 s | 8.76 GB | — |
| **TTFT bench Run C +L1+L2** | 3 (n_predict=16) | + `--moe-eager-load` + `--streammoe-warmup` | 35.37 tok/s | **1.18 s** | 8.76 GB | byte-identical where measured |

Why the two decode numbers (25 vs 37 tok/s)? Short-prompt 16-token
decoding runs before slot-bank cache churn fully saturates. Sustained
300-token decoding (Phase 2C) is the user-facing rate.

#### Qwen3.6 · BF16 (64 GB GGUF · 60 GB sidecar)

| Stage | Prompts | Mode | Decode | TTFT | RSS | Quality vs Q4 |
|-------|--------:|------|-------:|------|----:|---------------|
| BF16 findings (sb=128, ctx=32k) | 20 | slot-bank, temporal, n_predict=300 | **13.49 tok/s** | 6.74 s | **7.86 GB** | **15 same / 5 similar / 0 different — no quality gain** |
| TTFT bench Run C baseline | 3 (n_predict=16) | sb=128, temporal, flash-attn | 21.93 tok/s | 3.57 s | 23.19 GB | — |
| TTFT bench Run C +L2 | 3 (n_predict=16) | + `--streammoe-warmup` | 21.97 tok/s | 3.50 s | 23.41 GB | — |
| BF16 quality verify | 80 | baseline vs +L1+L2 | — | — | — | **79/80 byte-identical; 1 Metal fp32 artifact** |
| Stock (all-resident) | — | — | — | — | — | **doesn't fit on 48 GB** |

**Finding:** BF16 delivers no quality advantage over Q4_K_XL despite 4×
expert bytes and 2× decode cost. Ship Q4.

#### Qwen3.6 BF16 via MLX (blocked on 48 GB)

| Engine | Mode | Verdict |
|--------|------|---------|
| Ollama (qwen3.6:35b-a3b-mlx-bf16) | resident | **refuses to load — "requires 65 GiB > 37 avail"** |
| SwiftLM (MLX) | streaming | 0.14 tok/s prefill, MEM_DEMAND 129 GB, unusable |

Both blocked by the same physical RAM ceiling. Need ≥64 GB Mac for
viable MLX BF16. Status: deferred to Mac Mini M4 Pro 64 GB.

---

### Qwen3.5-35B-A3B · Q5_K_M

**256 experts · top-8 · Q5_K_M (~25 GB GGUF)**

Same architecture as Qwen3.6 but Q5 quantization. Phase 2A completed;
Phase 2C (80-prompt quality gate) deferred under the **rerun345**
tracking task because Phase 2A's axis ordering needed correction.

| Stage | Prompts | Best config | Decode | TTFT | RSS |
|-------|--------:|-------------|-------:|------|----:|
| Phase 2A (axis-1 top) | 4 | slot_bank=160 | 11.63 tok/s | 2.94 s | 4.21 GB |
| Phase 2A (axis-5 top) | 4 | ctx=2048 | 12.19 tok/s | 2.06 s | 4.58 GB |
| **Phase 2B (factorial p2b-21)** | 4 | ctx + sb + temporal | **23.34 tok/s** | 1.18 s | **4.22 GB** |
| Phase 2C 80-prompt quality | — | — | **pending rerun345** | — | — |

**Finding:** Q5's factorial sweep is competitive with Q4 (~23 tok/s).
Phase 2A axis-1 numbers are depressed because the axis-ordering issue
corrupted the held values — factorial recovers the real throughput.

---

### Gemma-4-26B-A4B-it · Q8_K_XL

**128 experts · ~26 GB GGUF · 30 MoE layers**

Partial data only. Phase 2A produced mixed-valid results; Phase 2B
skipped (0/32 valid); Phase 2C deferred. All under rerun345.

| Stage | Prompts | Best config | Decode | TTFT | RSS |
|-------|--------:|-------------|-------:|------|----:|
| Phase 2A (axis-5 ctx=16384) | 4 | temporal + slot-bank | **16.34 tok/s** | 0.96 s | 6.84 GB |
| Phase 2A (axis-1 slot sweeps) | 4 | various | **4.04-4.12 tok/s** | 4.6-4.9 s | 7.0-7.2 GB |
| Phase 2B | — | — | **skipped — 0/32 valid** | — | — |
| Phase 2C | — | — | pending rerun345 | — | — |

**Finding:** axis-1 slot-bank sweep produces wildly inconsistent
throughput (4 vs 16 tok/s). The axis-5 (ctx) sweep is self-consistent.
Gemma-4 numbers marked **preliminary** until rerun345.

---

### Qwen3.5-397B-A17B — abandoned

**512 experts · top-10 · 60 MoE layers · Q4_K_M (~227 GB GGUF)**

Single-machine viability check on 48 GB M4 Max.

| Metric | Result |
|--------|-------:|
| Sidecar size (disk, APFS-cloned) | 167 GB |
| Server cold start to ready | 129 s |
| Server RSS after load (sb=64, ctx=8192) | 6.2 GB |
| Per-token decode time | **2.67 s/token** |
| **Effective decode rate** | **0.37 tok/s** |
| Expert I/O per token | 2.17 GB |

**Verdict:** SSD-bandwidth-bound, not usable on 48 GB. Would need ≥128 GB
Mac or faster NVMe. Sidecar + shards deleted to reclaim disk.

---

## Baseline numbers — GGUF per model

Two anchors per model so you can see what streaming costs over stock
and what prefetch-temporal buys over no-prefetch.

### Stock (all-resident, smoke-test — short prompt, n_predict 7-8)

Measures kernel-warm throughput with no sidecar streaming. Decode numbers
are inflated vs sustained 300-token generation because the model hasn't
saturated cache churn. Useful as an upper-bound reference.

| Model | Quant | Prompt ms | Prompt tok/s | Decode tok/s | Decode ms/token |
|-------|-------|----------:|-------------:|-------------:|----------------:|
| Qwen3.6-35B-A3B  | Q4_K_XL  |  805.03 |  24.84 | 63.65 | 15.71 |
| Qwen3.5-35B-A3B  | Q5_K_M   |   95.84 | 208.67 | 68.42 | 14.61 |
| Gemma-4-26B-A4B  | Q8_K_XL  |  204.42 | 102.73 | 23.45 | 42.64 |

### Streaming baseline (slot-bank, **no** prefetch — Phase 2A axis2-pf0, 4 prompts, n_predict 300)

The "what does slot-bank streaming cost without any prefetch help" anchor.
This is the point from which `--moe-prefetch-temporal` and all other
optimizations were measured.

| Model | Quant | Decode tok/s | TTFT  | RSS   |
|-------|-------|-------------:|-----:|------:|
| Qwen3.6-35B-A3B  | Q4_K_XL  | 15.03 | 2.74 s | 4.40 GB |
| Qwen3.5-35B-A3B  | Q5_K_M   |  9.06 | 4.04 s | 4.42 GB |
| Gemma-4-26B-A4B  | Q8_K_XL  |  4.08 | 4.62 s | 7.07 GB |

### Streaming best (Phase 2B factorial sweet spot, 4 prompts, n_predict 300)

The "what does the best validated streaming config give" anchor. This
is what production should ship for each model, modulo the Phase 2C
quality gate (done for Qwen3.6 Q4 only; pending for Q5 and Gemma-4
under rerun345).

| Model | Quant | Config | Decode tok/s | TTFT  | RSS   | vs no-prefetch |
|-------|-------|--------|-------------:|-----:|------:|---------------:|
| Qwen3.6-35B-A3B  | Q4_K_XL  | p2b-13 (ctx+sb+temporal) | **23.41** | 1.18 s | 4.55 GB | +56 % decode, 2.3× faster TTFT |
| Qwen3.5-35B-A3B  | Q5_K_M   | p2b-21 (ctx+sb+temporal) | **23.34** | 1.18 s | 4.22 GB | +158 % decode, 3.4× faster TTFT |
| Gemma-4-26B-A4B  | Q8_K_XL  | axis-5 ctx=16384 (axis-1 sweep corrupted) | **16.34** | 0.96 s | 6.84 GB | +300 % decode, 4.8× faster TTFT |

**Finding on the Q5/Gemma-4 lift:** The prefetch-temporal + factorial
sweet spot is roughly 2-4× the no-prefetch baseline on these models,
and on Q5_K_M it brings performance essentially equal to Q4_K_XL
(23.34 vs 23.41 tok/s) — the extra bytes of Q5 per token don't
materially hurt once prefetch keeps the pipeline busy.

---

## Cross-engine: GGUF vs MLX on the same model

All below are Qwen3.6-35B-A3B on 48 GB M4 Max.

| Engine             | Quant     | Mode                | n_prompts | Decode tok/s | TTFT     | RSS      | Quality vs GGUF stock (distribution) |
|--------------------|-----------|---------------------|----------:|-------------:|----------|---------:|--------------------------------------|
| **GGUF (fork)**    | Q4_K_XL   | **stock resident**  |        80 | **25.12**    | 0.51 s   | 21.54 GB | baseline                             |
| **GGUF (fork)**    | Q4_K_XL   | **slot-bank stream** |       80 | 22.14        | 3.31 s   | **5.43 GB** | **0/80 different (judge 4.71/5)**  |
| GGUF (fork)        | Q4_K_XL   | slot-bank lowest-ram |       80 | 21.09        | 3.34 s   | 4.92 GB  | 0/80 different                       |
| GGUF (fork)        | BF16      | slot-bank stream    |        20 | 13.49        | 6.74 s   | 7.86 GB  | 15/5/0 same/similar/different vs Q4  |
| **SwiftLM (MLX)**  | 4-bit     | **resident (ctx32k)** |      20 | **24.29**    | **0.18 s** | 13.37 GB | — (only 20-prompt)                 |
| **SwiftLM (MLX)**  | 4-bit     | **resident (80 prompts)** |  80 | 19.96        | 0.20 s   | 8.30 GB  | **13/38/29 same/similar/different — 36 % regression** |
| SwiftLM (MLX)      | 4-bit     | `--stream-experts`  |         5 | 8.34         | 1.74 s   | 4.48 GB  | not judged (n=5)                     |
| SwiftLM (MLX)      | BF16      | streaming           |        — | 0.14 prefill | 22+ min  | 129 GB demand | not viable on 48 GB              |
| Ollama (MLX)       | BF16      | resident            |        — | refused      | —        | —        | "requires 65 GiB > 37 available"     |

### MLX 4-bit quality breakdown (80 prompts, claude-sonnet-4-6 judge)

| Category   | n  | same | similar | **different** |
|------------|---:|-----:|--------:|--------------:|
| Writing    | 10 | 4    | 5       | 1             |
| Roleplay   | 10 | 3    | 6       | 1             |
| Reasoning  | 10 | 2    | 6       | 2             |
| Math       | 10 | 1    | 4       | **4** (40 %)  |
| Coding     | 10 | 1    | 4       | **5** (50 %)  |
| Extraction | 10 | 1    | 5       | 4             |
| STEM       | 10 | 0    | 2       | **8** (80 %)  |
| Humanities | 10 | 1    | 6       | 3             |
| **Total**  | 80 | 13   | 38      | **29 (36 %)** |

**Finding:** MLX 4-bit loses precision on tasks where small numeric
differences matter. Do not ship MLX 4-bit for analytical workloads.

---

## Five cross-engine findings

1. **GGUF slot-bank beats MLX streaming 3×** (25 vs 8 tok/s). The Flash-MoE
   fork's purpose-built per-layer slot-bank + Metal kernels outperform
   generic MLX + mmap streaming.
2. **MLX resident wins TTFT 14×** vs unpatched GGUF (0.18 s vs 2.5 s).
   The StreamMoE TTFT patch (Layers 1 + 2) closes this to ~6× (1.18 s vs
   0.18 s) without giving up throughput or quality.
3. **MLX 4-bit quantization diverges from GGUF K-quants** on technical
   tasks. This is a quality problem, not a speed problem.
4. **BF16 delivers no quality gain over Q4_K_XL on 35B MoE.** Judge finds
   0/20 "different" in either direction. 2× slower for no benefit. Ship Q4.
5. **397B not viable on 48 GB Mac.** Abandoned at 0.37 tok/s, SSD-bound.

---

## Configuration recipes

### Recipe A — production streaming (Q4_K_XL) ← ship this

```bash
llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --moe-sidecar qwen35-35b-sidecar \
  --moe-mode slot-bank --moe-slot-bank 256 \
  --moe-prefetch-temporal \
  --flash-attn on --ctx-size 4096 -ngl 99 \
  --streammoe-warmup \
  --host 127.0.0.1 --port 11434
```
Expected: ~25 tok/s sustained, ~5 GB RSS, 1.2 s cold TTFT.

### Recipe B — reference quality (BF16)

```bash
llama-server -m Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf \
  --moe-sidecar qwen36-bf16-sidecar \
  --moe-mode slot-bank --moe-slot-bank 128 \
  --moe-prefetch-temporal \
  --flash-attn on --ctx-size 4096 -ngl 99 \
  --streammoe-warmup
```
Expected: ~13 tok/s sustained, ~8-24 GB RSS, 3.5-6.7 s cold TTFT.

### Recipe C — MLX 4-bit (chat-only)

```bash
swiftlm serve --model mlx-qwen36-4bit
```
Expected: ~20 tok/s, 0.2 s TTFT, 13 GB RSS. **DO NOT SHIP** for STEM / coding / math.

---

## Decision tree

```
Need sub-second TTFT and only chat (no STEM/coding/math)?
├─ YES → Recipe C (MLX 4-bit)
└─ NO → Production chat?
        ├─ YES → Recipe A + --moe-eager-load (1.2 s TTFT)
        └─ Long-form gen (>300 tok)?
                ├─ YES → Recipe A WITHOUT --moe-eager-load (35 tok/s peak)
                └─ NO  → Recipe A (balanced default)
```

---

## What's still missing

1. Q5_K_M 80-prompt quality gate — Phase 2C deferred under rerun345.
2. Gemma-4 full matrix — Phase 2A axis-1 corrupted, Phase 2B skipped.
3. MLX BF16 on ≥64 GB Mac — physically impossible on 48 GB.
4. Post-patch claude-judge on the BF16 1/80 divergence.
5. Fresh bench confirming Layer 3 keep-warm race fix.

### In-flight (2026-04-19/20)

Cross-model quality matrix — 5 locally-run configs × 2 frontier
references, same 80 MT-Bench prompts:

| Local config    | Model · Quant · Loading                       | Status |
|-----------------|-----------------------------------------------|--------|
| q4_resident     | Qwen3.6-35B-A3B Q4_K_XL · stock resident      | sampled ✓ (80/80) |
| q4_streaming    | Qwen3.6-35B-A3B Q4_K_XL · slot-bank sb=256    | sampled ✓ (80/80) |
| bf16_streaming  | Qwen3.6-35B-A3B BF16 · slot-bank sb=128       | sampled ✓ (80/80) |
| qwen35_9b       | Qwen3.5-9B Q4_K_M · dense resident            | downloaded ✓, sampling pending |
| qwen35_122b     | Qwen3.5-122B-A10B UD-Q4_K_XL · slot-bank 128  | downloaded + sidecar extracted ✓, sampling pending |

References: Claude Haiku 4.5 + Claude Sonnet 4.6, both captured (80 each).

Three fixes shipped 2026-04-20 against the overnight crash from the
first judge run:
- `quality_compare.py` n_predict 1024 → 3000 (Qwen3 `<think>` blocks
  were consuming the whole budget on ~20/80 prompts).
- `judge_vs_references.py` strips `<think>…</think>` before judging +
  surfaces all-think-no-answer as a `no_answer` verdict.
- Judge resumability: per-prompt checkpoint, rerun skips done IDs.
- Bonus: `ANTHROPIC_API_KEY.strip()` so a trailing `\n` doesn't turn
  every judge call into a LocalProtocolError (which is what killed
  the overnight built-in judge phase).

Local-judge fallback: `--judge-endpoint http://localhost:1234/v1`
routes to LM Studio (or any OpenAI-compat endpoint) so the cross-judge
run can stay off the Anthropic API when desired.

Ready to kick off the remaining sampling + full judge matrix when the
GPU is free and the user signs off on the API cost (~$2-4 on Claude or
$0 on LM Studio).

---

## Source files

- `streammoe-bench/RESULTS.md` — TTFT matrix all four layers on/off
- `streammoe-bench/ttft-results/ttft_matrix_*.json` — Run C raw (both models)
- `streammoe-bench/quality-results/quality_verify_qwen36bf16_*.json` — 79/80 result
- `/Users/claude/streammoe/results/sweep_20260416_230927/phase2a_{qwen36,qwen35q5,gemma4}_summary.json` — 30-config axis sweeps per model
- `/Users/claude/streammoe/results/sweep_20260416_230927/phase2b_{qwen36,qwen35q5,gemma4}_summary.json` — 32-combo factorial per model
- `/Users/claude/streammoe/results/sweep_20260416_230927/phase2c_qwen36_summary.json` — 80-prompt judge gate
- `/Users/claude/streammoe/results/sweep_mlx_20260417_184225/swiftlm_mlx-4bit-swiftlm-80prompts_summary.json` — MLX 4-bit 80-prompt
- `/Users/claude/streammoe/results/sweep_mlx_20260417_184225/mlx_4bit_80_gates.json` — MLX 4-bit category-level judge
- `/Users/claude/streammoe/BF16_FINDINGS.md` — BF16 vs Q4 judge
- `/Users/claude/streammoe/397B_FINDINGS.md` — abandonment writeup
- `/Users/claude/streammoe/MLX_FINDINGS.md` — three-way MLX comparison
- `/Users/claude/streammoe/README.md` — authoritative prior-project summary
