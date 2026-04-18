# StreamMoE — Project Overview

**Last updated:** 2026-04-18
**Status:** TTFT optimization patch shipped + measured (38% cold TTFT win on Q4_K_XL). Menu-bar app v0.1.0 production-ready (31/31 XCTest, .app bundles via build script). Safeguards added after BF16-on-48GiB caused host memory pressure.

**Repos as of v0.1.0:**

- **Fork:** `christopherhastings/anemll-flash-llama.cpp` branch `feature/moe-eager-load` — 4-layer TTFT patch + `/streammoe/status` + memory-safety guards
- **App:** `christopherhastings/streammoe-app` — menu-bar SwiftUI client, builds to `StreamMoE.app`
- **Bench:** `christopherhastings/streammoe-bench` — TTFT matrix harness + byte-identical quality verifier + measured RESULTS.md

---

## Measured TTFT improvements (M4 Max 48 GB, N=3)

### Q4_K_XL (18 GiB sidecar — fits in RAM)

| Config            | Cold TTFT (s) | Warm p50 (s) | RSS (GiB) |
|-------------------|--------------:|-------------:|----------:|
| baseline          |         1.707 |        0.284 |      7.34 |
| +L1 eager         |         1.061 |        0.191 |      6.91 |
| +L1+L2+L3+L4 full |         1.087 |        0.198 |      7.01 |

**Cold TTFT: 38 % reduction. Warm p50: 32 % reduction.** Layer 1 captures the bulk.

### BF16 (60 GiB sidecar — exceeds RAM, sb=128)

| Config            | Cold TTFT (s) | Warm p50 (s) | RSS (GiB) |
|-------------------|--------------:|-------------:|----------:|
| baseline          |         3.633 |        0.672 |     18.63 |
| +L1 eager         |         3.657 |        0.675 |     18.63 |
| +L1+L2 warmup     |         3.571 |        0.605 |     18.90 |
| +L1+L2+L3+L4 full |         3.506 |        0.998 |     18.88 |

**L1 is a no-op on BF16** (sidecar evicts from page cache before first prompt). **L2 alone gives a 10 % warm-p50 win.** **L3 regresses warm p50** (heartbeat steals Metal queue time during measurement; needs pause-on-activity gate — see streammoe-bench/RESULTS.md "Known issues").

**Recommended config:** `--streammoe-warmup` alone on BF16; full stack on Q4.

---

## Safeguards added after BF16 host memory pressure (2026-04-18)

Three behavior changes prevent the host from getting into a VM-thrash spiral when configs ask for more pinned memory than the machine has:

1. **Layer 4 default reverted to `-ub 1`** — the auto-bump to 256 was OOMing Metal on BF16 (256× larger activation footprint). Now opt-in via explicit `-ub 256 -b 256`.
2. **`eager_load_moe_sidecar` refuses mlock when sidecar > 60% of physical RAM** — falls back to read-through-only (still warms page cache).
3. **Read-throttle when sidecar > physical RAM** — sleeps 1 ms every 64 MiB so the kernel can evict cleanly between chunks.

These were the changes that made running BF16 on 48 GiB hardware destabilize the host. The crash was reproducible; the fix is in `c9b…` on the fork.

> **Primary source of truth:** `/Users/claude/streammoe/README.md` — it has the detailed findings, reproducible commands, and next-step recommendations. This doc is the planning-repo view and points at the active work.

---

## What this project validates

Can large Mixture-of-Experts language models run on consumer Apple Silicon by streaming cold experts from NVMe SSD instead of keeping all weights in RAM?

**Answer: yes, with measurable overhead.**

Qwen3.6-35B-A3B at 22-23 tok/s with ~4.4 GB RSS (vs ~25 tok/s at 21 GB stock). **5× less RAM for 10% throughput cost, zero quality regressions vs stock on 80-prompt MT-Bench.** Context up to 200k is effectively free.

---

## Hardware & workspaces

| Machine | Role | RAM | Status |
|---------|------|-----|--------|
| Mac Mini M4 Max | Primary dev/benchmark | 48 GB | Active (this sweep) |
| MacBook Air M3 | Secondary test | 24 GB | Port deferred |

Working directories:
- `~/streammoe/` (owned by `claude`, staff-group writable) — code, models, results, benchmarks
- `~/flash-moe-engine/` (this repo) — planning, overview, original spec

---

## Architecture

```
User → Open WebUI (Docker :3000)
         │
         ▼  OpenAI-compatible API
      llama-server (:8080) ← anemll-flash-llama.cpp fork
         │
         ├─ Attention, embeddings, router → Metal GPU (~4 GB)
         └─ Expert FFNs → slot-bank LRU cache
              ├─ hit  → Metal compute
              └─ miss → pread() from SSD sidecar
                        (temporal prefetch overlaps I/O with GPU compute)
```

**Sidecar files:** per-layer binary dumps of expert weights extracted from GGUF, plus `manifest.json` describing layout. ~18-23 GB per 35B-class model.

---

## Validated production config (Qwen3.6-35B-A3B-UD-Q4_K_XL)

```bash
~/streammoe/anemll-flash-llama.cpp/build/bin/llama-server \
  -m ~/Downloads/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 32768 \
  --flash-attn on --jinja -ngl 99 --seed 42 \
  --moe-sidecar ~/streammoe/models/qwen35-35b-sidecar \
  --moe-mode slot-bank \
  --moe-prefetch-temporal \
  --moe-slot-bank 256
```

Expected: ~22 tok/s decode, ~4.4 GB RSS, TTFT < 1s on short prompts, up to 200k context free.

---

## Five headline findings

1. **`temporal` prefetch alone wins** — combined prefetch modes degrade throughput 35-50%
2. **Slot-bank sweet spot: sb=192-256** on 48 GB Macs (90-99% of stock throughput)
3. **io_split, ctx size: free** — <5% variance on decode speed
4. **q4_0 KV cache: NOT free** — silent quality regression on algorithmic reasoning
5. **`--moe-topk` reduction does NOT produce a close draft** — Option B empirically dead, acceptance rate 6% vs 60% gate

Full analysis in `~/streammoe/README.md`.

---

## Status summary

| Phase | Status |
|-------|--------|
| Phase 0 — TDD harness, package | Complete (110 unit tests pass) |
| Phase 1 — Smoke + baseline | Complete |
| Phase 2A — One-at-a-time scans × 3 models | Complete (90 configs) |
| Phase 2B — Factorial × 2 models | Complete (64 configs; Gemma-4 skipped) |
| rerun345 — Corrected baseline on qwen36 | Complete |
| Phase 2C — 80-prompt MT-Bench quality gate | Complete (judge-based) |
| Step 1 — Extended context 32k→200k | Complete |
| Step 2 — Phase 4 acceptance rates | Complete — Option B NO-GO |
| **Step 3 — Option B implementation** | **SKIPPED** per gate |
| 397B sweep | **Ready for next session** (see `HANDOFF_397B.md`) |

---

## Deferred / future work

In priority order from `~/streammoe/README.md` → Next Steps:

**Tier 1 — low effort, fills known gaps:**
- q8_0 KV quality test (~30 min)
- rerun345 for Q5 and Gemma-4 (~2h)
- Extreme context 500k/1M (~30 min)

**Tier 2 — new work, larger scope:**
- MacBook Air M3 24 GB port
- Small-draft-model speculative decoding (standard, not self-speculative)
- Layer-skip drafting

**Tier 3 — scale up:**
- 397B sweep (see `HANDOFF_397B.md`)
- Gemma-4 Phase 2B

**Tier 4 — research:**
- Why q4 KV breaks algorithmic reasoning
- Document SWA+GQA's role in making long-context cheap

---

## File inventory

### In `~/streammoe/` (the working repo)
- `README.md` — authoritative findings + reproducible commands
- `HANDOFF_397B.md` — 397B session handoff with human go/no-go line
- `PHASE2C_FINAL.md` — quality verification methodology
- `OPTION_B_GO_NO_GO.md` — why --moe-topk speculative decoding fails
- `OPTION_B_MoE_SPARSE_DRAFT.md` — original design (marked CLOSED at top)
- `FUTURE_EXPERIMENTS.md` — experiment proposals (largely superseded by README)
- `streammoe_bench/` — Python package, 11 modules, 110 unit tests
- `scripts/` — chain runners (`run_all_phases.sh`, `retry_different_verdicts.py`, etc.)
- `results/sweep_20260416_230927/` — per-phase summary JSONs, anomalies log, baseline
- `tests/` — TDD harness (unit + integration)
- `docs/archive/` — superseded interim writeups

### In this repo (`~/flash-moe-engine/`)
- `PROJECT_OVERVIEW.md` — this file (high-level status)
- `StreamMoE_Aider_Implementation_Guide.md` — original phased plan (historical)

---

## Tooling that shipped

- **Multi-gate quality analysis** — strict token match, prefix match, and Claude-Sonnet-4.6 LLM-as-judge (tool-use enforced JSON)
- **Adaptive context sweep** — runs initial points, fires intermediates only when adjacent pairs show >20% decode drop
- **Atomic checkpointing** — every sweep is resumable mid-run
- **Watchdog** — hard/soft/warn thresholds on swap, memory pressure, temp
- **Prompt suites** — 20-prompt (frozen SHA-256) and full 80-prompt MT-Bench
- **Chain runner** — sequential Phase 2A × 3 models → Phase 2B × 3 → frontier → report

---

## Approach learnings (for future sweeps)

- **Run axes in dependency order** — the axis 2 winner (prefetch=temporal) should be the held value for axes 3-5. We got this wrong the first time; added `rerun345` command to recover.
- **Strict token match is a bad quality gate** for GPU inference (Metal fp32 ops aren't bit-reproducible). Use an LLM judge with strict JSON.
- **max_tokens matters** — 300 truncates most MT-Bench responses. Use ≥1000 for real quality verification.
- **Memory pressure accumulates** — cooldowns + reboots matter for sweeps lasting hours.
- **Bash supervisor scripts survive `pkill -f <python>`** — kill the supervisor separately, or use `pkill -f <script_name>`.
