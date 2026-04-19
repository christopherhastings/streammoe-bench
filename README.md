# StreamMoE

One-click macOS app for SSD-streamed Mixture-of-Experts inference. Ships a
custom `llama-server` on port 11434 (replacing Ollama's runtime) plus a
menu-bar client that wires Open WebUI, Cursor, LM Studio and friends
into it.

**Runs 35B-class MoE models at ~25 tok/s with ~5 GB RAM on a 48 GB Mac.**

## Repository layout

This is the **monorepo** for the user-facing and benchmarking code.
Everything lives under one tree:

```
streammoe-app/     Swift macOS menu-bar app (SwiftPM, SwiftUI/AppKit).
                   StreamMoECore library + 31 XCTest + scripts/build-app.sh
                   that bundles StreamMoE.app.

streammoe-bench/   Python benchmark harness. TTFT matrix + 80-prompt
                   quality verifier. Pulls production flag composition
                   from the sibling streammoe_bench project at
                   /Users/claude/streammoe/.

docs/              High-level project documents:
                   - UNIFIED_CONFIG_AND_RESULTS.md — the one place
                     that tells you what config to run and why.
                   - PROJECT_OVERVIEW.md — multi-session status.
                   - CODESIGNING.md — what you need to ship the .app.
```

**Only one other repo is separate:**
[`christopherhastings/anemll-flash-llama.cpp`](https://github.com/christopherhastings/anemll-flash-llama.cpp),
the C++ fork of [Anemll/anemll-flash-llama.cpp](https://github.com/Anemll/anemll-flash-llama.cpp).
It stays separate because it's a fork of an upstream project that needs
its own PR lifecycle — we don't want to drag app/bench code across when
we submit patches upstream.

## Quickstart

**Run the benchmark matrix** (needs the fork built at
`/Users/claude/streammoe/anemll-flash-llama.cpp/build/`):

```sh
cd streammoe-bench
python3.11 ttft_bench.py --iters 3                # full 2-model TTFT matrix
python3.11 quality_verify.py --model qwen36bf16   # 80-prompt quality gate
```

**Build the macOS app:**

```sh
cd streammoe-app
swift test                                        # 31 XCTest cases
swift build                                       # debug build
scripts/build-app.sh                              # creates build/StreamMoE.app
CODESIGN_IDENTITY="Developer ID Application: ..." \
    scripts/build-app.sh                          # + sign
```

## Headline results

See `docs/UNIFIED_CONFIG_AND_RESULTS.md` for the full matrix. Short version:

- **Production config: GGUF Q4_K_XL + slot-bank streaming + temporal prefetch** — 25 tok/s sustained, 5 GB RSS, 1.2 s cold TTFT with Layers 1+2.
- **Quality parity** with GGUF stock (all-resident) — judge-verified on 80 MT-Bench prompts.
- **TTFT patch shipped** on the fork branch `feature/moe-eager-load` — cold TTFT 2.14 s → 1.18 s (45 % reduction) on Q4.

## Status

v0.1.0 tagged on all three repos (monorepo + fork).
