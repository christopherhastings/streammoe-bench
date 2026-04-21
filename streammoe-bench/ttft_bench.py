#!/usr/bin/env python3
"""StreamMoE benchmark harness — three modes through one entrypoint.

  --mode ttft       (default) Short-prompt TTFT matrix. 4 configs × N models,
                    N cold + (N-1) warm runs per cell, reports TTFT +
                    decode tok/s + RSS + startup. Unchanged from prior work.

  --mode quality    80-prompt MT-Bench sampling. Fires the prompts against
                    each requested local config, collects full responses +
                    per-prompt RSS, writes one responses_<config>.json per
                    config. Resumable: already-saved prompt IDs are skipped.

  --mode judge      Takes the responses_*.json files from --mode quality
                    (or an earlier run) and pairwise-judges each local
                    config against each reference answer set (Haiku, Sonnet)
                    using either the Anthropic API or a local OpenAI-compat
                    endpoint (LM Studio, vLLM, etc). Qwen3 <think> blocks
                    are stripped before judging; all-think responses are
                    surfaced as a "no_answer" verdict. Resumable per pair.

The harness pulls llama-server flags from the validated production config
in /Users/claude/streammoe/streammoe_bench so that every cell runs against
the same flags the user-facing deployment ships. The only layer this file
adds on top of production flags is the StreamMoE TTFT patch flags
(--moe-eager-load / --streammoe-warmup / --moe-keep-warm) for TTFT mode.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

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
# Quality-mode config registry (ported from quality_compare.py)
# --------------------------------------------------------------------------- #
#
# The five configurations we pairwise-judge against the frontier references.
# Three reuse the production streammoe_bench models (qwen36 Q4 + BF16);
# two are locally-registered additions (Qwen3.5 9B dense + 122B-A10B MoE).
# Flag-builder functions mirror the production flags ships — not the TTFT
# patch layers, which are TTFT-mode-only.

from types import SimpleNamespace

_QUALITY_EXTRA_MODELS = {
    "qwen35_9b": SimpleNamespace(
        key="qwen35_9b",
        label="Qwen3.5-9B-Q4_K_M",
        model_path=Path("/Users/christopherhastings/Downloads/qwen35-new-models/Qwen3.5-9B-Q4_K_M.gguf"),
        sidecar_dir=None,       # dense — no sidecar
        total_experts=0,
        moe_layers=0,
    ),
    "qwen35_122b": SimpleNamespace(
        key="qwen35_122b",
        label="Qwen3.5-122B-A10B-UD-Q4_K_XL",
        model_path=Path("/Users/christopherhastings/Downloads/qwen35-new-models/Qwen3.5-122B-A10B-UD-Q4_K_XL-hf/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf"),
        sidecar_dir=Path("/Users/claude/streammoe/models/qwen35-122b-sidecar"),
        total_experts=128,
        moe_layers=48,
    ),
}


def _q4_resident_flags(model):
    # Stock all-resident; fits in 48 GiB for Q4.
    return []


def _q4_streaming_flags(model):
    return [
        "--moe-sidecar", str(model.sidecar_dir),
        "--moe-mode", "slot-bank", "--moe-slot-bank", "256",
        "--moe-prefetch-temporal",
        "--flash-attn", "on",
    ]


def _bf16_streaming_flags(model):
    return [
        "--moe-sidecar", str(model.sidecar_dir),
        "--moe-mode", "slot-bank", "--moe-slot-bank", "128",
        "--moe-prefetch-temporal",
        "--flash-attn", "on",
    ]


def _q4_streaming_sb128_flags(model):
    # Same Q4 model as q4_streaming but with slot-bank 128 instead of 256.
    # This is the honest sustained-workload RAM datapoint: sb=256 saturates
    # at ~21 GiB (no real win vs stock's 21.5 GiB), sb=128 targets ~11 GiB
    # at a modest throughput cost. Cross-judged against haiku + sonnet so
    # we can tell whether the smaller bank trades quality for RAM or both.
    return [
        "--moe-sidecar", str(model.sidecar_dir),
        "--moe-mode", "slot-bank", "--moe-slot-bank", "128",
        "--moe-prefetch-temporal",
        "--flash-attn", "on",
    ]


def _qwen35_9b_resident_flags(model):
    # Dense Q4 — flash-attn only, no sidecar / slot-bank. ~6 GB RAM.
    return ["--flash-attn", "on"]


def _qwen35_122b_streaming_flags(model):
    # 122B-A10B: ~66 GB sidecar; slot-bank 128 to keep Metal budget sane.
    return [
        "--moe-sidecar", str(model.sidecar_dir),
        "--moe-mode", "slot-bank", "--moe-slot-bank", "128",
        "--moe-prefetch-temporal",
        "--flash-attn", "on",
    ]


QUALITY_CONFIGS: dict[str, dict] = {
    "q4_resident":        {"model_key": "qwen36",     "flags": _q4_resident_flags,
                           "label": "GGUF Q4_K_XL resident (stock)"},
    "q4_streaming":       {"model_key": "qwen36",     "flags": _q4_streaming_flags,
                           "label": "GGUF Q4_K_XL slot-bank streaming (sb=256)"},
    "q4_streaming_sb128": {"model_key": "qwen36",     "flags": _q4_streaming_sb128_flags,
                           "label": "GGUF Q4_K_XL slot-bank streaming (sb=128)"},
    "bf16_streaming":     {"model_key": "qwen36bf16", "flags": _bf16_streaming_flags,
                           "label": "GGUF BF16 slot-bank streaming (sb=128)"},
    "qwen35_9b":          {"model_key": "qwen35_9b",   "flags": _qwen35_9b_resident_flags,
                           "label": "Qwen3.5-9B Q4_K_M resident (dense)"},
    "qwen35_122b":        {"model_key": "qwen35_122b", "flags": _qwen35_122b_streaming_flags,
                           "label": "Qwen3.5-122B-A10B slot-bank streaming (sb=128)"},
}


def _resolve_quality_model(model_key: str):
    """Look up a QUALITY_CONFIGS model by key, falling back to the
    production streammoe_bench registry if it's not a locally-registered
    extra (e.g. qwen36, qwen36bf16 live in production)."""
    if model_key in _QUALITY_EXTRA_MODELS:
        return _QUALITY_EXTRA_MODELS[model_key]
    return get_models()[model_key]


def _start_quality_server(binary: str, model, flags: list, port: int,
                          log_path: Path, ctx_size: int) -> subprocess.Popen:
    """Spawn llama-server for a quality-mode cell. Separate from TTFT's
    start_server because the signature is simpler — we already have a
    composed flag list."""
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


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Atomic-write helper (write-to-.tmp then rename)
# --------------------------------------------------------------------------- #

def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically. Crash between write and rename
    leaves the original file untouched — no corrupted partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # tempfile.NamedTemporaryFile in the same dir so rename is always atomic
    # (same filesystem). delete=False so we can close+rename ourselves.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                                prefix=path.name + ".",
                                suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise


# --------------------------------------------------------------------------- #
# Per-prompt structured logging + memory pressure
# --------------------------------------------------------------------------- #
#
# Alongside each responses_<config>.json we write a sibling
# responses_<config>.log.jsonl — one line per prompt — capturing wall time,
# RSS, decode tok/s, and macOS memory_pressure level. The log is a sidecar
# so quality_compare-compatible consumers don't see a schema change; when a
# sampling run goes sideways (OOM, Metal allocation failure, kernel swap
# thrash), the log is what tells us whether memory pressure preceded it.
#
# get_memory_pressure() shells out to the macOS `memory_pressure` tool once
# per prompt (between prompts, never during generation). The subprocess
# adds ~50-100 ms so we accept that cost once per prompt rather than
# polling continuously.

def _parse_memory_pressure_output(output: str) -> str:
    """Parse macOS `memory_pressure` stdout → normal/warning/critical/unknown.

    The tool does NOT emit the words normal/warning/critical. It emits
    a line of the form:
        System-wide memory free percentage: 39%
    among several pages of stats. We read the free-percentage line and
    bucket by:
        >= 20% free  -> normal
        10-19% free  -> warning
        <  10% free  -> critical
    Returns "unknown" when the expected line is absent or unparseable.
    """
    needle = "System-wide memory free percentage:"
    for line in output.splitlines():
        if needle in line:
            try:
                pct = int(line.split(":", 1)[1].strip().rstrip("%"))
            except (ValueError, IndexError):
                return "unknown"
            if pct >= 20:
                return "normal"
            if pct >= 10:
                return "warning"
            return "critical"
    return "unknown"


def get_memory_pressure() -> str:
    """Sample macOS memory pressure via free percentage.

    Delegates parsing to _parse_memory_pressure_output so tests can hit
    the logic directly without shelling out. Never raises — a missing
    `memory_pressure` binary (e.g. on Linux CI) or a stalled subprocess
    both return "unknown".
    """
    try:
        out = subprocess.check_output(["memory_pressure"], text=True, timeout=5)
    except Exception:
        return "unknown"
    return _parse_memory_pressure_output(out)


def get_log_path(responses_path: Path) -> Path:
    """responses_bf16_streaming.json  →  responses_bf16_streaming.log.jsonl"""
    p = Path(responses_path)
    # with_suffix("") strips .json; then replace with .log.jsonl
    return p.with_suffix("").with_suffix(".log.jsonl")


def build_log_entry(prompt_id: str, wall_s: float, rss_bytes: int | None,
                    memory_pressure: str, decode_tps: float, n_tokens: int,
                    timestamp: str | None = None) -> dict:
    """One structured log row. rss_gib is rounded convenience alongside the
    raw byte count so a human tailing the log doesn't have to divide."""
    return {
        "prompt_id": prompt_id,
        "wall_s": round(float(wall_s), 3),
        "rss_bytes": rss_bytes,
        "rss_gib": round(rss_bytes / 1024**3, 2) if rss_bytes else None,
        "memory_pressure": memory_pressure,
        "decode_tps": round(float(decode_tps), 2),
        "n_tokens": int(n_tokens),
        "timestamp": timestamp or datetime.utcnow().isoformat(),
    }


def append_log_entry(log_path: Path, entry: dict) -> None:
    """Append one JSON line to the log file. Never raises — logging must
    not break the sampling loop."""
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Qwen3 <think>-block handling
# --------------------------------------------------------------------------- #
#
# Qwen3 family models (including the 35B-A3B MoEs and Qwen3.5 9B / 122B-A10B)
# produce reasoning inside <think>...</think> before the actual answer. The
# judge should evaluate the post-think answer, not the reasoning trace —
# otherwise a model that thinks carefully but answers tersely gets graded
# down on style of its scratchpad rather than its conclusion. When the
# response is *all* thinking (n_predict cap hit mid-thought), we surface
# it as a distinct "no_answer" verdict rather than passing an empty string
# to the judge.

def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks and any unclosed <think>...EOF."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def is_no_answer(raw: str) -> bool:
    """True when the response has only <think> content and no actual answer.

    Includes two cases:
      - closed <think>...</think> followed by nothing — an answer block
        was never produced.
      - unclosed <think> to EOF — budget cap hit mid-thought.
    """
    if not raw.strip():
        # Empty input: arguably no_answer, but the caller already treats
        # empty as "no response" elsewhere. Keep False to stay in sync.
        return False
    return not strip_think(raw)


# --------------------------------------------------------------------------- #
# Valid judge verdicts
# --------------------------------------------------------------------------- #

_VALID_VERDICTS = {"same", "similar", "different", "no_answer", "parse_error"}


# --------------------------------------------------------------------------- #
# Quality-mode records
# --------------------------------------------------------------------------- #

@dataclass
class QualityResult:
    """One model's answer to one MT-Bench prompt, plus per-request telemetry.

    Schema matches what quality_compare.py used to write so the existing
    judge consumers keep working unchanged: prompt_id / category / prompt /
    content / n_tokens / decode_tps / ttft_s / wall_s. rss_bytes is new —
    captured per prompt so we can trace memory growth/pressure per config.
    """
    prompt_id: str
    category: str
    prompt: str
    content: str
    n_tokens: int
    decode_tps: float
    ttft_s: float
    wall_s: float
    rss_bytes: int | None = None
    memory_pressure: str | None = None  # macOS memory_pressure at sample time

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JudgeVerdict:
    """One pairwise judge verdict (local vs reference, one prompt).

    `verdict` is validated at construction time so downstream aggregation
    never has to guard against typos in judge output — a model returning
    garbage is caught as parse_error earlier.
    """
    prompt_id: str
    category: str
    verdict: str
    reasoning: str

    def __post_init__(self):
        if self.verdict not in _VALID_VERDICTS:
            raise ValueError(
                f"invalid verdict {self.verdict!r}; "
                f"expected one of {sorted(_VALID_VERDICTS)}"
            )

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_quality_prompts(path: Path) -> list[dict]:
    """Load the 80 MT-Bench prompts (one per JSONL line)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"prompts file not found: {path}")
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_responses_for_judge(path: Path) -> dict[str, dict]:
    """Load responses_<config>.json and key by prompt_id."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"responses file not found: {path}")
    blob = json.loads(path.read_text())
    return {r["prompt_id"]: r for r in blob.get("responses", [])
            if r.get("content")}


def load_reference_answers(path: Path) -> dict[str, dict]:
    """Load mtbench80_<model>.jsonl (reference answers) and key by id."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"reference file not found: {path}")
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            if rec.get("content"):
                out[rec["id"]] = rec
    return out


def load_completed_prompt_ids(path: Path) -> set[str]:
    """Prompt IDs already present in a responses_<config>.json file."""
    path = Path(path)
    if not path.exists():
        return set()
    try:
        blob = json.loads(path.read_text())
    except json.JSONDecodeError:
        # Corrupt partial — rerun from scratch rather than risk double-counting.
        return set()
    return {r["prompt_id"] for r in blob.get("responses", [])
            if r.get("prompt_id")}


def load_completed_judge_ids(path: Path) -> set[str]:
    """Prompt IDs already judged in a judge_<local>_vs_<ref>.json file."""
    path = Path(path)
    if not path.exists():
        return set()
    try:
        blob = json.loads(path.read_text())
    except json.JSONDecodeError:
        return set()
    return {v["prompt_id"] for v in blob.get("verdicts", [])
            if v.get("prompt_id") and v.get("verdict")}


# --------------------------------------------------------------------------- #
# Savers (atomic)
# --------------------------------------------------------------------------- #

def save_quality_responses(results: Iterable[QualityResult], name: str,
                           label: str, load_s: float, flags: list,
                           out_path: Path) -> None:
    """Write a responses_<config>.json atomically.

    Includes every QualityResult's full response text + timing + RSS.
    Called after every prompt so a crash loses at most one in-flight
    sample.
    """
    blob = {
        "name": name,
        "label": label,
        "load_s": load_s,
        "flags": list(flags),
        "responses": [r.to_dict() for r in results],
    }
    _atomic_write_text(Path(out_path), json.dumps(blob, indent=2))


def save_judge_results(out_path: Path, local_name: str, ref_name: str,
                       judge_model: str, counts: dict, by_category: dict,
                       verdicts: Iterable, n_compared: int) -> None:
    """Write a judge_<local>_vs_<ref>.json atomically.

    `verdicts` may be a list of JudgeVerdict objects OR plain dicts (the
    judge loop keeps both forms around) — both normalize to the same
    on-disk shape.
    """
    serialized = []
    for v in verdicts:
        if isinstance(v, JudgeVerdict):
            serialized.append(v.to_dict())
        else:
            serialized.append(dict(v))
    blob = {
        "local": local_name,
        "reference": ref_name,
        "judge_model": judge_model,
        "counts": dict(counts),
        "by_category": {k: dict(vv) for k, vv in by_category.items()},
        "verdicts": serialized,
        "n_compared": int(n_compared),
    }
    _atomic_write_text(Path(out_path), json.dumps(blob, indent=2))


# --------------------------------------------------------------------------- #
# LM Studio readiness
# --------------------------------------------------------------------------- #

def check_lm_studio_ready(endpoint: str, api_key: str) -> tuple[bool, list]:
    """Hit /v1/models and return (is_reachable, loaded_model_ids).

    Never raises: a dead endpoint / wrong URL / missing openai package all
    surface as (False, []). This is called before judge mode kicks off so
    the user gets an actionable error immediately, rather than N minutes
    of parse_errors down the line.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return False, []
    try:
        client = OpenAI(base_url=endpoint, api_key=api_key)
        models = client.models.list()
        return True, [m.id for m in models.data]
    except Exception:
        return False, []


def is_model_loaded(model_id: str, loaded_models: list) -> bool:
    return model_id in loaded_models


# --------------------------------------------------------------------------- #
# Quality-mode response collection
# --------------------------------------------------------------------------- #


def clear_kv_cache(port: int, session) -> None:
    """Tell llama-server to drop its KV cache between prompts.

    Each MT-Bench prompt is meant to be measured against a fresh context
    — otherwise the previous answer's tokens bleed into the attention
    history and cross-prompt quality becomes incomparable. The endpoint
    is cheap (~10 ms); failure is non-fatal.
    """
    try:
        session.post(f"http://127.0.0.1:{port}/cache/clear", timeout=5)
    except Exception:
        pass  # non-fatal — sampling loop continues


def collect_quality_response(session, port: int, prompt_id: str,
                             category: str, prompt: str,
                             n_predict: int, seed: int,
                             server_pid: int | None = None,
                             timeout_s: float = 600.0,
                             log_path: Path | None = None) -> QualityResult:
    """Fire one prompt at a running llama-server, parse an OpenAI-compatible
    chat-completions response, return a QualityResult.

    `session` is a requests-like object with a .post(url, json=..., timeout=...)
    method; the tests use a MagicMock. If `server_pid` is given, sample
    its RSS after the response arrives so the caller can plot memory
    usage per prompt. If `log_path` is given, one JSON line per prompt
    is appended there with {prompt_id, wall_s, rss_bytes, rss_gib,
    memory_pressure, decode_tps, n_tokens, timestamp}.

    Memory pressure is sampled AFTER the response arrives, i.e. between
    prompts — never during generation, so the memory_pressure subprocess
    doesn't compete with the running model for scheduling.
    """
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "max_tokens": n_predict,
        "stream": False,
    }
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    t0 = time.perf_counter()
    resp = session.post(url, json=body, timeout=timeout_s)
    t_end = time.perf_counter()
    data = resp.json()

    # OpenAI-compatible shape — all the llama.cpp / vllm / lmstudio servers
    # emit this. Fall back gracefully if a field is missing.
    choices = data.get("choices") or [{}]
    msg = choices[0].get("message") or {}
    content = msg.get("content") or ""
    usage = data.get("usage") or {}
    n_tok = int(usage.get("completion_tokens") or 0)
    timings = data.get("timings") or {}
    ttft_s = float(timings.get("prompt_ms", 0)) / 1000.0 if timings else 0.0
    wall_s = t_end - t0
    decode_tps = (n_tok / max(wall_s - ttft_s, 1e-6)) if n_tok else 0.0

    # Clear KV cache between prompts so each one is measured against a
    # fresh context (cross-prompt bleed would make the quality comparison
    # incomparable). Non-fatal on failure.
    clear_kv_cache(port, session)

    rss = process_rss_bytes(server_pid) if server_pid else None
    # Pressure check runs now — after the server response is in hand but
    # before we return to the sampling loop — so it's between prompts.
    pressure = get_memory_pressure() if log_path is not None else None

    result = QualityResult(
        prompt_id=prompt_id, category=category, prompt=prompt,
        content=content, n_tokens=n_tok, decode_tps=decode_tps,
        ttft_s=ttft_s, wall_s=wall_s, rss_bytes=rss,
        memory_pressure=pressure,
    )
    if log_path is not None:
        append_log_entry(log_path, build_log_entry(
            prompt_id=prompt_id,
            wall_s=wall_s,
            rss_bytes=rss,
            memory_pressure=pressure or "unknown",
            decode_tps=decode_tps,
            n_tokens=n_tok,
        ))
    return result


# --------------------------------------------------------------------------- #
# Judge calls
# --------------------------------------------------------------------------- #

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert judge evaluating LLM response quality. "
    "Given a prompt and two responses — Reference and Test — judge whether "
    "the Test answer is equivalent in quality to the Reference. "
    "Focus on correctness, completeness, and usefulness. "
    "Ignore length and formatting differences. "
    "You must call the submit_verdict function with your verdict."
)


def _user_msg_for_judge(prompt: str, ref: str, local: str) -> str:
    return (
        f"PROMPT:\n{prompt}\n\n"
        f"REFERENCE:\n{ref[:1200]}\n\n"
        f"TEST:\n{local[:1200]}"
    )


def judge_pair_local(client, prompt: str, ref_content: str,
                     local_content: str, model: str,
                     retries: int = 2, server_pid=None) -> dict:
    """Judge a pair via an OpenAI-compatible local endpoint (LM Studio,
    vLLM, etc.) with forced tool_use for strict-JSON verdict.

    Retries on any exception with exponential backoff (2^attempt seconds).
    Returns {"verdict": "parse_error", "reasoning": <err>} if the model
    never emits a tool_call after all retries.
    """
    tool = {
        "type": "function",
        "function": {
            "name": "submit_verdict",
            "description": "Submit quality verdict",
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["same", "similar", "different"],
                        "description": (
                            "same=equivalent quality, "
                            "similar=minor differences, "
                            "different=meaningful quality gap"
                        ),
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["verdict", "reasoning"],
            },
        },
    }
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user",   "content": _user_msg_for_judge(
            prompt, ref_content, local_content)},
    ]
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[tool],
                tool_choice={"type": "function",
                             "function": {"name": "submit_verdict"}},
                max_tokens=300,
                temperature=0,
            )
            for choice in resp.choices:
                tool_calls = getattr(choice.message, "tool_calls", None)
                if tool_calls:
                    return json.loads(tool_calls[0].function.arguments)
            # Model responded but didn't call the tool — retry (maybe prompt
            # was misformatted, temperature noise, etc).
        except Exception as e:
            last_err = e
            if attempt == retries:
                return {"verdict": "parse_error", "reasoning": str(e)}
            time.sleep(2 ** attempt)
    return {"verdict": "parse_error",
            "reasoning": f"no tool_call after {retries + 1} attempts"
                         + (f": {last_err}" if last_err else "")}


def judge_pair_anthropic(client, prompt: str, ref_content: str,
                         local_content: str, model: str,
                         server_pid=None) -> dict:
    """Judge via the hosted Claude API using the existing quality_gates
    implementation in ~/streammoe/streammoe_bench."""
    from streammoe_bench.quality_gates import judge_pair
    # judge_pair is positional: (client, prompt, stock, sidecar, model=...)
    v = judge_pair(client, prompt, ref_content, local_content, model=model)
    # normalize key name — quality_gates returns {"verdict","score","reason"};
    # our on-disk schema uses "reasoning" for judge rationale.
    if "reasoning" not in v and "reason" in v:
        v["reasoning"] = v["reason"]
    return v


def run_judge_pair_set(local_name: str, ref_name: str,
                       local_responses: dict, ref_responses: dict,
                       judge_fn: Callable,
                       out_path: Path | None = None,
                       judge_model: str = "",
                       prompts: dict | None = None) -> dict:
    """Run judge_fn on every (prompt_id) present in both local_responses
    and ref_responses, short-circuiting all-think no_answer cases.

    Resumable: if `out_path` exists, already-judged prompt IDs are skipped
    and the existing counts are carried forward.

    `judge_fn` signature: (prompt, ref_content, local_content) -> dict
    with at least a "verdict" key. `judge_model` is only carried through
    to the saved output for provenance.
    """
    EMPTY_COUNTS = {"same": 0, "similar": 0, "different": 0,
                    "no_answer": 0, "parse_error": 0}

    common_ids = sorted(set(local_responses) & set(ref_responses))
    done_ids: set[str] = set()
    counts = dict(EMPTY_COUNTS)
    by_category: dict[str, dict] = {}
    verdicts: list[dict] = []

    if out_path is not None:
        out_path = Path(out_path)
        if out_path.exists():
            try:
                prior = json.loads(out_path.read_text())
                for v in prior.get("verdicts", []):
                    if v.get("prompt_id") and v.get("verdict"):
                        done_ids.add(v["prompt_id"])
                        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
                        cat = v.get("category") or "unknown"
                        bc = by_category.setdefault(cat, dict(EMPTY_COUNTS))
                        bc[v["verdict"]] = bc.get(v["verdict"], 0) + 1
                        verdicts.append(dict(v))
            except Exception:
                # Corrupt partial — start over rather than risk double-count.
                done_ids.clear(); counts = dict(EMPTY_COUNTS)
                by_category.clear(); verdicts.clear()

    for pid in common_ids:
        if pid in done_ids:
            continue
        local = local_responses[pid]
        ref   = ref_responses[pid]
        prompt = (prompts or {}).get(pid, {}).get("prompt") \
                 or local.get("prompt") or ref.get("prompt") or ""
        category = (prompts or {}).get(pid, {}).get("category") \
                 or local.get("category") or ref.get("category") or "unknown"

        raw_local = local.get("content", "")
        if is_no_answer(raw_local):
            v = {"verdict": "no_answer",
                 "reasoning": "Model produced only <think> tokens; "
                              "no actual answer."}
        else:
            stripped = strip_think(raw_local)
            v = judge_fn(prompt, ref.get("content", ""), stripped)

        verdict = v.get("verdict", "parse_error")
        reasoning = v.get("reasoning") or v.get("reason") or ""
        counts[verdict] = counts.get(verdict, 0) + 1
        bc = by_category.setdefault(category, dict(EMPTY_COUNTS))
        bc[verdict] = bc.get(verdict, 0) + 1
        entry = {"prompt_id": pid, "category": category,
                 "verdict": verdict, "reasoning": reasoning}
        verdicts.append(entry)

        if out_path is not None:
            save_judge_results(out_path, local_name, ref_name, judge_model,
                               counts, by_category, verdicts, len(common_ids))

    return {
        "local": local_name,
        "reference": ref_name,
        "judge_model": judge_model,
        "counts": counts,
        "by_category": by_category,
        "verdicts": verdicts,
        "n_compared": len(common_ids),
    }


# --------------------------------------------------------------------------- #
# Server lifecycle (TTFT mode only)
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Single CLI for all three modes. Extracted so tests can parse without
    invoking main() side-effects."""
    ap = argparse.ArgumentParser(
        description="StreamMoE benchmark harness (ttft / quality / judge)."
    )
    ap.add_argument("--mode", choices=["ttft", "quality", "judge"],
                    default="ttft",
                    help="which benchmark to run (default: ttft).")

    # Shared
    ap.add_argument("--binary",
                    default="/Users/claude/streammoe/anemll-flash-llama.cpp/build/bin/llama-server")
    ap.add_argument("--output-dir", default="./ttft-results",
                    help="mode=ttft: results dir. mode=quality: ./quality-results/compare overrides. mode=judge: where judge_*.json land.")
    ap.add_argument("--prompts", default="mtbench80.jsonl",
                    help="MT-Bench prompts file (quality + judge modes).")
    ap.add_argument("--n-predict", type=int, default=3000,
                    help="max tokens per request (default 3000 — Qwen3 <think> blocks can consume 1k+ alone).")
    ap.add_argument("--models", default=",".join(DEFAULT_MODEL_KEYS),
                    help="comma-separated keys from streammoe_bench.config.get_models() (ttft mode).")
    ap.add_argument("--only", default="",
                    help="quality mode: comma-separated subset of configs to sample.")

    # TTFT mode
    ap.add_argument("--iters", type=int, default=3,
                    help="ttft mode: cold+warm iterations per cell.")
    ap.add_argument("--prompt", default="Briefly: what is 2+2?",
                    help="ttft mode: synthetic prompt.")
    ap.add_argument("--layers", default="",
                    help="ttft mode: subset of {baseline,L1_eager,L1L2_warmup,L1L2L3L4_full}.")

    # Quality + judge shared paths
    ap.add_argument("--compare-dir", default="./quality-results/compare",
                    help="judge mode: where responses_<config>.json live.")
    ap.add_argument("--ref-dir", default="./reference-answers",
                    help="judge mode: where mtbench80_<model>.jsonl references live.")
    ap.add_argument("--locals", default="",
                    help="judge mode: comma-separated local config names to judge (default: every responses_*.json in --compare-dir).")
    ap.add_argument("--refs", default="haiku,sonnet",
                    help="judge mode: comma-separated reference model names.")

    # Judge mode
    ap.add_argument("--judge-model", default="claude-sonnet-4-6",
                    help="judge mode: model id (claude-* for hosted, or the LM Studio model id when --judge-endpoint is set).")
    ap.add_argument("--judge-endpoint", default=None,
                    help="judge mode: OpenAI-compatible base URL for local judge (e.g. http://localhost:1234/v1 for LM Studio). If set, bypasses the Anthropic API.")
    ap.add_argument("--judge-api-key", default="lm-studio",
                    help="judge mode: api key for local judge endpoint (default 'lm-studio'; most local servers ignore value).")
    ap.add_argument("--skip-sampling", action="store_true",
                    help="judge mode: assume responses_<config>.json already exist; just judge them.")
    return ap


# --------------------------------------------------------------------------- #
# Mode dispatchers
# --------------------------------------------------------------------------- #

def run_ttft_mode(args) -> int:
    """Original TTFT matrix. Spawns a fresh server per cell, measures cold
    + warm first-token latency, emits a JSON matrix."""
    out = Path(args.output_dir)
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    layer_names = [l.strip() for l in args.layers.split(",") if l.strip()]
    # TTFT uses a smaller n_predict by default so short prompts don't sit
    # through a full 3k-token generation per cell. Respect the user override
    # if they passed it, but cap at 32 for the synthetic prompt.
    n_predict = args.n_predict if args.n_predict < 3000 else 32
    results = run_matrix(args.binary, out, args.iters, args.prompt,
                         n_predict, model_keys, layer_names)
    path = out / f"ttft_matrix_{int(time.time())}.json"
    _atomic_write_text(path, json.dumps(results, indent=2))
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


def run_quality_mode(args) -> int:
    """80-prompt MT-Bench sampler across QUALITY_CONFIGS.

    Resumable per config: if responses_<config>.json already contains
    some prompts, those are skipped and new ones are appended. Writes
    the full file atomically after every prompt, so a crash loses at
    most one in-flight sample.
    """
    try:
        import requests
    except ImportError:
        print("[fatal] pip install requests  (needed for --mode quality)",
              file=sys.stderr)
        return 2

    prompts_file = Path(args.prompts)
    if not prompts_file.exists():
        prompts_file = Path(__file__).parent / args.prompts
    prompts = load_quality_prompts(prompts_file)

    out_dir = Path(args.compare_dir if args.compare_dir else args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    selected = [n for n in QUALITY_CONFIGS if (not only or n in only)]
    if not selected:
        print(f"[fatal] --only {args.only!r} matched no configs in "
              f"{list(QUALITY_CONFIGS)}", file=sys.stderr)
        return 2

    n_predict = args.n_predict
    ctx_size = 8192
    session = requests.Session()

    for name in selected:
        cfg = QUALITY_CONFIGS[name]
        model = _resolve_quality_model(cfg["model_key"])
        flags = cfg["flags"](model)
        port = 11500 + (abs(hash(name)) % 80)
        log_path = out_dir / f"server_{name}.log"
        out_path = out_dir / f"responses_{name}.json"

        # Resumability — carry forward existing responses and skip their
        # prompt IDs. Re-derives flags/label from the current registry
        # regardless of what the old file recorded, so a changed flag set
        # is reflected on the next save.
        existing_results: list[QualityResult] = []
        completed_ids = load_completed_prompt_ids(out_path)
        if completed_ids and out_path.exists():
            try:
                blob = json.loads(out_path.read_text())
                for r in blob.get("responses", []):
                    if not r.get("prompt_id"): continue
                    existing_results.append(QualityResult(
                        prompt_id=r["prompt_id"],
                        category=r.get("category") or "unknown",
                        prompt=r.get("prompt", ""),
                        content=r.get("content", ""),
                        n_tokens=int(r.get("n_tokens") or 0),
                        decode_tps=float(r.get("decode_tps") or 0.0),
                        ttft_s=float(r.get("ttft_s") or 0.0),
                        wall_s=float(r.get("wall_s") or 0.0),
                        rss_bytes=r.get("rss_bytes"),
                    ))
            except Exception as e:
                print(f"  [warn] couldn't resume {name}: {e} — starting fresh",
                      file=sys.stderr)
                existing_results = []; completed_ids = set()

        todo = [p for p in prompts if p["id"] not in completed_ids]
        if not todo:
            print(f"[{name}] already complete ({len(existing_results)}/80)")
            continue

        print(f"\n=== sampling {name} ({cfg['label']}) on port {port} ===",
              flush=True)
        if completed_ids:
            print(f"[{name}] resuming — {len(completed_ids)}/80 already done",
                  flush=True)

        t_load = time.perf_counter()
        proc = _start_quality_server(args.binary, model, flags, port,
                                      log_path, ctx_size)
        deadline = 1500 if "122b" in name else (
                    900 if "bf16" in name else 360)
        if not wait_ready(port, time.monotonic() + deadline):
            print(f"[{name}] server did not become ready", file=sys.stderr)
            stop_server(proc)
            continue
        load_s = time.perf_counter() - t_load
        time.sleep(2.0)  # warmup window

        results = list(existing_results)
        log_path = get_log_path(out_path)
        # Exit-safety: if memory_pressure reports "critical" for three
        # prompts in a row, stop this cell cleanly with what we've got
        # rather than risk a kernel-level hang. macOS under sustained
        # critical memory pressure starts compressing/swapping aggressively
        # and an M-class Metal workload can wedge the graphics stack.
        consecutive_critical = 0
        bailed = False
        try:
            for i, p in enumerate(todo):
                try:
                    r = collect_quality_response(
                        session=session, port=port,
                        prompt_id=p["id"], category=p.get("category", "unknown"),
                        prompt=p["prompt"],
                        n_predict=n_predict, seed=42,
                        server_pid=proc.pid,
                        log_path=log_path,
                    )
                    results.append(r)
                except Exception as e:
                    print(f"  [err] prompt {i} ({p.get('id')}): {e}",
                          file=sys.stderr)
                    results.append(QualityResult(
                        prompt_id=p["id"],
                        category=p.get("category", "unknown"),
                        prompt=p["prompt"], content="",
                        n_tokens=0, decode_tps=0.0,
                        ttft_s=0.0, wall_s=0.0, rss_bytes=None))
                # Per-prompt atomic checkpoint.
                save_quality_responses(results, name, cfg["label"],
                                        load_s, flags, out_path)

                # Memory-pressure watchdog — read pressure off the last
                # result (collect_quality_response stamped it there).
                pressure = getattr(results[-1], "memory_pressure", None)
                if pressure == "critical":
                    consecutive_critical += 1
                    print(f"  [WARN] memory pressure: critical at "
                          f"{results[-1].prompt_id} "
                          f"(streak {consecutive_critical})",
                          flush=True)
                elif pressure == "warning":
                    consecutive_critical = 0
                    print(f"  [WARN] memory pressure: warning at "
                          f"{results[-1].prompt_id}", flush=True)
                else:
                    consecutive_critical = 0

                if consecutive_critical >= 3:
                    print(f"  [ABORT] {name}: memory_pressure=critical on "
                          f"3 consecutive prompts — stopping this cell "
                          f"cleanly at {len(results)}/{len(prompts)} to "
                          f"avoid a kernel freeze. Re-run later (it's "
                          f"resumable).", flush=True)
                    bailed = True
                    break

                if (i + 1) % 10 == 0 or (i + 1) == len(todo):
                    last = results[-1]
                    print(f"  {name} {len(results)}/{len(prompts)}  "
                          f"last: n_tok={last.n_tokens} "
                          f"wall={last.wall_s:.1f}s "
                          f"press={last.memory_pressure}", flush=True)
        finally:
            stop_server(proc)
        if bailed:
            # Skip remaining configs — the whole machine is under memory
            # pressure, not just this config. User can rerun with --only
            # for the remaining ones after freeing memory.
            print(f"\n[abort] stopping further configs; rerun after "
                  f"pressure subsides.", flush=True)
            return 0

    print(f"\nquality responses written under {out_dir}")
    return 0


def run_judge_mode(args) -> int:
    """Pairwise judge runner — local × reference."""
    compare_dir = Path(args.compare_dir)
    ref_dir     = Path(args.ref_dir)
    out_dir     = Path(args.compare_dir)  # judge_*.json co-located with responses
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide which judge path to use + verify readiness BEFORE sampling.
    if args.judge_endpoint:
        ready, loaded = check_lm_studio_ready(args.judge_endpoint,
                                              args.judge_api_key)
        if not ready:
            print(f"[fatal] LM Studio not reachable at {args.judge_endpoint}",
                  file=sys.stderr)
            print("        Start LM Studio and ensure the server is running "
                  "on that port.", file=sys.stderr)
            return 1
        if not is_model_loaded(args.judge_model, loaded):
            print(f"[fatal] Model '{args.judge_model}' is not loaded in "
                  f"LM Studio.", file=sys.stderr)
            print(f"        Loaded models: {loaded}", file=sys.stderr)
            print(f"        Load '{args.judge_model}' in LM Studio, "
                  f"then retry.", file=sys.stderr)
            return 1
        print(f"[ok] LM Studio ready, model '{args.judge_model}' loaded",
              flush=True)
        try:
            from openai import OpenAI
        except ImportError:
            print("[fatal] pip install openai  (--judge-endpoint requires it)",
                  file=sys.stderr)
            return 2
        judge_client = OpenAI(base_url=args.judge_endpoint,
                              api_key=args.judge_api_key)
        use_local = True
    else:
        try:
            import anthropic
        except ImportError:
            print("[fatal] pip install anthropic", file=sys.stderr); return 2
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            print("[fatal] ANTHROPIC_API_KEY not set (or pass --judge-endpoint)",
                  file=sys.stderr); return 2
        judge_client = anthropic.Anthropic(api_key=api_key)
        use_local = False

    # Optional sampling pass (if the user asks for a full run from scratch).
    if not args.skip_sampling:
        rc = run_quality_mode(args)
        if rc != 0: return rc

    # Resolve which locals to judge.
    if args.locals:
        local_names = [s.strip() for s in args.locals.split(",") if s.strip()]
    else:
        # auto-detect from compare_dir
        local_names = sorted(
            p.stem[len("responses_"):]
            for p in compare_dir.glob("responses_*.json")
        )
    ref_names = [s.strip() for s in args.refs.split(",") if s.strip()]

    prompts_file = Path(args.prompts)
    if not prompts_file.exists():
        prompts_file = Path(__file__).parent / args.prompts
    prompts = {p["id"]: p for p in load_quality_prompts(prompts_file)}

    # Load everything up front so missing files surface before we burn
    # API calls.
    locals_data: dict[str, dict] = {}
    for n in local_names:
        p = compare_dir / f"responses_{n}.json"
        try:
            locals_data[n] = load_responses_for_judge(p)
        except FileNotFoundError as e:
            print(f"[skip local] {n}: {e}", file=sys.stderr)
    refs_data = {n: load_reference_answers(ref_dir / f"mtbench80_{n}.jsonl")
                 for n in ref_names}

    # Construct a closure that hides which backend we're using so
    # run_judge_pair_set doesn't have to branch on it.
    if use_local:
        def judge_fn(prompt, ref, local):
            return judge_pair_local(judge_client, prompt, ref, local,
                                     model=args.judge_model)
    else:
        def judge_fn(prompt, ref, local):
            return judge_pair_anthropic(judge_client, prompt, ref, local,
                                         model=args.judge_model)

    grid = []
    for ln in local_names:
        if ln not in locals_data: continue
        for rn in ref_names:
            out_path = out_dir / f"judge_{ln}_vs_{rn}.json"
            print(f"\n=== judging {ln} vs {rn} ===", flush=True)
            result = run_judge_pair_set(
                local_name=ln, ref_name=rn,
                local_responses=locals_data[ln], ref_responses=refs_data[rn],
                judge_fn=judge_fn,
                out_path=out_path,
                judge_model=args.judge_model,
                prompts=prompts,
            )
            grid.append(result)

    # Summary
    print(f"\n{'local':20s} {'reference':10s} {'n':>4s} "
          f"{'same':>5s} {'sim':>5s} {'diff':>5s} {'na':>4s} {'equiv%':>7s}")
    for r in grid:
        c = r["counts"]
        total = sum(c.values())
        equiv = c.get("same", 0) + c.get("similar", 0)
        print(f"{r['local']:20s} {r['reference']:10s} {r['n_compared']:>4d} "
              f"{c.get('same',0):>5d} {c.get('similar',0):>5d} "
              f"{c.get('different',0):>5d} {c.get('no_answer',0):>4d} "
              f"{100*equiv/max(1,total):>7.1f}")
    return 0


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.mode == "ttft":
        return run_ttft_mode(args)
    if args.mode == "quality":
        return run_quality_mode(args)
    if args.mode == "judge":
        return run_judge_mode(args)
    print(f"[fatal] unknown mode: {args.mode}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
