#!/usr/bin/env python3
"""One-time migration: retroactively fix decode_tps in saved responses.

Walks every responses_<config>.json under quality-results/compare/ and
its sibling .log.jsonl and rewrites decode_tps using the corrected
compute_decode_tps() logic (wall-math only: n_tok / (wall_s - ttft_s),
0.0 when the denominator is non-positive).

Also parses the matching `slot print_timing` block from server_<config>.log
to attach server-side timings as metadata on each fixed row (NOT used for
the computation — the server timer drifts on this fork — but useful for
forensics and to flag rows where server predicted_n disagrees with the
recorded n_tokens, which suggests retry/skip misalignment).

Reports a per-file summary so a human can see exactly which rows moved
and by how much. Safe to re-run (idempotent — recomputing from client
wall math always produces the same number).

Usage:
    python3.11 fix_decode_tps.py                       # all configs
    python3.11 fix_decode_tps.py --config q4_streaming_sb128
    python3.11 fix_decode_tps.py --dry-run             # report only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ttft_bench import compute_decode_tps, _atomic_write_text  # noqa: E402


# llama.cpp `slot print_timing` block format (see server_*.log):
#
#   slot print_timing: id  3 | task 0 |
#   prompt eval time =  4503.54 ms /    32 tokens (  140.74 ms per token,     7.11 tokens per second)
#          eval time = 119626.10 ms /  3000 tokens (   39.88 ms per token,    25.08 tokens per second)
#
# We pull out (prompt_ms, prompt_n, predicted_ms, predicted_n) from the
# two "eval time" lines; the prompt one has "prompt eval time =" and the
# decode one has just "       eval time =".

_PROMPT_LINE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
_DECODE_LINE = re.compile(
    r"^\s+eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", re.MULTILINE)


def parse_server_timings(log_text: str) -> list[dict]:
    """Extract (prompt_ms, prompt_n, predicted_ms, predicted_n) blocks in
    log order. One entry per llama.cpp print_timing block = one prompt.

    The two lines appear consecutively; we pair them by iterating the
    log line-by-line rather than by regex-find-all, so a missed line in
    one pair doesn't contaminate the next."""
    entries: list[dict] = []
    current: dict = {}
    for line in log_text.splitlines():
        m_p = _PROMPT_LINE.search(line)
        if m_p:
            current = {
                "prompt_ms": float(m_p.group(1)),
                "prompt_n":  int(m_p.group(2)),
            }
            continue
        m_d = _DECODE_LINE.search(line)
        if m_d and current:
            current["predicted_ms"] = float(m_d.group(1))
            current["predicted_n"]  = int(m_d.group(2))
            entries.append(current)
            current = {}
    return entries


def fix_responses_file(responses_path: Path, server_log_path: Path,
                       dry_run: bool = False) -> dict:
    """Recompute decode_tps for every row. Writes back responses_*.json
    + responses_*.log.jsonl atomically. Returns a summary dict."""
    if not responses_path.exists():
        return {"error": f"missing: {responses_path}"}
    if not server_log_path.exists():
        return {"error": f"missing server log: {server_log_path}"}

    blob = json.loads(responses_path.read_text())
    responses = blob.get("responses", [])
    timings = parse_server_timings(server_log_path.read_text())

    n_resp = len(responses)
    n_timings = len(timings)
    # Position-in-log-order matches position-in-responses-order because
    # the sampling loop sends one prompt at a time in order and waits
    # for each response. If the counts disagree something interrupted
    # the run — we still try but warn.
    if n_timings != n_resp:
        print(f"  [warn] {responses_path.name}: "
              f"{n_timings} timing blocks vs {n_resp} responses — "
              f"matching by position; last "
              f"{abs(n_timings - n_resp)} rows may be skipped/degraded",
              file=sys.stderr)

    n_fixed = 0
    n_unchanged = 0
    changes: list[dict] = []

    for i, r in enumerate(responses):
        old = float(r.get("decode_tps") or 0.0)
        n_tok = int(r.get("n_tokens") or 0)
        wall_s = float(r.get("wall_s") or 0.0)
        ttft_s = float(r.get("ttft_s") or 0.0)
        t = timings[i] if i < n_timings else {}

        # Sanity-check the pairing: the server's predicted_n should match
        # what we recorded in n_tokens. If they diverge by more than a
        # token or two something is misaligned (retry? skipped?).
        if t.get("predicted_n") and n_tok and abs(t["predicted_n"] - n_tok) > 2:
            print(f"  [warn] {responses_path.name} row {i} "
                  f"({r.get('prompt_id')}): server predicted_n="
                  f"{t.get('predicted_n')} vs recorded n_tokens={n_tok} — "
                  f"alignment suspect", file=sys.stderr)
            continue

        new = compute_decode_tps(t, n_tok, wall_s, ttft_s)
        if abs(new - old) < 0.01:
            n_unchanged += 1
            continue

        n_fixed += 1
        changes.append({
            "prompt_id": r.get("prompt_id"),
            "old_decode_tps": round(old, 2),
            "new_decode_tps": round(new, 2),
            "n_tokens": n_tok,
            "wall_s": round(wall_s, 2),
            "ttft_s": round(ttft_s, 2),
        })
        r["decode_tps"] = round(new, 2)
        # Also attach the server-truth timings for future re-derivation.
        if t:
            r["server_timings"] = {
                "prompt_ms":    t.get("prompt_ms"),
                "prompt_n":     t.get("prompt_n"),
                "predicted_ms": t.get("predicted_ms"),
                "predicted_n":  t.get("predicted_n"),
            }

    # Fix the .log.jsonl sidecar if present (keyed by prompt_id).
    log_path = responses_path.with_suffix("").with_suffix(".log.jsonl")
    log_lines_written = 0
    if log_path.exists():
        new_by_pid = {c["prompt_id"]: c["new_decode_tps"] for c in changes}
        new_lines = []
        with log_path.open() as f:
            for line in f:
                if not line.strip():
                    new_lines.append(line.rstrip("\n")); continue
                entry = json.loads(line)
                pid = entry.get("prompt_id")
                if pid in new_by_pid:
                    entry["decode_tps"] = new_by_pid[pid]
                    log_lines_written += 1
                new_lines.append(json.dumps(entry))
        if not dry_run:
            _atomic_write_text(log_path, "\n".join(new_lines) + "\n")

    if not dry_run:
        _atomic_write_text(responses_path, json.dumps(blob, indent=2))

    return {
        "file": str(responses_path.name),
        "n_responses": n_resp,
        "n_timing_blocks": n_timings,
        "n_fixed": n_fixed,
        "n_unchanged": n_unchanged,
        "log_lines_updated": log_lines_written,
        "changes": changes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-dir",
                    default="./quality-results/compare")
    ap.add_argument("--config", default="",
                    help="only fix one config (e.g. q4_streaming_sb128)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cmp_dir = Path(args.compare_dir)
    if args.config:
        targets = [cmp_dir / f"responses_{args.config}.json"]
    else:
        targets = sorted(cmp_dir.glob("responses_*.json"))

    any_changes = False
    for responses_path in targets:
        # server_q4_streaming_sb128.log  pairs with responses_q4_streaming_sb128.json
        config_name = responses_path.stem.removeprefix("responses_")
        server_log_path = cmp_dir / f"server_{config_name}.log"
        summary = fix_responses_file(responses_path, server_log_path,
                                     dry_run=args.dry_run)
        print(f"\n=== {summary.get('file', responses_path.name)} ===")
        if "error" in summary:
            print(f"  ERROR: {summary['error']}")
            continue
        print(f"  responses: {summary['n_responses']}, "
              f"server timing blocks: {summary['n_timing_blocks']}, "
              f"fixed: {summary['n_fixed']}, unchanged: {summary['n_unchanged']}, "
              f"log rows: {summary['log_lines_updated']}")
        if summary["n_fixed"]:
            any_changes = True
            print(f"  changes:")
            for c in summary["changes"]:
                print(f"    {c['prompt_id']:8s} "
                      f"{c['old_decode_tps']:>14.2f} tok/s -> "
                      f"{c['new_decode_tps']:>8.2f} tok/s  "
                      f"(n_tok={c['n_tokens']}, wall={c['wall_s']}s, "
                      f"ttft={c['ttft_s']}s)")

    if args.dry_run and any_changes:
        print("\n(dry-run: no files modified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
