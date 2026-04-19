#!/usr/bin/env python3
"""Generate reference answers for the 80 MT-Bench prompts using Claude models.

Calls the Anthropic Messages API once per prompt per model (Haiku 4.5 and
Sonnet 4.6) with the same deterministic settings our local-model harness
uses (temperature=0, max_tokens=1024, no system prompt), and writes
JSONL with per-prompt response text + usage for later judge comparison.

Safe to run alongside a Metal/llama-server benchmark — this is CPU-light
and network-bound, with a small concurrency cap so we don't bump into
Anthropic's per-minute rate limits.

Usage:
    export ANTHROPIC_API_KEY=...
    python3 generate_reference_answers.py
    python3 generate_reference_answers.py --models haiku   # one model only
    python3 generate_reference_answers.py --concurrency 2  # slower but gentler
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

# Model registry. Keep IDs as-is per shared/models.md — don't append dates.
MODELS = {
    "haiku":  "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-6",
}


def load_prompts(path: Path) -> list[dict]:
    prompts = []
    with path.open() as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def answer_one(client: anthropic.Anthropic, model_id: str, prompt: dict) -> dict:
    """Call the Messages API once. Returns the reference-answer record.

    Matches the local harness: no system prompt, temperature=0, max_tokens=1024.
    Note: Sonnet 4.6 accepts temperature=0 without error; Opus 4.7 would not.
    Haiku 4.5 doesn't support adaptive thinking, so we keep the request minimal."""
    msg = client.messages.create(
        model=model_id,
        max_tokens=1024,
        temperature=0,
        messages=[{"role": "user", "content": prompt["prompt"]}],
    )
    # Concatenate all text blocks; the models don't use thinking at temp=0/no-system,
    # but filter defensively in case that ever changes.
    text = "".join(b.text for b in msg.content if b.type == "text")
    return {
        "id":            prompt["id"],
        "category":      prompt.get("category"),
        "prompt":        prompt["prompt"],
        "content":       text,
        "model":         model_id,
        "stop_reason":   msg.stop_reason,
        "input_tokens":  msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
    }


def run_for_model(model_key: str, prompts: list[dict], out_dir: Path,
                  concurrency: int) -> Path:
    model_id = MODELS[model_key]
    out_path = out_dir / f"mtbench80_{model_key}.jsonl"

    # Resume support: if we already have some responses, skip those prompt IDs.
    seen_ids: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    seen_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    if seen_ids:
        print(f"[{model_key}] resuming — {len(seen_ids)}/{len(prompts)} already complete",
              flush=True)
    todo = [p for p in prompts if p["id"] not in seen_ids]
    if not todo:
        print(f"[{model_key}] nothing to do — {out_path} is complete")
        return out_path

    # Strip whitespace/newlines from the API key. Some shells or .env loaders
    # leave a trailing '\n' that curl tolerates but httpx rejects as an
    # illegal header value ("LocalProtocolError: Illegal header value").
    api_key = os.environ["ANTHROPIC_API_KEY"].strip()
    client = anthropic.Anthropic(api_key=api_key)

    # Append each record as soon as it's done (line-buffered) so a kill/crash
    # preserves everything up to that point — matches resume logic above.
    fout = out_path.open("a", buffering=1)

    errors: list[tuple[str, str]] = []
    t_start = time.monotonic()
    completed = len(seen_ids)
    total = len(prompts)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(answer_one, client, model_id, p): p for p in todo}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:
                errors.append((p["id"], f"{type(e).__name__}: {e}"))
                print(f"[{model_key}] {p['id']}: ERROR {e}", file=sys.stderr, flush=True)
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            completed += 1
            if completed % 10 == 0 or completed == total:
                dt = time.monotonic() - t_start
                print(f"[{model_key}] {completed}/{total}  "
                      f"last: {rec['id']} out_tok={rec['output_tokens']} "
                      f"({dt:.0f}s elapsed)", flush=True)

    fout.close()

    if errors:
        err_path = out_dir / f"mtbench80_{model_key}.errors.json"
        err_path.write_text(json.dumps(errors, indent=2))
        print(f"[{model_key}] {len(errors)} errors → {err_path}", file=sys.stderr)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="mtbench80.jsonl")
    ap.add_argument("--output-dir", default="./reference-answers")
    ap.add_argument("--models", default="haiku,sonnet",
                    help="comma-separated subset of: " + ",".join(MODELS))
    ap.add_argument("--concurrency", type=int, default=4,
                    help="in-flight requests per model (default 4)")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[fatal] ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2

    prompts_file = Path(args.prompts)
    if not prompts_file.exists():
        prompts_file = Path(__file__).parent / args.prompts
    if not prompts_file.exists():
        print(f"[fatal] prompts file not found: {args.prompts}", file=sys.stderr)
        return 2
    prompts = load_prompts(prompts_file)
    print(f"loaded {len(prompts)} prompts from {prompts_file}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    for mk in model_keys:
        if mk not in MODELS:
            print(f"[fatal] unknown model '{mk}'. Known: {list(MODELS)}",
                  file=sys.stderr)
            return 2

    # Run models sequentially (one model's 80-prompt batch finishes before
    # the next starts). This keeps the concurrency cap honest — we use up
    # to `args.concurrency` in-flight requests total, not 2x that.
    outputs = []
    for mk in model_keys:
        path = run_for_model(mk, prompts, out_dir, args.concurrency)
        outputs.append(path)

    print(f"\nDone. Wrote: {', '.join(str(p) for p in outputs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
