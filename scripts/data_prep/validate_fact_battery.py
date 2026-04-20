#!/usr/bin/env python3
"""
Validate `fact_battery.json` for mechanistic-interpretability constraints.

Checks:
1) Prompt alignment: clean_prompt and corrupt_prompt tokenize to the same length.
2) Single-token targets: clean_target and corrupt_target each tokenize to exactly 1 id.

Usage (from repo root):
  python3 scripts/data_prep/validate_fact_battery.py

Notes:
- Tries local cache first (local_files_only=True) to avoid network / gated issues.
- If cache load fails, it falls back to normal loading (uses HF_TOKEN / HUGGINGFACE_HUB_TOKEN if set).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

MODEL_ID_DEFAULT = "google/gemma-2b"
BATTERY_PATH_DEFAULT = Path(__file__).resolve().parents[2] / "fact_battery.json"


def _load_tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception as e_cache:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        kwargs: Dict[str, Any] = {}
        if token:
            kwargs["token"] = token
        try:
            return AutoTokenizer.from_pretrained(model_id, **kwargs)
        except Exception as e_net:
            raise RuntimeError(
                "Tokenizer load failed.\n\n"
                f"- model_id: {model_id}\n"
                f"- local_files_only error: {e_cache}\n"
                f"- network/token error: {e_net}\n\n"
                "Fix options:\n"
                "1) Ensure the Gemma tokenizer is already present in your HF cache for this account.\n"
                "2) Authenticate on HPC: `huggingface-cli login` or export HF_TOKEN.\n"
            ) from e_net


def _enc(tok: Any, s: str) -> List[int]:
    return list(tok.encode(s, add_special_tokens=False))


def validate(battery: List[Dict[str, str]], tok: Any) -> List[Tuple[Any, ...]]:
    failures: List[Tuple[Any, ...]] = []
    for i, e in enumerate(battery):
        cp, xp = e["clean_prompt"], e["corrupt_prompt"]
        ct, xt = e["clean_target"], e["corrupt_target"]

        cplen, xplen = len(_enc(tok, cp)), len(_enc(tok, xp))
        if cplen != xplen:
            failures.append((i, "prompt_len", cplen, xplen, cp, xp))

        ct_ids, xt_ids = _enc(tok, ct), _enc(tok, xt)
        if len(ct_ids) != 1:
            failures.append((i, "clean_target_tokens", len(ct_ids), ct, ct_ids))
        if len(xt_ids) != 1:
            failures.append((i, "corrupt_target_tokens", len(xt_ids), xt, xt_ids))

    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    ap.add_argument("--battery", type=Path, default=BATTERY_PATH_DEFAULT)
    ap.add_argument("--max-print", type=int, default=100)
    args = ap.parse_args()

    battery_path: Path = args.battery
    if not battery_path.exists():
        print(f"Battery file not found: {battery_path}", file=sys.stderr)
        return 2

    battery = json.loads(battery_path.read_text(encoding="utf-8"))
    if not isinstance(battery, list):
        print(f"{battery_path} must contain a JSON array.", file=sys.stderr)
        return 2

    try:
        tok = _load_tokenizer(args.model_id)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 2

    failures = validate(battery, tok)
    print(f"pairs: {len(battery)}  failures: {len(failures)}")
    for f in failures[: args.max_print]:
        print(f)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

