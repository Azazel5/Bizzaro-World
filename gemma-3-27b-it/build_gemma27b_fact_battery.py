#!/usr/bin/env python3
"""
Phase 1 — Tokenizer filter for google/gemma-3-27b-it.

Downloads only the tokenizer (no GPU needed) and filters the master
battery for single-token targets + equal-length prompt pairs.

Usage:
    HF_TOKEN=your_token python gemma-3-27b-it/build_gemma27b_fact_battery.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.build_llama3_fact_battery import filter_battery_for_tokenizer  # noqa: E402


MODEL_ID_DEFAULT = "google/gemma-3-27b-it"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Filter the master battery (gemma-2b.json) for Gemma 3 27B-IT "
            "tokenizer compatibility — single-token targets + prompt-length alignment."
        )
    )
    ap.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    ap.add_argument(
        "--in-battery",
        type=Path,
        default=REPO_ROOT / "fact_battery" / "gemma-2b.json",
    )
    ap.add_argument(
        "--out-battery",
        type=Path,
        default=REPO_ROOT / "fact_battery" / "gemma-3-27b-it.json",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "fact_battery" / "gemma-3-27b-it.drop_report.json",
    )
    ap.add_argument("--use-fast", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    battery = json.loads(args.in_battery.read_text(encoding="utf-8"))
    if not isinstance(battery, list):
        raise TypeError(f"{args.in_battery} must contain a JSON array")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tok_kwargs: Dict[str, Any] = {"use_fast": bool(args.use_fast)}
    if token:
        tok_kwargs["token"] = token

    print(f"Loading tokenizer for {args.model_id}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id, **tok_kwargs)

    kept, dropped = filter_battery_for_tokenizer(battery, tok)

    args.out_battery.parent.mkdir(parents=True, exist_ok=True)
    args.out_battery.write_text(
        json.dumps(kept, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "input_battery": str(args.in_battery),
                "output_battery": str(args.out_battery),
                "n_in": len(battery),
                "n_kept": len(kept),
                "n_dropped": len(dropped),
                "dropped": [{"battery_idx": i, "reason": r} for i, r in dropped],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Kept {len(kept)}/{len(battery)} rows.  Wrote {args.out_battery}")
    print(f"Drop report: {args.report}")
    if dropped:
        import json as _json
        b2 = _json.loads(args.in_battery.read_text())
        print(f"\nDropped prompts ({len(dropped)}):")
        for i, reason in dropped:
            e = b2[i]
            print(f"  [{i:2d}] {reason}: {e['clean_prompt']!r}")
            print(f"        clean={e['clean_target']!r}  corrupt={e['corrupt_target']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
