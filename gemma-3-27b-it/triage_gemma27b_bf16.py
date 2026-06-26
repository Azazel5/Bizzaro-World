#!/usr/bin/env python3
"""
Behavioral triage for google/gemma-3-27b-it in full bfloat16 (no quantization).

Runs 2 forward passes per prompt (clean + corrupt), computes TotalSwing =
clean_ld - corrupt_ld for each entry, and outputs:
  - fact_battery_triage_bf16.csv  ranked by TotalSwing descending
  - gemma-3-27b-it.json           filtered battery (TotalSwing >= --min-swing)

Memory: ~54 GB bf16 weights — requires A100 80 GB.

Usage:
    HF_TOKEN=your_token python gemma-3-27b-it/triage_gemma27b_bf16.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.triage_bf16 import _load_model_and_tokenizer, run_fact_battery  # noqa: E402
from shared.triage_bf16 import write_triage_csv, write_battery_json, print_summary  # noqa: E402
from shared.fact_battery import load_fact_battery  # noqa: E402


MODEL_ID_DEFAULT = "google/gemma-3-27b-it"
BATTERY_DEFAULT  = REPO_ROOT / "fact_battery" / "gemma-3-27b-it.json"
OUTDIR_DEFAULT   = REPO_ROOT / "gemma-3-27b-it" / "triage_bf16"
OUT_BATTERY_DEFAULT = REPO_ROOT / "fact_battery" / "gemma-3-27b-it.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BF16 triage for Gemma 3 27B-IT — measures TotalSwing per prompt."
    )
    p.add_argument("--model-id",    default=MODEL_ID_DEFAULT)
    p.add_argument("--battery",     type=Path, default=BATTERY_DEFAULT)
    p.add_argument("--outdir",      type=Path, default=OUTDIR_DEFAULT)
    p.add_argument(
        "--out-battery", type=Path, default=OUT_BATTERY_DEFAULT,
        help="Overwrite battery with prompts that survive the triage (TotalSwing >= --min-swing).",
    )
    p.add_argument(
        "--min-swing", type=float, default=0.0,
        help="Minimum TotalSwing to keep in --out-battery (default: 0.0).",
    )
    p.add_argument("--use-fast", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    battery = load_fact_battery(args.battery)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    print(f"Loading {args.model_id} in bfloat16 (no quantization)...", flush=True)
    model, tok = _load_model_and_tokenizer(
        args.model_id, use_fast=bool(args.use_fast), token=token
    )

    print(f"Running triage on {len(battery)} prompts...", flush=True)
    rows = run_fact_battery(model, tok, battery)
    ranked = sorted(
        rows,
        key=lambda r: (r["total_swing"] is not None, r["total_swing"] or -999),
        reverse=True,
    )

    out_csv = args.outdir / "fact_battery_triage_bf16.csv"
    write_triage_csv(ranked, out_csv)
    print(f"\nWrote triage CSV: {out_csv}", flush=True)

    if args.out_battery is not None:
        n_kept = write_battery_json(rows, battery, args.min_swing, args.out_battery)
        print(
            f"Wrote filtered battery — {n_kept}/{len(battery)} kept "
            f"(min_swing={args.min_swing}): {args.out_battery}",
            flush=True,
        )

    print_summary(rows, args.battery, args.model_id)


if __name__ == "__main__":
    main()
