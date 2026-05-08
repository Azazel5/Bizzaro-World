#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.fact_battery import load_fact_battery  # noqa: E402


MODEL_ID_DEFAULT = "google/gemma-3-12b-it"
BATTERY_DEFAULT = REPO_ROOT / "fact_battery" / "gemma-2b.json"
OUT_CSV_DEFAULT = Path.home() / "smoke_test_gemma12B_triage.csv"


def _load_model_and_tokenizer(model_id: str, *, use_fast: bool):
    quant = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tok


def _encode_single_token_id(tok: Any, token_str: str) -> int:
    ids = tok.encode(token_str, add_special_tokens=False)
    if isinstance(ids, int):
        ids = [ids]
    if len(ids) != 1:
        raise ValueError(
            f"{token_str!r} encodes to {ids} (len={len(ids)}), expected exactly 1 token id."
        )
    return int(ids[0])


def _final_logits(model: Any, tok: Any, prompt: str) -> torch.Tensor:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    return out.logits[0, -1, :]


def _ld_and_probs(logits_last: torch.Tensor, clean_id: int, corrupt_id: int) -> Tuple[float, float, float]:
    lf = logits_last.float()
    ld = (lf[clean_id] - lf[corrupt_id]).item()
    probs = torch.softmax(lf, dim=-1)
    return float(ld), float(probs[clean_id].item()), float(probs[corrupt_id].item())


def _is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def run_one_pair(model: Any, tok: Any, entry: Dict[str, str], *, pair_idx: int) -> Dict[str, Any]:
    clean_prompt = entry["clean_prompt"]
    corrupt_prompt = entry["corrupt_prompt"]
    clean_tid = _encode_single_token_id(tok, entry["clean_target"])
    corrupt_tid = _encode_single_token_id(tok, entry["corrupt_target"])

    lf_clean = _final_logits(model, tok, clean_prompt)
    ld_clean, p_clean_on_clean, _ = _ld_and_probs(lf_clean, clean_tid, corrupt_tid)

    lf_corrupt = _final_logits(model, tok, corrupt_prompt)
    ld_corrupt, _, p_corrupt_on_corrupt = _ld_and_probs(lf_corrupt, clean_tid, corrupt_tid)

    total_swing = float(ld_clean - ld_corrupt)
    values = [ld_clean, ld_corrupt, total_swing, p_clean_on_clean, p_corrupt_on_corrupt]
    if not all(_is_finite(v) for v in values):
        raise RuntimeError(
            "Smoke test produced non-finite values. "
            f"ld_clean={ld_clean}, ld_corrupt={ld_corrupt}, total_swing={total_swing}, "
            f"p_clean={p_clean_on_clean}, p_corrupt={p_corrupt_on_corrupt}"
        )

    return {
        "rank": 1,
        "battery_idx": int(pair_idx),
        "total_swing": total_swing,
        "ld_clean": float(ld_clean),
        "ld_corrupt": float(ld_corrupt),
        "p_clean_target_on_clean": float(p_clean_on_clean),
        "p_corrupt_target_on_corrupt": float(p_corrupt_on_corrupt),
        "category": entry.get("category", ""),
        "clean_prompt": clean_prompt,
        "corrupt_prompt": corrupt_prompt,
        "clean_target": entry["clean_target"],
        "corrupt_target": entry["corrupt_target"],
        "clean_target_id": int(clean_tid),
        "corrupt_target_id": int(corrupt_tid),
    }


def write_csv(row: Dict[str, Any], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "battery_idx",
        "total_swing",
        "ld_clean",
        "ld_corrupt",
        "p_clean_target_on_clean",
        "p_corrupt_target_on_corrupt",
        "category",
        "clean_prompt",
        "corrupt_prompt",
        "clean_target",
        "corrupt_target",
        "clean_target_id",
        "corrupt_target_id",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(
            {
                **row,
                "total_swing": f"{row['total_swing']:.6f}",
                "ld_clean": f"{row['ld_clean']:.6f}",
                "ld_corrupt": f"{row['ld_corrupt']:.6f}",
                "p_clean_target_on_clean": f"{row['p_clean_target_on_clean']:.8f}",
                "p_corrupt_target_on_corrupt": f"{row['p_corrupt_target_on_corrupt']:.8f}",
            }
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Smoke-test Gemma-12B triage on exactly one prompt pair and write "
            "~/smoke_test_gemma12B_triage.csv by default."
        )
    )
    p.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    p.add_argument("--battery", type=Path, default=BATTERY_DEFAULT)
    p.add_argument(
        "--pair-idx",
        type=int,
        default=0,
        help="Index into battery JSON (default: 0).",
    )
    p.add_argument("--out-csv", type=Path, default=OUT_CSV_DEFAULT)
    p.add_argument(
        "--use-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fast tokenizer when available (default: true).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    battery: List[Dict[str, str]] = load_fact_battery(args.battery)
    if args.pair_idx < 0 or args.pair_idx >= len(battery):
        raise IndexError(
            f"--pair-idx {args.pair_idx} out of range for battery size {len(battery)}"
        )
    entry = battery[args.pair_idx]

    print(f"Loading {args.model_id} in 8-bit (bitsandbytes)...", flush=True)
    model, tok = _load_model_and_tokenizer(args.model_id, use_fast=bool(args.use_fast))

    print(
        f"Running smoke test on battery index {args.pair_idx} "
        f"({entry.get('category', 'unknown')})...",
        flush=True,
    )
    row = run_one_pair(model, tok, entry, pair_idx=args.pair_idx)
    write_csv(row, args.out_csv)
    print(f"Wrote smoke-test CSV: {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
