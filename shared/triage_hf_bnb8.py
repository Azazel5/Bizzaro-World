#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
BATTERY_DEFAULT = REPO_ROOT / "fact_battery" / "gemma-3-12b-it.json"
OUTDIR_DEFAULT = REPO_ROOT / "gemma-12b-it" / "triage"


def _load_model_and_tokenizer(model_id: str, *, use_fast: bool):
    quant = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tok


def _encode_single_token_id(tok: Any, token_str: str) -> int:
    ids = tok.encode(token_str, add_special_tokens=False)
    if isinstance(ids, int):
        ids = [ids]
    if len(ids) != 1:
        raise ValueError(f"{token_str!r} encodes to {ids} (len={len(ids)}), expected 1 token id.")
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


def run_fact_battery(model: Any, tok: Any, battery: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, entry in enumerate(battery):
        clean_prompt = entry["clean_prompt"]
        corrupt_prompt = entry["corrupt_prompt"]

        clean_tid = _encode_single_token_id(tok, entry["clean_target"])
        corrupt_tid = _encode_single_token_id(tok, entry["corrupt_target"])

        lf_clean = _final_logits(model, tok, clean_prompt)
        ld_clean, p_clean_on_clean, _p_corrupt_on_clean = _ld_and_probs(lf_clean, clean_tid, corrupt_tid)

        lf_corrupt = _final_logits(model, tok, corrupt_prompt)
        ld_corrupt, _p_clean_on_corrupt, p_corrupt_on_corrupt = _ld_and_probs(
            lf_corrupt, clean_tid, corrupt_tid
        )

        total_swing = ld_clean - ld_corrupt
        rows.append(
            {
                "idx": i,
                "category": entry.get("category", ""),
                "clean_prompt": clean_prompt,
                "corrupt_prompt": corrupt_prompt,
                "clean_target": entry["clean_target"],
                "corrupt_target": entry["corrupt_target"],
                "clean_target_id": clean_tid,
                "corrupt_target_id": corrupt_tid,
                "ld_clean": float(ld_clean),
                "ld_corrupt": float(ld_corrupt),
                "total_swing": float(total_swing),
                "p_clean": float(p_clean_on_clean),
                "p_corrupt": float(p_corrupt_on_corrupt),
            }
        )
    return rows


def write_triage_csv(ranked: List[Dict[str, Any]], out_csv: Path) -> None:
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
        for rank, r in enumerate(ranked, start=1):
            w.writerow(
                {
                    "rank": rank,
                    "battery_idx": r["idx"],
                    "total_swing": f"{r['total_swing']:.6f}",
                    "ld_clean": f"{r['ld_clean']:.6f}",
                    "ld_corrupt": f"{r['ld_corrupt']:.6f}",
                    "p_clean_target_on_clean": f"{r['p_clean']:.8f}",
                    "p_corrupt_target_on_corrupt": f"{r['p_corrupt']:.8f}",
                    "category": r["category"],
                    "clean_prompt": r["clean_prompt"],
                    "corrupt_prompt": r["corrupt_prompt"],
                    "clean_target": r["clean_target"],
                    "corrupt_target": r["corrupt_target"],
                    "clean_target_id": r["clean_target_id"],
                    "corrupt_target_id": r["corrupt_target_id"],
                }
            )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 8-bit triage for any HF causal-LM model.")
    p.add_argument(
        "--outdir",
        type=Path,
        default=OUTDIR_DEFAULT,
        help=f"Directory for fact_battery_triage.csv (default: {OUTDIR_DEFAULT})",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID_DEFAULT,
        help=f"Hugging Face model id (default: {MODEL_ID_DEFAULT})",
    )
    p.add_argument(
        "--battery",
        type=Path,
        default=BATTERY_DEFAULT,
        help=f"Battery JSON path (default: {BATTERY_DEFAULT})",
    )
    p.add_argument(
        "--use-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fast tokenizer when available (default: true).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    battery = load_fact_battery(args.battery)

    print(f"Loading {args.model_id} in 8-bit (bitsandbytes)...", flush=True)
    model, tok = _load_model_and_tokenizer(args.model_id, use_fast=bool(args.use_fast))

    print(f"Running fact battery ({len(battery)} rows)...", flush=True)
    rows = run_fact_battery(model, tok, battery)
    ranked = sorted(rows, key=lambda r: r["total_swing"], reverse=True)

    out_csv = args.outdir / "fact_battery_triage.csv"
    write_triage_csv(ranked, out_csv)
    print(f"Wrote triage CSV: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
