#!/usr/bin/env python3
"""
Triage a fact battery against a model loaded in full bfloat16 (no quantization).

Differences from triage_hf_bnb8.py:
  - No BitsAndBytesConfig — full bf16 weights.
  - Explicitly separates tokenizer-level drops from behavioural drops so we can
    tell whether missing prompts were filtered by the vocab or by model failure.
  - Writes --out-battery: a filtered battery JSON (prompts that pass tokenizer
    check AND have total_swing >= --min-swing).

Hypothesis use-case:
  Run against gemma-2b.json (all 60 original prompts) to see whether the 3
  prompts absent from gemma-3-12b-it.json were dropped because of tokenizer
  incompatibility (multi-token targets) or because the quantized model got them
  wrong.  Repeat for google/gemma-4-31B once tokenizer-filtered battery is ready.

Memory requirements (bf16, no quant):
  Gemma 3 12B-IT  ~24 GB  — A100 40 GB or 80 GB
  Gemma 4 31B     ~62 GB  — A100 80 GB only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.fact_battery import load_fact_battery  # noqa: E402


MODEL_ID_DEFAULT = "google/gemma-3-12b-it"
BATTERY_DEFAULT = REPO_ROOT / "fact_battery" / "gemma-2b.json"
OUTDIR_DEFAULT = REPO_ROOT / "gemma-12b-it" / "triage_bf16"


# ── model loading ─────────────────────────────────────────────────────────────

def _load_model_and_tokenizer(
    model_id: str, *, use_fast: bool, token: Optional[str]
) -> Tuple[Any, Any]:
    tok_kwargs: Dict[str, Any] = {"use_fast": bool(use_fast)}
    if token:
        tok_kwargs["token"] = token
    tok = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if token:
        model_kwargs["token"] = token
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return model, tok


# ── per-prompt helpers ────────────────────────────────────────────────────────

def _try_single_token_id(tok: Any, token_str: str) -> Optional[int]:
    """Return the token id if token_str encodes to exactly one token, else None."""
    ids = tok.encode(token_str, add_special_tokens=False)
    if isinstance(ids, int):
        ids = [ids]
    return int(ids[0]) if len(ids) == 1 else None


def _final_logits(model: Any, tok: Any, prompt: str) -> torch.Tensor:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    return out.logits[0, -1, :]


def _ld_and_probs(
    logits_last: torch.Tensor, clean_id: int, corrupt_id: int
) -> Tuple[float, float, float]:
    lf = logits_last.float()
    ld = float((lf[clean_id] - lf[corrupt_id]).item())
    probs = torch.softmax(lf, dim=-1)
    return ld, float(probs[clean_id].item()), float(probs[corrupt_id].item())


def _is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


# ── main triage loop ──────────────────────────────────────────────────────────

def run_fact_battery(
    model: Any, tok: Any, battery: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for i, entry in enumerate(battery):
        category = entry.get("category", "")
        clean_target = entry["clean_target"]
        corrupt_target = entry["corrupt_target"]

        # Tokenizer check first — no forward pass needed
        clean_tid = _try_single_token_id(tok, clean_target)
        corrupt_tid = _try_single_token_id(tok, corrupt_target)

        if clean_tid is None or corrupt_tid is None:
            bad = []
            if clean_tid is None:
                bad.append(f"clean_target {clean_target!r} is multi-token")
            if corrupt_tid is None:
                bad.append(f"corrupt_target {corrupt_target!r} is multi-token")
            reason = "; ".join(bad)
            print(
                f"  [{i+1:3d}/{len(battery)}] SKIP (tokenizer) — {category}: "
                f"{entry['clean_prompt']!r}  ({reason})",
                flush=True,
            )
            rows.append({
                "idx": i,
                "category": category,
                "clean_prompt": entry["clean_prompt"],
                "corrupt_prompt": entry["corrupt_prompt"],
                "clean_target": clean_target,
                "corrupt_target": corrupt_target,
                "clean_target_id": clean_tid,
                "corrupt_target_id": corrupt_tid,
                "ld_clean": None,
                "ld_corrupt": None,
                "total_swing": None,
                "p_clean": None,
                "p_corrupt": None,
                "drop_reason": reason,
            })
            continue

        # Forward passes
        print(
            f"  [{i+1:3d}/{len(battery)}] {category}: {entry['clean_prompt']!r}",
            flush=True,
        )
        lf_clean = _final_logits(model, tok, entry["clean_prompt"])
        ld_clean, p_clean, _ = _ld_and_probs(lf_clean, clean_tid, corrupt_tid)

        lf_corrupt = _final_logits(model, tok, entry["corrupt_prompt"])
        ld_corrupt, _, p_corrupt = _ld_and_probs(lf_corrupt, clean_tid, corrupt_tid)

        total_swing = ld_clean - ld_corrupt
        values = [ld_clean, ld_corrupt, total_swing, p_clean, p_corrupt]
        if not all(_is_finite(v) for v in values):
            raise RuntimeError(
                f"Non-finite values at battery_idx={i} ({category}): "
                f"ld_clean={ld_clean}, ld_corrupt={ld_corrupt}"
            )

        drop_reason = "" if total_swing > 0 else "behavioural: total_swing <= 0"
        rows.append({
            "idx": i,
            "category": category,
            "clean_prompt": entry["clean_prompt"],
            "corrupt_prompt": entry["corrupt_prompt"],
            "clean_target": clean_target,
            "corrupt_target": corrupt_target,
            "clean_target_id": clean_tid,
            "corrupt_target_id": corrupt_tid,
            "ld_clean": float(ld_clean),
            "ld_corrupt": float(ld_corrupt),
            "total_swing": float(total_swing),
            "p_clean": float(p_clean),
            "p_corrupt": float(p_corrupt),
            "drop_reason": drop_reason,
        })

    return rows


# ── output writers ────────────────────────────────────────────────────────────

def write_triage_csv(ranked: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank", "battery_idx", "total_swing",
        "ld_clean", "ld_corrupt",
        "p_clean_target_on_clean", "p_corrupt_target_on_corrupt",
        "category", "clean_prompt", "corrupt_prompt",
        "clean_target", "corrupt_target",
        "clean_target_id", "corrupt_target_id",
        "drop_reason",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank, r in enumerate(ranked, start=1):
            def _fmt(v, fmt):
                return format(v, fmt) if v is not None else ""
            w.writerow({
                "rank": rank,
                "battery_idx": r["idx"],
                "total_swing": _fmt(r["total_swing"], ".6f"),
                "ld_clean": _fmt(r["ld_clean"], ".6f"),
                "ld_corrupt": _fmt(r["ld_corrupt"], ".6f"),
                "p_clean_target_on_clean": _fmt(r["p_clean"], ".8f"),
                "p_corrupt_target_on_corrupt": _fmt(r["p_corrupt"], ".8f"),
                "category": r["category"],
                "clean_prompt": r["clean_prompt"],
                "corrupt_prompt": r["corrupt_prompt"],
                "clean_target": r["clean_target"],
                "corrupt_target": r["corrupt_target"],
                "clean_target_id": r["clean_target_id"] if r["clean_target_id"] is not None else "",
                "corrupt_target_id": r["corrupt_target_id"] if r["corrupt_target_id"] is not None else "",
                "drop_reason": r["drop_reason"],
            })


def write_battery_json(
    rows: List[Dict[str, Any]],
    battery: List[Dict[str, str]],
    min_swing: float,
    out_json: Path,
) -> int:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    kept = [
        battery[r["idx"]]
        for r in rows
        if r["total_swing"] is not None and r["total_swing"] >= min_swing
    ]
    out_json.write_text(
        json.dumps(kept, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return len(kept)


def print_summary(rows: List[Dict[str, Any]], battery_path: Path, model_id: str) -> None:
    total = len(rows)
    tokenizer_drops = [r for r in rows if r["total_swing"] is None]
    behavioural_drops = [r for r in rows if r["total_swing"] is not None and r["total_swing"] <= 0]
    kept = [r for r in rows if r["total_swing"] is not None and r["total_swing"] > 0]

    print()
    print("=" * 65)
    print(f"TRIAGE SUMMARY — {model_id}")
    print(f"Battery: {battery_path}  ({total} prompts)")
    print("=" * 65)
    print(f"  Kept (total_swing > 0):      {len(kept):3d} / {total}")
    print(f"  Dropped — tokenizer:         {len(tokenizer_drops):3d}  (multi-token target in this vocab)")
    print(f"  Dropped — behavioural:       {len(behavioural_drops):3d}  (model doesn't know fact in bf16)")
    print()

    if tokenizer_drops:
        print("Tokenizer drops:")
        for r in tokenizer_drops:
            print(f"  [{r['idx']:3d}] {r['category']}: {r['clean_prompt']!r}")
            print(f"        reason: {r['drop_reason']}")
    if behavioural_drops:
        print("Behavioural drops (model fails in bf16):")
        for r in sorted(behavioural_drops, key=lambda x: x["total_swing"]):
            print(f"  [{r['idx']:3d}] swing={r['total_swing']:+.4f}  "
                  f"{r['category']}: {r['clean_prompt']!r}")
    print("=" * 65)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Triage a fact battery in full bfloat16 (no quantization). "
            "Explicitly reports tokenizer-level vs behavioural drops."
        )
    )
    p.add_argument(
        "--model-id", default=MODEL_ID_DEFAULT,
        help=f"HuggingFace model id (default: {MODEL_ID_DEFAULT})",
    )
    p.add_argument(
        "--battery", type=Path, default=BATTERY_DEFAULT,
        help=f"Input battery JSON (default: gemma-2b.json — all 60 original prompts)",
    )
    p.add_argument(
        "--outdir", type=Path, default=OUTDIR_DEFAULT,
        help="Directory for fact_battery_triage_bf16.csv",
    )
    p.add_argument(
        "--out-battery", type=Path, default=None,
        help=(
            "If set, write a filtered battery JSON containing only prompts "
            "that pass the tokenizer check AND have total_swing >= --min-swing."
        ),
    )
    p.add_argument(
        "--min-swing", type=float, default=0.0,
        help=(
            "Minimum total_swing threshold for --out-battery. "
            "Default 0.0 keeps every prompt the model gets right at all."
        ),
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

    # Sort: kept prompts first (descending swing), then drops at the bottom
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
