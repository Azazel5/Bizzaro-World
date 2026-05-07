#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer


MODEL_ID_DEFAULT = "meta-llama/Meta-Llama-3-70B"


def _enc(tok: Any, s: str) -> List[int]:
    return list(tok.encode(s, add_special_tokens=False))


def _is_single_token(tok: Any, token_str: str) -> bool:
    return len(_enc(tok, token_str)) == 1


def filter_battery_for_tokenizer(
    battery: List[Dict[str, str]], tok: Any
) -> Tuple[List[Dict[str, str]], List[Tuple[int, str]]]:
    """
    Keep only rows that satisfy:
    - clean/corrupt prompts tokenize to equal length (position-aligned patching)
    - clean_target and corrupt_target each tokenize to exactly 1 token id
    """
    kept: List[Dict[str, str]] = []
    dropped: List[Tuple[int, str]] = []

    for i, e in enumerate(battery):
        cp, xp = e["clean_prompt"], e["corrupt_prompt"]
        ct, xt = e["clean_target"], e["corrupt_target"]

        if len(_enc(tok, cp)) != len(_enc(tok, xp)):
            dropped.append((i, "prompt_len_mismatch"))
            continue
        if not _is_single_token(tok, ct):
            dropped.append((i, "clean_target_not_single_token"))
            continue
        if not _is_single_token(tok, xt):
            dropped.append((i, "corrupt_target_not_single_token"))
            continue

        kept.append(e)

    return kept, dropped


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Filter the Gemma-aligned fact battery for LLaMA-3 tokenizer compatibility."
    )
    ap.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    ap.add_argument(
        "--in-battery",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fact_battery" / "gemma-2b.json",
    )
    ap.add_argument(
        "--out-battery",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fact_battery" / "llama3-70b.json",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fact_battery" / "llama3-70b.drop_report.json",
    )
    args = ap.parse_args()

    battery = json.loads(args.in_battery.read_text(encoding="utf-8"))
    if not isinstance(battery, list):
        raise TypeError(f"{args.in_battery} must contain a JSON array")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tok_kwargs: Dict[str, Any] = {"use_fast": True}
    if token:
        tok_kwargs["token"] = token

    tok = AutoTokenizer.from_pretrained(args.model_id, **tok_kwargs)

    kept, dropped = filter_battery_for_tokenizer(battery, tok)
    args.out_battery.parent.mkdir(parents=True, exist_ok=True)
    args.out_battery.write_text(json.dumps(kept, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "input_battery": str(args.in_battery),
                "output_battery": str(args.out_battery),
                "n_in": len(battery),
                "n_kept": len(kept),
                "n_dropped": len(dropped),
                "dropped": [{"battery_idx": i, "reason": r} for (i, r) in dropped],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Kept {len(kept)}/{len(battery)} rows. Wrote {args.out_battery}")
    print(f"Drop report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

