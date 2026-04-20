#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _trunc(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must be a JSON object at top level")
    if "pairs" not in data or not isinstance(data["pairs"], list):
        raise TypeError(f"{path} must contain a top-level 'pairs' array")
    return data


def _iter_rows(pairs: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for i, p in enumerate(pairs):
        if not isinstance(p, dict):
            raise TypeError(f"pairs[{i}] must be an object")
        yield p


def _fmt_prob(x: Any) -> str:
    try:
        return f"{float(x):.8f}"
    except Exception:
        return "NA"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit p_clean_target_on_clean and p_corrupt_target_on_corrupt in experiment JSON."
    )
    ap.add_argument(
        "experiment_json",
        type=Path,
        help="Path to experiment_A.json (or any experiment_{mode}.json).",
    )
    ap.add_argument(
        "--prompt-trunc",
        type=int,
        default=40,
        help="Truncate clean_prompt to this many characters (default 40).",
    )
    args = ap.parse_args()

    data = _load(args.experiment_json)
    pairs: List[Dict[str, Any]] = data["pairs"]

    rows: List[Tuple[float, Dict[str, Any]]] = []
    low_lt_05: List[Dict[str, Any]] = []
    low_lt_01: List[Dict[str, Any]] = []
    clean_high_08: List[bool] = []
    low_categories_05: Counter[str] = Counter()
    low_categories_01: Counter[str] = Counter()

    for p in _iter_rows(pairs):
        rank = p.get("rank")
        category = str(p.get("category", ""))
        clean_prompt = str(p.get("clean_prompt", ""))
        pc = p.get("p_clean_target_on_clean")
        px = p.get("p_corrupt_target_on_corrupt")

        try:
            px_f = float(px)
        except Exception:
            px_f = float("nan")

        flag = False
        try:
            flag = float(px) < 0.5
        except Exception:
            flag = True

        try:
            clean_high_08.append(float(pc) > 0.8)
        except Exception:
            clean_high_08.append(False)

        if flag:
            low_lt_05.append(p)
            low_categories_05[category] += 1
        try:
            if float(px) < 0.1:
                low_lt_01.append(p)
                low_categories_01[category] += 1
        except Exception:
            pass

        rows.append(
            (
                px_f,
                {
                    "rank": rank,
                    "category": category,
                    "clean_prompt_trunc": _trunc(clean_prompt, args.prompt_trunc),
                    "p_clean_target_on_clean": pc,
                    "p_corrupt_target_on_corrupt": px,
                    "flag_lt_0_5": flag,
                },
            )
        )

    rows.sort(key=lambda t: (t[0] != t[0], t[0]))  # NaNs last

    # Print table
    headers = [
        "rank",
        "category",
        "clean_prompt",
        "p_clean_target_on_clean",
        "p_corrupt_target_on_corrupt",
        "flag(px<0.5)",
    ]
    print(" | ".join(headers))
    print("-" * 120)
    for _px, r in rows:
        print(
            f"{str(r['rank']):>4} | "
            f"{r['category']:<22} | "
            f"{r['clean_prompt_trunc']:<40} | "
            f"{_fmt_prob(r['p_clean_target_on_clean'])} | "
            f"{_fmt_prob(r['p_corrupt_target_on_corrupt'])} | "
            f"{str(bool(r['flag_lt_0_5']))}"
        )

    n = len(pairs)
    n_lt_05 = len(low_lt_05)
    n_lt_01 = len(low_lt_01)
    n_clean_hi = sum(1 for b in clean_high_08 if b)

    def _cats(counter: Counter[str]) -> str:
        if not counter:
            return "(none)"
        items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        return ", ".join(f"{k}({v})" for k, v in items)

    print("\nSummary")
    print("-" * 120)
    print(f"pairs total: {n}")
    print(f"p_corrupt_target_on_corrupt < 0.5: {n_lt_05}")
    print(f"p_corrupt_target_on_corrupt < 0.1: {n_lt_01}")
    print(f"categories among <0.5: {_cats(low_categories_05)}")
    print(f"categories among <0.1: {_cats(low_categories_01)}")
    print(f"p_clean_target_on_clean > 0.8 count: {n_clean_hi}/{n}")
    print(f"p_clean_target_on_clean > 0.8 across all pairs: {n_clean_hi == n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

