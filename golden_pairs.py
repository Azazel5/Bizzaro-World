from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional


SelectionMode = Literal["A", "B", "C"]


@dataclass(frozen=True)
class GoldenPair:
    """
    A prompt-pair item selected from `fact_battery_triage.csv`.

    Fields match the triage export so the next phase (patching) can use either
    raw strings or token ids.
    """

    rank: int
    battery_idx: int
    category: str
    clean_prompt: str
    corrupt_prompt: str
    clean_target: str
    corrupt_target: str
    clean_target_id: int
    corrupt_target_id: int
    total_swing: float
    ld_clean: float
    ld_corrupt: float
    p_clean_target_on_clean: float
    p_corrupt_target_on_corrupt: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "rank": self.rank,
            "battery_idx": self.battery_idx,
            "category": self.category,
            "clean_prompt": self.clean_prompt,
            "corrupt_prompt": self.corrupt_prompt,
            "clean_target": self.clean_target,
            "corrupt_target": self.corrupt_target,
            "clean_target_id": self.clean_target_id,
            "corrupt_target_id": self.corrupt_target_id,
            "total_swing": self.total_swing,
            "ld_clean": self.ld_clean,
            "ld_corrupt": self.ld_corrupt,
            "p_clean_target_on_clean": self.p_clean_target_on_clean,
            "p_corrupt_target_on_corrupt": self.p_corrupt_target_on_corrupt,
        }


def _parse_int(x: str) -> int:
    return int(x.strip())


def _parse_float(x: str) -> float:
    return float(x.strip())


def read_triage_csv(path: Path) -> List[GoldenPair]:
    """
    Read `fact_battery_triage.csv` produced by `behavioral_friction_gemma2b.py`.
    """
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows: List[GoldenPair] = []
        for row in r:
            rows.append(
                GoldenPair(
                    rank=_parse_int(row["rank"]),
                    battery_idx=_parse_int(row["battery_idx"]),
                    total_swing=_parse_float(row["total_swing"]),
                    ld_clean=_parse_float(row["ld_clean"]),
                    ld_corrupt=_parse_float(row["ld_corrupt"]),
                    p_clean_target_on_clean=_parse_float(row["p_clean_target_on_clean"]),
                    p_corrupt_target_on_corrupt=_parse_float(
                        row["p_corrupt_target_on_corrupt"]
                    ),
                    category=row["category"],
                    clean_prompt=row["clean_prompt"],
                    corrupt_prompt=row["corrupt_prompt"],
                    clean_target=row["clean_target"],
                    corrupt_target=row["corrupt_target"],
                    clean_target_id=_parse_int(row["clean_target_id"]),
                    corrupt_target_id=_parse_int(row["corrupt_target_id"]),
                )
            )
    # CSV is already written in ranked order, but keep this robust.
    rows.sort(key=lambda gp: gp.rank)
    return rows


def select_golden_pairs(
    triage_csv_path: Path,
    mode: SelectionMode,
    *,
    top_n: int = 15,
    per_category: int = 1,
) -> List[GoldenPair]:
    """
    Select prompt pairs from the triage CSV.

    Modes:
    - A: top-N globally by TotalSwing (uses CSV rank order). Default N=15.
    - B: best-per-category (top `per_category` per category). Default per_category=1.
    - C: all pairs in ranked order (returns the full CSV).
    """
    ranked = read_triage_csv(triage_csv_path)

    if mode == "A":
        return ranked[: max(0, top_n)]

    if mode == "B":
        out: List[GoldenPair] = []
        counts: Dict[str, int] = {}
        for gp in ranked:
            c = counts.get(gp.category, 0)
            if c >= per_category:
                continue
            out.append(gp)
            counts[gp.category] = c + 1
        return out

    if mode == "C":
        return ranked

    raise ValueError(f"Unknown mode {mode!r}; expected one of 'A', 'B', 'C'.")


def select_prompt_dicts(
    triage_csv_path: Path,
    mode: SelectionMode,
    *,
    top_n: int = 15,
    per_category: int = 1,
) -> List[Dict[str, object]]:
    """
    Convenience wrapper that returns plain dicts (easy to serialize / feed into notebooks).
    """
    return [
        gp.as_dict()
        for gp in select_golden_pairs(
            triage_csv_path, mode, top_n=top_n, per_category=per_category
        )
    ]

