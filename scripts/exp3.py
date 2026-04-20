#!/usr/bin/env python3
"""
Experiment 3 — (scaffold) standardized output handling.

This file exists to lock in the OUTDIR + Slurm submission pattern for all future
experiments in this repo.

Outputs (written to --outdir, default current working directory):
  - experiment3.json
  - experiment3.log

Mode selection follows `golden_pairs.select_golden_pairs` so experiments can be run
consistently across A/B/C.

NOTE: Experiment 3's actual interventions are intentionally not implemented yet.
This script writes metadata + selected pairs to make the OUTDIR plumbing testable.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from golden_pairs import GoldenPair, SelectionMode, select_golden_pairs  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("A", "B", "C"),
        default="A",
        help="Golden-pair selection from triage CSV (default: A).",
    )
    p.add_argument(
        "--triage-csv",
        type=Path,
        default=REPO_ROOT / "fact_battery_triage.csv",
        help="Path to fact_battery_triage.csv",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Directory to write JSON and log output (default: current directory).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    mode: SelectionMode = args.mode
    triage_path: Path = args.triage_csv
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if not triage_path.is_file():
        raise FileNotFoundError(
            f"Triage CSV not found: {triage_path}\n"
            "Copy fact_battery_triage.csv here or pass --triage-csv."
        )

    pairs: List[GoldenPair] = select_golden_pairs(triage_path, mode)
    if not pairs:
        raise RuntimeError("No pairs selected (empty triage?).")

    out_json = outdir / "experiment3.json"
    out_log = outdir / "experiment3.log"

    out_log.write_text(
        f"Experiment 3 scaffold (no interventions yet).\\n"
        f"mode={mode} n_pairs={len(pairs)}\\n",
        encoding="utf-8",
    )

    payload: Dict[str, Any] = {
        "experiment": "3",
        "description": "scaffold: standardized OUTDIR + mode selection plumbing",
        "selection_mode": mode,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_pairs": len(pairs),
        "pairs": [p.as_dict() for p in pairs],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {out_json.resolve()}", flush=True)
    print(f"Wrote {out_log.resolve()}", flush=True)


if __name__ == "__main__":
    main()

