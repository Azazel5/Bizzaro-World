#!/usr/bin/env python3
"""
Infer and add `entity_token` to each entry in fact_battery.json.

Heuristic: compare clean_prompt and corrupt_prompt word sequences and pick the first
clean-side "replace" token from a diff (strip punctuation). This matches the project
assumption that each pair differs by a single factual entity.

Usage:
  python3 scripts/data_prep/add_entity_tokens.py
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
BATTERY_PATH = REPO_ROOT / "fact_battery.json"

_PUNCT_RE = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")


def _words(s: str) -> List[str]:
    return s.split()


def infer_entity_token(clean_prompt: str, corrupt_prompt: str) -> str:
    cw = _words(clean_prompt)
    xw = _words(corrupt_prompt)
    sm = SequenceMatcher(a=cw, b=xw)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            chunk = cw[i1:i2]
            for tok in chunk:
                t = _PUNCT_RE.sub("", tok)
                if t:
                    return t
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            chunk = xw[j1:j2]
            for tok in chunk:
                t = _PUNCT_RE.sub("", tok)
                if t:
                    return t
    raise ValueError(
        f"Could not infer entity token from prompts:\n{clean_prompt}\n{corrupt_prompt}"
    )


def main() -> None:
    data = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("fact_battery.json must be a JSON array")

    changed = 0
    for i, e in enumerate(data):
        if not isinstance(e, dict):
            raise TypeError(f"Entry {i} must be an object")
        if "entity_token" in e and str(e["entity_token"]).strip():
            continue
        ent = infer_entity_token(str(e["clean_prompt"]), str(e["corrupt_prompt"]))
        e["entity_token"] = ent
        changed += 1

    BATTERY_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Updated {BATTERY_PATH} (added entity_token to {changed} entries).")


if __name__ == "__main__":
    main()

