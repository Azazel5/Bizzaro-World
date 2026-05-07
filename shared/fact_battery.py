from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_fact_battery(path: Path) -> List[Dict[str, str]]:
    """
    Load aligned prompt pairs from JSON.

    Each entry is coerced to `Dict[str, str]` so downstream code can be simple.
    """
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON array of objects")
    out: List[Dict[str, str]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"{path}[{i}] must be an object")
        out.append({str(k): str(v) for k, v in item.items()})
    return out


def fact_battery_dir(repo_root: Path) -> Path:
    return repo_root / "fact_battery"


def model_fact_battery_path(repo_root: Path, model_slug: str) -> Path:
    """
    Path to a model-specific fact battery file under `fact_battery/`.

    Example slugs:
    - "gemma-2b"
    - "llama3-70b"
    """
    return fact_battery_dir(repo_root) / f"{model_slug}.json"

