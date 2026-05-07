from __future__ import annotations

from pathlib import Path


def repo_root_from_here(here: str | Path, *, levels_up: int) -> Path:
    """
    Resolve a repo root by walking `levels_up` parents from a known file.

    This keeps "where is the repo root?" logic consistent across scripts that
    live in nested model-specific directories.
    """
    p = Path(here).resolve()
    for _ in range(levels_up):
        p = p.parent
    return p

