#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path("exp3")
OUTPUT_DIR = Path("outputs/exp3_drop_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["A", "B", "C"]
N_LAYERS = 18


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


@dataclass(frozen=True)
class PairRow:
    rank: int
    category: str
    entity: str
    total_swing: float
    delta: np.ndarray  # shape (18,)
    drop_layer: Optional[int]
    max_damage_layer: int
    release_layer: Optional[int]
    early_mean: float
    late_mean: float
    ratio: float


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "pairs" not in data:
        raise TypeError(f"{path} must contain top-level JSON object with 'pairs'")
    return data


def _first_layer_where(pred: List[bool]) -> Optional[int]:
    for i, ok in enumerate(pred):
        if ok:
            return i
    return None


def _compute_metrics(p: Dict[str, Any]) -> PairRow:
    rank = int(p.get("rank"))
    category = str(p.get("category", ""))
    entity = str(p.get("entity_token", ""))
    total_swing = float(p.get("total_swing", float("nan")))

    delta_list = p.get("ld_delta_vs_clean_baseline_by_layer")
    if not isinstance(delta_list, list) or len(delta_list) != N_LAYERS:
        raise ValueError(f"rank={rank} expected delta length {N_LAYERS}, got {type(delta_list)} len={getattr(delta_list, '__len__', lambda: 'NA')()}")
    delta = np.array([float(x) for x in delta_list], dtype=float)

    # Drop layer: first layer where abs(delta[L]) < 0.2 * abs(delta[0])
    d0 = abs(float(delta[0]))
    if d0 <= 0:
        drop_layer = None
    else:
        thr = 0.2 * d0
        drop_layer = _first_layer_where([abs(float(d)) < thr for d in delta])

    # Most negative layer (max damage): argmin
    max_damage_layer = int(np.argmin(delta))

    # Release layer: first layer where abs(delta) < 2.0
    release_layer = _first_layer_where([abs(float(d)) < 2.0 for d in delta])

    early_mean = float(np.mean(np.abs(delta[0:8])))
    late_mean = float(np.mean(np.abs(delta[15:18])))
    ratio = float("inf") if late_mean == 0 else float(early_mean / late_mean)

    return PairRow(
        rank=rank,
        category=category,
        entity=entity,
        total_swing=total_swing,
        delta=delta,
        drop_layer=drop_layer,
        max_damage_layer=max_damage_layer,
        release_layer=release_layer,
        early_mean=early_mean,
        late_mean=late_mean,
        ratio=ratio,
    )


def _fmt_ratio(x: float) -> str:
    if not math.isfinite(x):
        return "inf"
    if x >= 100:
        return f"{x:.0f}x"
    if x >= 10:
        return f"{x:.1f}x"
    return f"{x:.2f}x"


def _print_mode_table(mode: str, rows: List[PairRow]) -> None:
    print(f"\nMode {mode} (n={len(rows)}):")
    print("rank | category               | entity            | max_damage_L | release_L | early_mean | late_mean | ratio")
    print("-" * 110)
    rows_sorted = sorted(rows, key=lambda r: r.rank)
    for r in rows_sorted:
        rel = "NA" if r.release_layer is None else str(r.release_layer)
        print(
            f"{r.rank:>4} | "
            f"{r.category:<22} | "
            f"{r.entity:<16} | "
            f"{r.max_damage_layer:>12} | "
            f"{rel:>9} | "
            f"{r.early_mean:>9.3f} | "
            f"{r.late_mean:>8.3f} | "
            f"{_fmt_ratio(r.ratio):>6}"
        )


def _hist_counts(xs: List[Optional[int]]) -> Tuple[List[int], int]:
    counts = [0 for _ in range(N_LAYERS)]
    missing = 0
    for x in xs:
        if x is None:
            missing += 1
        else:
            if 0 <= x < N_LAYERS:
                counts[x] += 1
            else:
                missing += 1
    return counts, missing


def _mean_median_int(xs: List[Optional[int]]) -> Tuple[float, float, int]:
    vals = [x for x in xs if x is not None]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(statistics.mean(vals)), float(statistics.median(vals)), len(vals)


def _percent_between(xs: List[Optional[int]], lo: int, hi: int) -> float:
    vals = [x for x in xs if x is not None]
    if not vals:
        return float("nan")
    hit = sum(1 for x in vals if lo <= x <= hi)
    return 100.0 * hit / len(vals)


def _figure1_release_hist(all_rows: Dict[str, List[PairRow]]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle(
        "Layer where entity signal releases (first layer abs delta < 2.0).",
        fontsize=13,
        fontweight="500",
        y=1.03,
    )

    for ax, mode in zip(axes, MODES):
        rel = [r.release_layer for r in all_rows[mode]]
        counts, missing = _hist_counts(rel)
        xs = np.arange(N_LAYERS)
        ax.bar(xs, counts, color="#4C78A8", alpha=0.9, edgecolor="white", linewidth=0.6)
        ax.set_title(f"Mode {mode} (n={len(all_rows[mode])})", fontweight="500")
        ax.set_xlabel("Layer")
        ax.set_xticks([0, 3, 6, 9, 12, 15, 17])
        if mode == "A":
            ax.set_ylabel("Count")
        if missing:
            ax.text(
                0.98,
                0.95,
                f"missing={missing}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="#555",
            )

    plt.tight_layout()
    out = OUTPUT_DIR / "fig1_release_layer_hist.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _figure2_scatter(all_rows: Dict[str, List[PairRow]]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    fig.suptitle("Max damage layer vs release layer (colored by category)", fontsize=13, fontweight="500", y=1.03)

    for ax, mode in zip(axes, MODES):
        rows = all_rows[mode]
        cats = sorted({r.category for r in rows})
        cmap = plt.get_cmap("tab20")
        color_map = {c: cmap(i % 20) for i, c in enumerate(cats)}

        for c in cats:
            xs = [r.max_damage_layer for r in rows if r.category == c and r.release_layer is not None]
            ys = [r.release_layer for r in rows if r.category == c and r.release_layer is not None]
            if not xs:
                continue
            ax.scatter(xs, ys, s=38, alpha=0.85, color=color_map[c], label=c, edgecolor="white", linewidth=0.4)

        ax.set_title(f"Mode {mode} (n={len(rows)})", fontweight="500")
        ax.set_xlabel("max_damage_layer")
        if mode == "A":
            ax.set_ylabel("release_layer")
        ax.set_xlim(-0.5, N_LAYERS - 0.5)
        ax.set_ylim(-0.5, N_LAYERS - 0.5)
        ax.set_xticks([0, 3, 6, 9, 12, 15, 17])
        ax.set_yticks([0, 3, 6, 9, 12, 15, 17])
        ax.plot([0, N_LAYERS - 1], [0, N_LAYERS - 1], linestyle="--", linewidth=1.0, color="#333", alpha=0.5)

        if len(cats) <= 14:
            ax.legend(fontsize=7, loc="upper left", frameon=False)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig2_scatter_max_damage_vs_release.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _figure3_top5_modeA(all_rows: Dict[str, List[PairRow]], raw_pairs_A: List[Dict[str, Any]]) -> Path:
    # Top 5 by TotalSwing from the raw JSON (ties resolved by rank).
    pairs_sorted = sorted(
        raw_pairs_A,
        key=lambda p: (-float(p.get("total_swing", float("-inf"))), int(p.get("rank", 10**9))),
    )
    top5_raw = pairs_sorted[:5]
    top5 = [_compute_metrics(p) for p in top5_raw]

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5))
    ax.set_title("Mode A: top 5 pairs by TotalSwing (delta curves + release layer)", fontweight="500")
    layers = np.arange(N_LAYERS)

    colors = plt.get_cmap("tab10")
    for i, r in enumerate(top5):
        label = f"#{r.rank} {r.category}/{r.entity} (swing={r.total_swing:.2f})"
        ax.plot(layers, r.delta, linewidth=2.0, color=colors(i), label=label)
        if r.release_layer is not None:
            ax.axvline(r.release_layer, color=colors(i), linestyle=":", linewidth=1.5, alpha=0.85)

    ax.axhline(0.0, color="#333", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("ld_delta_vs_clean_baseline")
    ax.set_xticks([0, 3, 6, 9, 12, 15, 17])
    ax.legend(fontsize=8, loc="lower right", frameon=False)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig3_modeA_top5_delta_curves.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    data: Dict[str, Dict[str, Any]] = {}
    rows_by_mode: Dict[str, List[PairRow]] = {}
    raw_pairs_by_mode: Dict[str, List[Dict[str, Any]]] = {}

    for mode in MODES:
        path = BASE_DIR / mode / f"experiment3_{mode}.json"
        d = _load_json(path)
        pairs = d.get("pairs", [])
        if not isinstance(pairs, list):
            raise TypeError(f"{path} 'pairs' must be a list")
        data[mode] = d
        raw_pairs_by_mode[mode] = pairs
        rows_by_mode[mode] = [_compute_metrics(p) for p in pairs]

    # Per-mode tables (rank order)
    for mode in MODES:
        _print_mode_table(mode, rows_by_mode[mode])

    # Aggregate stats per mode
    print("\n" + "=" * 80)
    print("Aggregate statistics per mode")
    print("=" * 80)
    for mode in MODES:
        rows = rows_by_mode[mode]
        release_layers = [r.release_layer for r in rows]
        max_damage_layers = [r.max_damage_layer for r in rows]

        mean_rel, med_rel, n_rel = _mean_median_int(release_layers)
        mean_max, med_max, _n_max = _mean_median_int(max_damage_layers)  # always present
        counts, missing = _hist_counts(release_layers)
        pct_13_15 = _percent_between(release_layers, 13, 15)

        print(f"\nMode {mode} (n={len(rows)}):")
        print(f"  release_layer: mean={mean_rel:.2f}  median={med_rel:.2f}  valid={n_rel}/{len(rows)}  missing={missing}")
        print(f"  max_damage_layer: mean={mean_max:.2f}  median={med_max:.2f}")
        print("  release_layer histogram (layer:count):")
        hist_str = "  " + " ".join(f"{i}:{c}" for i, c in enumerate(counts))
        print(hist_str)
        print(f"  % pairs with release_layer in [13,15]: {pct_13_15:.1f}%")

    # Figures
    fig1 = _figure1_release_hist(rows_by_mode)
    fig2 = _figure2_scatter(rows_by_mode)
    fig3 = _figure3_top5_modeA(rows_by_mode, raw_pairs_by_mode["A"])

    print("\nSaved figures:")
    print(f"  - {fig1}")
    print(f"  - {fig2}")
    print(f"  - {fig3}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

