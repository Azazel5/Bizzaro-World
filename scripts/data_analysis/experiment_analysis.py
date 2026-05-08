#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

MODE_ORDER = ["A", "B", "C"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

COLORS = {"A": "#7F77DD", "B": "#1D9E75", "C": "#D85A30"}
ALPHA_INDIVIDUAL = 0.16


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "pairs" not in data or not isinstance(data["pairs"], list):
        raise TypeError(f"{path} must contain a top-level JSON object with a 'pairs' list")
    return data


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _ld_deltas(pair: Dict[str, Any]) -> np.ndarray:
    values = pair.get("ld_delta_vs_clean_baseline_by_layer")
    if not isinstance(values, list):
        raise ValueError("pair missing ld_delta_vs_clean_baseline_by_layer list")
    arr = np.array([_to_float(x) for x in values], dtype=float)
    return arr


def _min_damage(pair: Dict[str, Any]) -> float:
    return float(np.min(_ld_deltas(pair)))


def _worst_layer(pair: Dict[str, Any]) -> int:
    return int(np.argmin(_ld_deltas(pair)))


def _release_layer(pair: Dict[str, Any], threshold: float = 2.0) -> Optional[int]:
    delta = np.abs(_ld_deltas(pair))
    for idx, value in enumerate(delta):
        if value < threshold:
            return idx
    return None


def _top_pairs_by_total_swing(pairs: List[Dict[str, Any]], topk: int = 5) -> List[Dict[str, Any]]:
    sorted_pairs = sorted(
        pairs,
        key=lambda p: (-_to_float(p.get("total_swing", float("nan"))), int(p.get("rank", 10**9))),
    )
    return sorted_pairs[:topk]


def _mean_ld_curve(pairs: List[Dict[str, Any]]) -> np.ndarray:
    if not pairs:
        return np.array([])
    all_deltas = np.stack([_ld_deltas(p) for p in pairs])
    return np.nanmean(all_deltas, axis=0)


def _layer_count_for_pairs(pairs: List[Dict[str, Any]]) -> int:
    if not pairs:
        return 0
    return len(_ld_deltas(pairs[0]))


def _safe_linregress(x: np.ndarray, y: np.ndarray) -> Any:
    if len(x) < 2 or np.all(np.isnan(x)) or np.all(np.isnan(y)):
        return None
    return stats.linregress(x, y)


def _plot_exp1_mean_ld_delta_curves(pairs_by_mode: Dict[str, List[Dict[str, Any]]], modes: Sequence[str], output_path: Path, model_name: str) -> None:
    layers = np.arange(max(_layer_count_for_pairs(pairs) for pairs in pairs_by_mode.values()))
    fig, axes = plt.subplots(1, len(modes), figsize=(15, 5), sharey=True)
    fig.suptitle(f"{model_name}: Mean ld_delta curves by mode", fontsize=13, fontweight="500", y=1.01)

    for ax, mode in zip(axes, modes):
        pairs = pairs_by_mode.get(mode, [])
        if not pairs:
            ax.set_visible(False)
            continue
        all_deltas = np.stack([_ld_deltas(p) for p in pairs])
        mean_delta = np.nanmean(all_deltas, axis=0)
        for row in all_deltas:
            ax.plot(np.arange(len(row)), row, color=COLORS.get(mode, '#444444'), alpha=ALPHA_INDIVIDUAL, linewidth=0.8)
        ax.plot(np.arange(len(mean_delta)), mean_delta, color=COLORS.get(mode, '#444444'), linewidth=2.5, label="Mean", zorder=5)
        ax.axhline(-5, color="#888", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(len(mean_delta) * 0.75, color="#D85A30", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_title(f"Mode {mode}  (n={len(pairs)})", fontweight="500")
        ax.set_xlabel("Layer")
        ax.set_xticks(np.linspace(0, len(mean_delta) - 1, min(7, len(mean_delta))).astype(int).tolist())
        if mode == modes[0]:
            ax.set_ylabel("ld_delta vs clean baseline")
        onset_layers = [
            next((i for i, d in enumerate(_ld_deltas(p)) if d < -5), None)
            for p in pairs
        ]
        valid = [o for o in onset_layers if o is not None]
        if valid:
            mean_onset = np.mean(valid)
            ax.axvline(mean_onset, color=COLORS.get(mode, '#444444'), linewidth=1.2, linestyle="--", alpha=0.7,
                       label=f"Mean onset L{mean_onset:.1f}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_exp1_worst_layer_distribution(pairs_by_mode: Dict[str, List[Dict[str, Any]]], modes: Sequence[str], output_path: Path, model_name: str) -> None:
    fig, axes = plt.subplots(1, len(modes), figsize=(13, 4))
    fig.suptitle(f"{model_name}: Worst layer distribution", fontsize=13, fontweight="500")

    for ax, mode in zip(axes, modes):
        pairs = pairs_by_mode.get(mode, [])
        if not pairs:
            ax.set_visible(False)
            continue
        worst = [_worst_layer(p) for p in pairs]
        layers = sorted(set(worst))
        counts = [worst.count(l) for l in layers]
        bars = ax.bar(layers, counts, color=COLORS.get(mode, '#444444'), alpha=0.85, width=0.6, edgecolor="white")
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        str(count), ha="center", va="bottom", fontsize=10, fontweight="500")
        ax.set_title(f"Mode {mode}  (n={len(pairs)})", fontweight="500")
        ax.set_xlabel("Worst layer")
        ax.set_ylabel("Count")
        ax.set_xticks(layers)
        mean_w = statistics.mean(worst)
        ax.axvline(mean_w, color="#333", linewidth=1.5, linestyle="--", alpha=0.7, label=f"Mean={mean_w:.2f}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_exp1_correlations(pairs_by_mode: Dict[str, List[Dict[str, Any]]], output_path: Path, model_name: str, x_key: str, x_label: str, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{model_name}: {title}", fontsize=13, fontweight="500")

    for ax, mode in zip(axes, MODE_ORDER):
        pairs = pairs_by_mode.get(mode, [])
        if not pairs:
            ax.set_visible(False)
            continue
        x = np.array([_to_float(p.get(x_key)) for p in pairs])
        y = np.array([_min_damage(p) for p in pairs])
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() == 0:
            continue
        x = x[valid]
        y = y[valid]
        slope, intercept, r, pval, _ = stats.linregress(x, y)
        r2 = r ** 2
        ax.scatter(x, y, color=COLORS[mode], alpha=0.75, s=60, edgecolors='white', linewidth=0.5, zorder=4)
        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(x_line, slope * x_line + intercept, color=COLORS[mode], linewidth=1.8, alpha=0.8)
        ax.set_title(f"Mode {mode}  r={r:.3f}  p={pval:.4f}", fontweight="500")
        ax.set_xlabel(x_label)
        ax.set_ylabel("min ld_delta (max damage)")
        stats_text = f"r = {r:.3f}\nr² = {r2:.3f}\np = {pval:.4f}\nn = {len(x)}"
        ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#ddd'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_exp3_release_hist(rows_by_mode: Dict[str, List[Dict[str, Any]]], modes: Sequence[str], output_path: Path, model_name: str) -> None:
    max_layer_all = max((max((r for r in [_release_layer(p) for p in rows] if r is not None), default=0) for rows in rows_by_mode.values()), default=0)
    figwidth = 15 if max_layer_all <= 20 else 20
    fig, axes = plt.subplots(1, len(modes), figsize=(figwidth, 5), sharey=True)
    fig.suptitle(f"{model_name}: release-layer histogram (entity-position patching)", fontsize=13, fontweight="500", y=1.03)
    for ax, mode in zip(axes, modes):
        rows = rows_by_mode.get(mode, [])
        rel = [_release_layer(p) for p in rows]
        max_layer = max((r for r in rel if r is not None), default=0)
        counts = [rel.count(i) for i in range(max_layer + 1)]
        xs = list(range(len(counts)))
        ax.bar(xs, counts, color=COLORS.get(mode, '#444444'), alpha=0.9, edgecolor='white', linewidth=0.6)
        ax.set_title(f"Mode {mode} (n={len(rows)})", fontweight="500")
        ax.set_xlabel("Release layer")
        if mode == modes[0]:
            ax.set_ylabel("Count")
        
        # Sparse x-axis ticks for readability with many layers
        tick_step = max(1, len(counts) // 15)
        xticks = [i for i in range(0, len(counts), tick_step)]
        if len(counts) - 1 not in xticks:
            xticks.append(len(counts) - 1)
        ax.set_xticks(xticks)
        ax.tick_params(axis='x', rotation=45)
        
        missing = sum(1 for x in rel if x is None)
        if missing:
            ax.text(0.98, 0.95, f"missing={missing}", transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_exp3_max_damage_vs_release(rows_by_mode: Dict[str, List[Dict[str, Any]]], modes: Sequence[str], output_path: Path, model_name: str) -> None:
    fig, axes = plt.subplots(1, len(modes), figsize=(15, 4), sharex=True, sharey=True)
    fig.suptitle(f"{model_name}: max damage layer vs release layer", fontsize=13, fontweight="500", y=1.03)
    for ax, mode in zip(axes, MODE_ORDER):
        rows = rows_by_mode.get(mode, [])
        cats = sorted({p.get('category', '') for p in rows})
        color_map = {c: plt.get_cmap('tab20')(i % 20) for i, c in enumerate(cats)}
        for c in cats:
            xs = [ _worst_layer(p) for p in rows if p.get('category', '') == c and _release_layer(p) is not None ]
            ys = [ _release_layer(p) for p in rows if p.get('category', '') == c and _release_layer(p) is not None ]
            if not xs:
                continue
            ax.scatter(xs, ys, s=38, alpha=0.85, color=color_map[c], label=c, edgecolor='white', linewidth=0.4)
        ax.set_title(f"Mode {mode} (n={len(rows)})", fontweight="500")
        ax.set_xlabel("max_damage_layer")
        if mode == 'A':
            ax.set_ylabel("release_layer")
        ax.set_xlim(-0.5,  max((_worst_layer(p) for p in rows), default=0) + 0.5)
        ax.set_ylim(-0.5, max((x for x in (_release_layer(p) for p in rows) if x is not None), default=0) + 0.5)
        ax.set_xticks([0, 3, 6, 9, 12, 15, 17])
        ax.set_yticks([0, 3, 6, 9, 12, 15, 17])
        ax.plot([0, ax.get_xlim()[1]], [0, ax.get_ylim()[1]], linestyle='--', linewidth=1.0, color='#333', alpha=0.5)
        if len(cats) <= 14:
            ax.legend(fontsize=7, loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_exp3_modeA_top5(rows: List[Dict[str, Any]], output_path: Path, model_name: str) -> None:
    top5 = _top_pairs_by_total_swing(rows, topk=5)
    if not top5:
        return
    n_layers = _layer_count_for_pairs(top5)
    layers = np.arange(n_layers)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_title(f"{model_name}: mode A top 5 TotalSwing delta curves", fontweight="500")
    colors = plt.get_cmap("tab10")
    for i, p in enumerate(top5):
        label = f"#{p.get('rank')} {p.get('category')}"
        delta = _ld_deltas(p)
        ax.plot(layers, delta, linewidth=2.0, color=colors(i), label=label)
        release_layer = _release_layer(p)
        if release_layer is not None:
            ax.axvline(release_layer, color=colors(i), linestyle=":", linewidth=1.5, alpha=0.85)
    ax.axhline(0.0, color="#333", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("ld_delta_vs_clean_baseline")
    ax.set_xticks(np.linspace(0, n_layers - 1, min(7, n_layers)).astype(int).tolist())
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_mode_jsons(base_dir: Path, subdir: str, pattern: str, modes: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for mode in modes:
        path = base_dir / subdir / mode / pattern.format(mode=mode)
        if not path.exists():
            raise FileNotFoundError(f"Missing expected JSON: {path}")
        result[mode] = _load_json(path)["pairs"]
    return result


def _print_exp1_stats(pairs_by_mode: Dict[str, List[Dict[str, Any]]], model_name: str) -> None:
    print(f"\n=== {model_name} exp1 summary ===")
    for mode, pairs in pairs_by_mode.items():
        worst = [_worst_layer(p) for p in pairs]
        x = np.array([_to_float(p.get("baseline_ld_clean")) for p in pairs])
        y = np.array([_min_damage(p) for p in pairs])
        valid = ~(np.isnan(x) | np.isnan(y))
        r, pval = (stats.pearsonr(x[valid], y[valid]) if valid.sum() > 1 else (float('nan'), float('nan')))
        onset = [next((i for i, d in enumerate(_ld_deltas(p)) if d < -5), None) for p in pairs]
        valid_onset = [o for o in onset if o is not None]
        print(f"Mode {mode} (n={len(pairs)}):")
        print(f"  worst layer: min={min(worst)} max={max(worst)} mean={statistics.mean(worst):.2f}")
        print(f"  all worst >= 15: {all(w >= 15 for w in worst)}")
        print(f"  all worst >= 13: {all(w >= 13 for w in worst)}")
        print(f"  mean onset layer (<-5): {statistics.mean(valid_onset):.2f}  (valid={len(valid_onset)}/{len(pairs)})")
        print(f"  Pearson r (confidence vs damage): {r:.3f}, p={pval:.4f}")


def _print_exp3_stats(rows_by_mode: Dict[str, List[Dict[str, Any]]], model_name: str) -> None:
    print(f"\n=== {model_name} exp3 summary ===")
    for mode, rows in rows_by_mode.items():
        release_layers = [_release_layer(p) for p in rows]
        max_layers = [_worst_layer(p) for p in rows]
        valid_release = [r for r in release_layers if r is not None]
        mean_rel = statistics.mean(valid_release) if valid_release else float('nan')
        pct_13_15 = 100.0 * sum(1 for r in valid_release if 13 <= r <= 15) / len(rows) if rows else float('nan')
        print(f"Mode {mode} (n={len(rows)}): mean release={mean_rel:.2f}, % [13-15]={pct_13_15:.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate experiment analysis figures for one model.")
    ap.add_argument("--model-name", required=True, help="Short label for this model (used in titles and output subdir)")
    ap.add_argument("--base-dir", required=True, type=Path, help="Root directory containing phase1 and phase3 output subdirs")
    ap.add_argument("--output-dir", type=Path, default=Path("scripts/data_analysis/outputs"), help="Base output dir for generated figures")
    ap.add_argument("--modes", nargs="*", default=MODE_ORDER, help="Experiment modes to process")
    args = ap.parse_args()

    phase1_base = args.base_dir / "phase1"
    phase3_base = args.base_dir / "phase3"
    out_dir = args.output_dir / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_exp1 = _load_mode_jsons(args.base_dir, "phase1", "experiment_{mode}.json", args.modes)
    rows_exp3 = _load_mode_jsons(args.base_dir, "phase3", "experiment3_{mode}.json", args.modes)

    _print_exp1_stats(pairs_exp1, args.model_name)
    _print_exp3_stats(rows_exp3, args.model_name)

    _plot_exp1_mean_ld_delta_curves(pairs_exp1, args.modes, out_dir / "exp1_mean_ld_delta_curves.png", args.model_name)
    _plot_exp1_worst_layer_distribution(pairs_exp1, args.modes, out_dir / "exp1_worst_layer_distribution.png", args.model_name)
    _plot_exp1_correlations(pairs_exp1, out_dir / "exp1_confidence_vs_damage.png", args.model_name,
                            x_key="baseline_ld_clean",
                            x_label="baseline_ld_clean (model confidence)",
                            title="Correlation: model confidence vs max damage")
    _plot_exp1_correlations(pairs_exp1, out_dir / "exp1_totalswing_vs_damage.png", args.model_name,
                            x_key="total_swing",
                            x_label="TotalSwing (LD_clean - LD_corrupt)",
                            title="Correlation: TotalSwing vs max damage")

    _plot_exp3_release_hist(rows_exp3, args.modes, out_dir / "exp3_release_layer_hist.png", args.model_name)
    _plot_exp3_max_damage_vs_release(rows_exp3, args.modes, out_dir / "exp3_max_damage_vs_release.png", args.model_name)
    if rows_exp3.get("A"):
        _plot_exp3_modeA_top5(rows_exp3["A"], out_dir / "exp3_modeA_top5_delta_curves.png", args.model_name)

    print(f"\nGenerated figures in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
