#!/usr/bin/env python3
"""
Figure 1 -- "Lollipop": head-level path-patching effect vs. relative depth,
three models stacked (2B / 12B / 27B), shared y-axis.

Data: path_patching/results/<model>/path_patch_final_resid.json
    Schema: {"shape": [n_layers, n_heads], "values": [[float, ...], ...]}
    Values are ALREADY (patched_ld - clean_ld) / (clean_ld - corrupt_ld) --
    i.e. already expressed in units of that model's own clean-corrupt swing.
    Never renormalize these again. baseline_metrics.json is read only to
    quote total_swing in the caption/stdout, never to rescale a plotted
    value -- dividing by total_swing a second time would be a double
    normalization and is wrong.

Depth convention (must match make_fig2_decodability.py exactly):
    relative_depth = layer_index / (n_layers - 1)
    n_layers = true layer count: 2B=18, 12B=48, 27B=62 -> divide by 17/47/61.

No invented data: every stem plotted is a real (layer, head, value) triple
read directly from JSON. No smoothing, no interpolation, no synthetic points.

Usage:
    python make_fig1_lollipop.py
Output:
    fig1_lollipop.png (written next to this script)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent  # analysis/scripts/figures -> repo root
PP_RESULTS = REPO_ROOT / "path_patching" / "results"
OUT_DIR = SCRIPT_DIR  # figures live alongside the script that makes them

# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"        # load-bearing (negative score)
VERMILLION = "#D55E00"  # suppressive (positive score)
GREY = "#888888"
BAND_GREY = "#DDDDDD"

MODELS = [
    ("gemma_2b", "Gemma-2B (18L)", 18, 8),
    ("gemma_12b", "Gemma-3-12B-IT (48L)", 48, 16),
    ("gemma_27b", "Gemma-3-27B-IT (62L)", 62, 32),
]

# Required annotations: (model_key, layer, head, expected_value) -- verified
# against the JSON in the same run this script performs; if a value doesn't
# match, the script stops rather than silently plotting a wrong label.
ANNOTATIONS = {
    "gemma_2b": [(16, 2, -0.2305)],
    "gemma_12b": [(46, 5, 1.1094), (46, 4, -0.6133), (38, 8, -0.7461)],
    "gemma_27b": [(54, 23, -0.1768)],
}

Y_LIMITS = (-1.25, 1.25)
BAND_LO, BAND_HI = 0.80, 0.98


def load_model_data(model_key: str, n_layers: int, n_heads: int) -> dict:
    resid_path = PP_RESULTS / model_key / "path_patch_final_resid.json"
    d = json.loads(resid_path.read_text())
    shape = d["shape"]
    if shape != [n_layers, n_heads]:
        raise ValueError(
            f"{model_key}: shape mismatch -- JSON says {shape}, expected "
            f"[{n_layers}, {n_heads}]. Stopping rather than plotting "
            f"against a wrong assumption about the grid."
        )
    values = d["values"]

    baseline_path = PP_RESULTS / model_key / "baseline_metrics.json"
    baseline = json.loads(baseline_path.read_text())
    total_swing = baseline["total_swing"]

    return {"values": values, "total_swing": total_swing}


def verify_annotations(model_key: str, values: list[list[float]]) -> None:
    for layer, head, expected in ANNOTATIONS.get(model_key, []):
        actual = values[layer][head]
        if abs(actual - expected) > 0.001:
            raise ValueError(
                f"{model_key} L{layer}H{head}: JSON value {actual:.6f} does not "
                f"match spec's expected {expected} (tolerance 0.001). Stopping -- "
                f"per the non-negotiable rules, a mismatch must be reported, "
                f"not silently plotted."
            )
        print(f"  [verify] {model_key} L{layer}H{head}: {actual:+.4f} == expected {expected:+} OK")


def main() -> None:
    print("=" * 70)
    print("Figure 1 data verification (every value checked before plotting)")
    print("=" * 70)

    model_data = {}
    for model_key, title, n_layers, n_heads in MODELS:
        d = load_model_data(model_key, n_layers, n_heads)
        print(f"\n{model_key}: shape=[{n_layers},{n_heads}] total_swing={d['total_swing']}")
        verify_annotations(model_key, d["values"])
        model_data[model_key] = d

    fig, axes = plt.subplots(
        3, 1, figsize=(3.4, 4.2), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.25}, constrained_layout=True,
    )
    plt.rcParams.update({"font.size": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})

    print("\n" + "=" * 70)
    print("Headline plotted numbers (for cross-checking against paper text)")
    print("=" * 70)

    for ax_idx, (model_key, title, n_layers, n_heads) in enumerate(MODELS):
        ax = axes[ax_idx]
        values = model_data[model_key]["values"]
        denom = n_layers - 1  # 17 / 47 / 61

        xs, ys, colors = [], [], []
        for layer in range(n_layers):
            for head in range(n_heads):
                v = values[layer][head]
                if v == 0.0:
                    continue
                xs.append(layer / denom)
                ys.append(v)
                colors.append(BLUE if v < 0 else VERMILLION)

        print(f"\n{model_key}: {len(xs)} non-zero heads plotted "
              f"(x = layer / {denom})")
        vmax_idx = max(range(len(ys)), key=lambda i: abs(ys[i]))
        print(f"  largest |score|: {ys[vmax_idx]:+.4f} at relative_depth={xs[vmax_idx]:.4f}")

        # shaded dominant band, drawn first (behind stems)
        ax.axvspan(BAND_LO, BAND_HI, color=BAND_GREY, zorder=0)

        # stems
        for x, y, c in zip(xs, ys, colors):
            ax.plot([x, x], [0, y], color=c, linewidth=0.6, zorder=2)
            ax.plot(x, y, "o", color=c, markersize=2.5, zorder=3)

        # reference lines
        ax.axhline(0, color=GREY, linewidth=0.5, zorder=1)
        ax.axhline(1, color=GREY, linewidth=0.5, linestyle="--", zorder=1)
        ax.axhline(-1, color=GREY, linewidth=0.5, linestyle="--", zorder=1)

        ax.text(0.03, 0.90, title, transform=ax.transAxes, fontsize=8,
                ha="left", va="top", fontweight="bold")

        if ax_idx == 0:
            # Legend explaining the colour coding -- this is what a reader
            # actually needs to parse the figure; the dashed ±1 lines and
            # grey band are self-evident enough to leave unlabeled in-plot
            # and explained in the caption instead.
            legend_handles = [
                plt.Line2D([0], [0], marker="o", color=BLUE, linestyle="-",
                           markersize=3, linewidth=0.8,
                           label="load-bearing (patching reduces margin)"),
                plt.Line2D([0], [0], marker="o", color=VERMILLION, linestyle="-",
                           markersize=3, linewidth=0.8,
                           label="suppressive (patching increases margin)"),
            ]
            # Anchored in axes-fraction, not the "lower left" default --
            # that default hugs the bottom corner and collides with the
            # y=-1 reference line. This sits in the empty middle band
            # instead (2B has essentially no data between y=-0.6 and 0.6).
            ax.legend(handles=legend_handles, loc="center left",
                      bbox_to_anchor=(0.01, 0.32), fontsize=5.5,
                      frameon=False, handletextpad=0.4, borderaxespad=0)

        # annotate specific heads -- offset kept small (7pt) and clipped to
        # the panel so a near-ylim value (e.g. 12B L46H5 at +1.11) can't
        # bleed its label into the neighboring panel. Points near the right
        # edge (x > 0.9) get right-aligned, leftward-growing labels instead
        # of centered ones so the text itself doesn't run past x=1.0.
        for layer, head, expected in ANNOTATIONS.get(model_key, []):
            v = values[layer][head]
            x = layer / denom
            label = f"L{layer}H{head}"
            # Points close enough to the y-limit that an outward label
            # would be clipped (e.g. 12B L46H5 at +1.11, ylim tops at 1.25)
            # get their label placed on the inward side instead.
            near_top = v > Y_LIMITS[1] - 0.2
            near_bottom = v < Y_LIMITS[0] + 0.2
            label_above = (v >= 0 and not near_top) or near_bottom
            offset_y = 7 if label_above else -7
            va = "bottom" if label_above else "top"
            ha = "right" if x > 0.9 else "center"
            offset_x = -2 if x > 0.9 else 0
            ann = ax.annotate(
                label, xy=(x, v), xytext=(offset_x, offset_y), textcoords="offset points",
                fontsize=6, ha=ha, va=va, clip_on=True, annotation_clip=True,
                arrowprops=dict(arrowstyle="-", color="#333333", linewidth=0.4,
                                 clip_on=True),
            )
            print(f"  annotated {model_key} {label}: x={x:.4f} y={v:+.4f}")

        ax.set_ylim(*Y_LIMITS)
        ax.set_xlim(0, 1)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("relative depth (layer / final layer)", fontsize=8)
    fig.supylabel("path-patching score (multiples of clean-corrupt gap)", fontsize=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "fig1_lollipop.png"
    fig.savefig(png_path, dpi=300)
    print(f"\n[save] wrote {png_path}")


if __name__ == "__main__":
    main()
