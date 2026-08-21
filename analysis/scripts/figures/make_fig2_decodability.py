#!/usr/bin/env python3
"""
Figure 2 -- "Decodability": median rank of the correct answer under the
Jacobian lens vs. relative depth, for 12B and 27B, with the path-patching
dominant band (from Figure 1) overlaid to show cross-method convergence.

Data: jlens_pipeline/results/jlens_ranks_gemma_<model>.json
    Each record's clean_rank_by_layer has length 47 (12B) / 61 (27B) --
    one shorter than the model's true layer count, since the lens does not
    cover the final layer. Index k in that list IS layer k.

Depth convention (must match make_fig1_lollipop.py exactly):
    relative_depth = k / (n_layers - 1), i.e. k/47 for 12B, k/61 for 27B --
    NOT k / len(clean_rank_by_layer)-1 (which would be k/46 or k/60). The
    array is shorter than the layer count; the divisor is the layer count,
    not the array length.

There is no 2B J-lens data -- only 12B and 27B are plotted, by design, not
by omission.

No invented data: medians are computed directly from the 57 real per-prompt
rank values at each layer. No smoothing, no interpolation across missing
layers.

Usage:
    python make_fig2_decodability.py
Output:
    fig2_decodability.png (written next to this script)
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
JLENS_RESULTS = REPO_ROOT / "jlens_pipeline" / "results"
OUT_DIR = SCRIPT_DIR  # figures live alongside the script that makes them

GREEN = "#009E73"   # 12B, Okabe-Ito bluish-green
ORANGE = "#E69F00"  # 27B, Okabe-Ito orange
BAND_GREY = "#DDDDDD"
GREY = "#888888"

BAND_LO, BAND_HI = 0.80, 0.98

MODELS = [
    ("12b", "12B", GREEN, 48, 38),
    ("27b", "27B", ORANGE, 62, 54),
]

# Expected values from the spec -- checked, not assumed. A mismatch prints
# a warning but does not stop the script, since (as found during
# verification) 27B has a genuine 3-way tie in its minimum median rank
# (layers 53/54/55 all = 5) and the spec's choice of L54 as "the peak" is a
# defensible tie-break (it coincides with the causal layer) rather than a
# wrong number -- see the printed tie note below.
EXPECTED = {
    "12b": {"peak_layer": 41, "peak_rank": 19, "causal_rank": 24, "last_rank": 24},
    "27b": {"peak_layer": 54, "peak_rank": 5, "causal_rank": 5, "last_rank": 26},
}


def load_model(model_key: str, n_layers: int, causal_layer: int) -> dict:
    d = json.loads((JLENS_RESULTS / f"jlens_ranks_gemma_{model_key}.json").read_text())
    if d["position"] != -2:
        raise ValueError(f"{model_key}: expected position=-2, got {d['position']}")
    if d["causal_layer"] != causal_layer:
        raise ValueError(f"{model_key}: expected causal_layer={causal_layer}, got {d['causal_layer']}")
    layers = d["layers"]
    records = d["records"]
    n_k = len(records[0]["clean_rank_by_layer"])

    per_prompt = [r["clean_rank_by_layer"] for r in records]  # 57 x n_k
    medians = [statistics.median(per_prompt[p][k] for p in range(len(records))) for k in range(n_k)]

    denom = n_layers - 1  # 47 / 61 -- the model's true layer count minus 1, NOT n_k-1
    xs = [k / denom for k in range(n_k)]

    # peak: minimum median rank. Ties broken toward the causal layer if the
    # causal layer is among the tied minimum, else toward the first
    # occurrence (lowest layer index).
    min_val = min(medians)
    tied_idx = [k for k in range(n_k) if medians[k] == min_val]
    causal_idx = layers.index(causal_layer)
    if len(tied_idx) > 1:
        print(f"  [note] {model_key}: {len(tied_idx)}-way tie for minimum median rank "
              f"({min_val}) at layers {[layers[k] for k in tied_idx]}")
    peak_idx = causal_idx if causal_idx in tied_idx else tied_idx[0]

    return {
        "layers": layers, "xs": xs, "medians": medians, "per_prompt": per_prompt,
        "n_k": n_k, "denom": denom, "causal_idx": causal_idx, "peak_idx": peak_idx,
    }


def verify(model_key: str, data: dict) -> None:
    exp = EXPECTED[model_key]
    layers = data["layers"]
    peak_layer = layers[data["peak_idx"]]
    peak_rank = data["medians"][data["peak_idx"]]
    causal_rank = data["medians"][data["causal_idx"]]
    last_rank = data["medians"][-1]

    print(f"  peak: layer={peak_layer} median_rank={peak_rank}  (expected L{exp['peak_layer']} rank {exp['peak_rank']})")
    print(f"  causal layer median_rank={causal_rank}  (expected {exp['causal_rank']})")
    print(f"  last lens layer ({layers[-1]}) median_rank={last_rank}  (expected {exp['last_rank']})")

    mismatches = []
    if peak_layer != exp["peak_layer"]:
        mismatches.append(f"peak_layer {peak_layer} != {exp['peak_layer']}")
    if peak_rank != exp["peak_rank"]:
        mismatches.append(f"peak_rank {peak_rank} != {exp['peak_rank']}")
    if causal_rank != exp["causal_rank"]:
        mismatches.append(f"causal_rank {causal_rank} != {exp['causal_rank']}")
    if last_rank != exp["last_rank"]:
        mismatches.append(f"last_rank {last_rank} != {exp['last_rank']}")
    if mismatches:
        print(f"  [WARNING] {model_key} mismatches vs spec: {mismatches}")
    else:
        print(f"  [verify] {model_key}: all values match spec")


def main() -> None:
    print("=" * 70)
    print("Figure 2 data verification")
    print("=" * 70)

    model_data = {}
    for model_key, label, color, n_layers, causal_layer in MODELS:
        print(f"\n{model_key}:")
        d = load_model(model_key, n_layers, causal_layer)
        verify(model_key, d)
        model_data[model_key] = d

    fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
    plt.rcParams.update({"font.size": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})

    # dominant band first, behind everything
    ax.axvspan(BAND_LO, BAND_HI, color=BAND_GREY, zorder=0)
    ax.text((BAND_LO + BAND_HI) / 2, 1.15, "path-patching\ndominant band",
            transform=ax.get_xaxis_transform(), fontsize=5.5, ha="center", va="bottom",
            color="#555555")

    print("\n" + "=" * 70)
    print("Headline plotted numbers")
    print("=" * 70)

    for model_key, label, color, n_layers, causal_layer in MODELS:
        d = model_data[model_key]
        xs = d["xs"]

        # spaghetti: all 57 per-prompt trajectories, very thin/faint
        for prompt_ranks in d["per_prompt"]:
            ax.plot(xs, prompt_ranks, color=color, linewidth=0.3, alpha=0.08, zorder=1)

        # median line
        ax.plot(xs, d["medians"], color=color, linewidth=1.2, solid_capstyle="round",
                label=f"{label} (median, n=57)", zorder=3)

        # causal-layer vertical dotted marker
        causal_x = causal_layer / d["denom"]
        ax.axvline(causal_x, color=color, linestyle=":", linewidth=0.9, zorder=2)
        ax.text(causal_x, 1.02, f"L{causal_layer}", transform=ax.get_xaxis_transform(),
                fontsize=6, ha="center", va="bottom", color=color)

        # peak marker
        peak_idx = d["peak_idx"]
        peak_x, peak_y = xs[peak_idx], d["medians"][peak_idx]
        peak_layer = d["layers"][peak_idx]
        ax.plot(peak_x, peak_y, "o", markerfacecolor="white", markeredgecolor=color,
                markersize=4, markeredgewidth=1.1, zorder=4)
        label_va = "bottom" if model_key == "27b" else "top"
        label_dy = 6 if model_key == "27b" else -6
        # Both peaks sit in the right half of the panel (x > 0.75) --
        # a rightward-growing label would run past x=1.0 and get clipped
        # (as it did on the first render), so grow leftward instead.
        near_right_edge = peak_x > 0.75
        label_ha = "right" if near_right_edge else "left"
        label_dx = -4 if near_right_edge else 4
        ax.annotate(f"L{peak_layer}, rank {int(peak_y)}", xy=(peak_x, peak_y),
                    xytext=(label_dx, label_dy), textcoords="offset points",
                    fontsize=6, ha=label_ha, va=label_va, color=color, clip_on=True)

        print(f"{model_key}: causal_x={causal_x:.4f}  peak=(L{peak_layer}, rank={int(peak_y)}) "
              f"at x={peak_x:.4f}  last_layer_rank={int(d['medians'][-1])}")

    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("relative depth (layer / final layer)", fontsize=8)
    ax.set_ylabel("median rank of correct answer (57 prompts)", fontsize=8)
    ax.text(1.01, 0.02, "better ↑", transform=ax.transAxes, fontsize=6.5,
            ha="left", va="bottom", rotation=90, color="#333333")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(loc="lower left", frameon=False, fontsize=6.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "fig2_decodability.png"
    fig.savefig(png_path, dpi=300)
    print(f"\n[save] wrote {png_path}")


if __name__ == "__main__":
    main()
