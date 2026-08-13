#!/usr/bin/env python3
"""
Plot mean correct-answer-token rank vs. layer depth, clean vs. corrupt, with
a vertical reference line at the causally-identified layer from path_patching
(38 for 12B, 54 for 27B) -- the headline figure comparing where the Jacobian
lens shows the answer becoming legible against where path-patching found the
network causally commits to it.

Usage:
    python plot_jlens_results.py --model gemma_12b
    python plot_jlens_results.py --model gemma_27b
    python plot_jlens_results.py --model gemma_12b --results_dir results

Runs anywhere -- no GPU, no model, just matplotlib over the saved JSON from
apply_jacobian_lens.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent


def plot_model(model_key: str, results_dir: Path, out_dir: Path) -> None:
    ranks_path = results_dir / f"jlens_ranks_{model_key}.json"
    if not ranks_path.exists():
        raise FileNotFoundError(
            f"missing {ranks_path} -- run apply_jacobian_lens.py --model {model_key} first."
        )
    data = json.loads(ranks_path.read_text())
    layers = data["layers"]
    mean_clean = data["aggregate"]["mean_clean_rank_by_layer"]
    mean_corrupt = data["aggregate"]["mean_corrupt_rank_by_layer"]
    causal_layer = data["causal_layer"]
    n_prompts = data["n_prompts"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, mean_clean, marker="o", markersize=4, label="clean (correct answer)", color="#1f77b4")
    ax.plot(layers, mean_corrupt, marker="o", markersize=4, label="corrupt (correct answer)", color="#d62728")

    ax.axvline(causal_layer, linestyle="--", color="gray", linewidth=1.5,
               label=f"causally-identified layer (L{causal_layer}, path_patching)")

    ax.set_yscale("log")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean rank of correct-answer token (log scale)", fontsize=12)
    ax.set_title(
        f"{data['hf_name']} -- Jacobian lens: answer legibility vs. depth\n"
        f"(n={n_prompts} fact-battery pairs, position={data['position']})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"jlens_rank_vs_layer_{model_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[save] wrote {out_path}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Jacobian lens rank-vs-layer results.")
    parser.add_argument("--model", required=True, choices=["gemma_12b", "gemma_27b"])
    parser.add_argument("--results_dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Defaults to --results_dir/figures")
    args = parser.parse_args()
    out_dir = args.out_dir or (args.results_dir / "figures")

    plot_model(args.model, args.results_dir, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
