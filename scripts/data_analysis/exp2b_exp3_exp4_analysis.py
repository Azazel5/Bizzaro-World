#!/usr/bin/env python3
"""
Generate figures for Experiments 2B (sublayer decomposition), 3 (entity token),
and 4 (attention head routing).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_json_files(mode: str, exp_name: str, data_root: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Load all JSON files for a given experiment and mode.
    
    Returns tuple of (pairs list, n_layers) where n_layers comes from the JSON metadata.
    """
    exp_prefix = {
        "2b": "experiment2b_",
        "3": "experiment3_",
        "4": "experiment4_",
    }[exp_name]
    
    # Determine which directory structure to use
    if data_root:
        # Use model-specific structure (e.g., gemma-12b-it/phase{N}/{MODE}/)
        if exp_name == "2b":
            json_file = data_root / "phase2" / mode.upper() / "exp2b" / f"{exp_prefix}{mode.upper()}.json"
        else:
            json_file = data_root / f"phase{exp_name}" / mode.upper() / f"{exp_prefix}{mode.upper()}.json"
    else:
        # Use root-level structure (e.g., exp2b/{MODE}/, exp3/{MODE}/)
        for dir_variant in [f"exp{exp_name}", f"exp{exp_name.upper()}"]:
            json_file = Path(dir_variant) / mode.upper() / f"{exp_prefix}{mode.upper()}.json"
            if json_file.exists():
                with open(json_file) as f:
                    data = json.load(f)
                n_layers = data.get("n_layers")
                return data.get("pairs", []), n_layers
        return [], None
    
    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
        n_layers = data.get("n_layers")
        return data.get("pairs", []), n_layers
    
    return [], None


def _plot_exp2b_sublayer(
    rows_by_mode: Dict[str, List[Dict[str, Any]]],
    modes: Sequence[str],
    output_path: Path,
    model_name: str,
    n_layers: Optional[int] = None,
) -> None:
    """Plot sublayer decomposition (resid_pre, attn_out, resid_mid, mlp_out, resid_post)."""
    fig, axes = plt.subplots(1, len(modes), figsize=(15, 5), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    
    hooks = ["resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"]
    
    for ax, mode in zip(axes, modes):
        rows = rows_by_mode.get(mode, [])
        if not rows:
            continue
        
        if n_layers is None:
            n_layers_to_use = max(
                int(layer) for row in rows for layer in row.get("results_by_layer", {}).keys()
            ) + 1
        else:
            n_layers_to_use = n_layers
        hook_deltas = {hook: [] for hook in hooks}
        
        for layer in range(n_layers_to_use):
            layer_str = str(layer)
            for hook in hooks:
                hook_key = f"hook_{hook}"
                deltas = []
                for row in rows:
                    results = row.get("results_by_layer", {}).get(layer_str, {})
                    if hook_key in results:
                        deltas.append(results[hook_key].get("ld_delta", 0))
                hook_deltas[hook].append(np.mean(deltas) if deltas else 0)
        
        layers = np.arange(n_layers_to_use)
        colors = {"resid_pre": "C0", "attn_out": "C1", "resid_mid": "C2", "mlp_out": "C3", "resid_post": "C4"}
        for hook in hooks:
            ax.plot(layers, hook_deltas[hook], marker="o", label=hook, color=colors[hook])
        
        ax.set_xlabel("Layer")
        ax.set_xlim(0, n_layers_to_use - 1)
        ax.set_xticks(np.arange(0, n_layers_to_use, 4))
        if mode == modes[0]:
            ax.set_ylabel("Mean ld_delta")
        ax.set_title(f"Mode {mode.upper()} (n={len(rows)})")
        ax.grid(True, alpha=0.3)
        if mode == modes[0]:
            ax.legend(loc="best", fontsize=9)
    
    fig.suptitle(f"{model_name}: Sublayer Decomposition (Exp 2B)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path / "exp2b_sublayer_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ exp2b_sublayer_decomposition.png")


def _plot_exp3_entity(rows_by_mode: Dict[str, List[Dict[str, Any]]], 
                      modes: Sequence[str], output_path: Path, model_name: str,
                      n_layers: Optional[int] = None) -> None:
    """Plot entity token activation across layers."""
    fig, axes = plt.subplots(1, len(modes), figsize=(15, 5), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    
    for ax, mode in zip(axes, modes):
        rows = rows_by_mode.get(mode, [])
        if not rows:
            continue
        
        # Dynamically determine n_layers from data if not provided
        if n_layers is None:
            max_layer = max(
                max(int(k) for k in row.get("results_by_layer", {}).keys()) 
                for row in rows
            )
            n_layers_to_use = max_layer + 1
        else:
            n_layers_to_use = n_layers
        
        entity_activations = []

        for row in rows:
            # Exp3 JSON stores canonical per-layer deltas as an array.
            # Keep a fallback to nested maps for older/alternate schemas.
            ld_deltas_array = row.get("ld_delta_vs_clean_baseline_by_layer")
            if isinstance(ld_deltas_array, list):
                layer_damages = [
                    float(ld_deltas_array[layer]) if layer < len(ld_deltas_array) else 0.0
                    for layer in range(n_layers_to_use)
                ]
            else:
                layer_damages = []
                for layer in range(n_layers_to_use):
                    ld_delta = row.get("results_by_layer", {}).get(str(layer), {}).get("ld_delta", 0)
                    layer_damages.append(ld_delta)
            entity_activations.append(layer_damages)
        
        entity_mean = np.mean(entity_activations, axis=0)
        layers = np.arange(n_layers_to_use)
        
        ax.plot(layers, entity_mean, marker="o", linewidth=2, markersize=6, color="C0")
        ax.set_xlabel("Layer")
        if mode == modes[0]:
            ax.set_ylabel("Mean ld_delta")
        ax.set_title(f"Mode {mode.upper()} (n={len(rows)})")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"{model_name}: Entity Token Activation Profile (Exp 3)", 
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path / "exp3_entity_token.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ exp3_entity_token.png")


def _plot_exp4_head_heatmap(rows_by_mode: Dict[str, List[Dict[str, Any]]], 
                            modes: Sequence[str], output_path: Path, model_name: str) -> None:
    """Plot attention head routing heatmap."""
    fig, axes = plt.subplots(1, len(modes), figsize=(18, 6))
    if len(modes) == 1:
        axes = [axes]
    
    for ax, mode in zip(axes, modes):
        rows = rows_by_mode.get(mode, [])
        if not rows:
            continue
        
        results_by_layer_head = rows[0].get("results_by_layer_head", {})
        n_layers = len(results_by_layer_head)
        n_heads = len(results_by_layer_head.get("0", {}))
        
        heatmap = np.zeros((n_layers, n_heads))
        
        for layer in range(n_layers):
            for head in range(n_heads):
                head_deltas = []
                for row in rows:
                    results = row.get("results_by_layer_head", {})
                    if str(layer) in results and str(head) in results[str(layer)]:
                        ld_delta = results[str(layer)][str(head)].get("ld_delta", 0)
                        head_deltas.append(ld_delta)
                heatmap[layer, head] = np.mean(head_deltas) if head_deltas else 0
        
        im = ax.imshow(heatmap, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)
        ax.set_xlabel("Attention Head")
        if mode == modes[0]:
            ax.set_ylabel("Layer")
        ax.set_title(f"Mode {mode.upper()} (n={len(rows)})")
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(0, n_layers, 2))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean ld_delta", fontsize=9)
    
    fig.suptitle(f"{model_name}: Attention Head Routing (Exp 4)", 
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path / "exp4_head_routing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ exp4_head_routing.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gemma-2b", help="Model name for output directory")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional root directory containing phase*/MODE JSON files",
    )
    args = parser.parse_args()
    
    output_base = Path(args.model_name) / "figures"
    output_base.mkdir(parents=True, exist_ok=True)
    
    modes = ["A", "B", "C"]
    
    # Load and plot data for all experiments
    for exp_name in ["2b", "3", "4"]:
        print(f"\n=== Loading Experiment {exp_name.upper()} ===")
        rows_by_mode = {}
        exp_n_layers: Optional[int] = None
        for mode in modes:
            rows, n_layers = _load_json_files(mode.lower(), exp_name, args.data_root)
            if rows:
                rows_by_mode[mode] = rows
                if n_layers is not None:
                    exp_n_layers = n_layers
                print(f"Mode {mode}: {len(rows)} pairs")
        
        if not rows_by_mode:
            print(f"No data for experiment {exp_name}, skipping")
            continue
        
        # Generate figures
        if exp_name == "2b":
            _plot_exp2b_sublayer(
                rows_by_mode,
                list(rows_by_mode.keys()),
                output_base,
                args.model_name,
                n_layers=exp_n_layers,
            )
        elif exp_name == "3":
            _plot_exp3_entity(
                rows_by_mode,
                list(rows_by_mode.keys()),
                output_base,
                args.model_name,
                n_layers=exp_n_layers,
            )
        elif exp_name == "4":
            _plot_exp4_head_heatmap(rows_by_mode, list(rows_by_mode.keys()), output_base, args.model_name)
    
    print(f"\n✓ All figures generated in {output_base}")


if __name__ == "__main__":
    main()
