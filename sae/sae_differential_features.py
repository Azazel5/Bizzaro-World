#!/usr/bin/env python3
"""
Differential SAE feature extraction between clean and corrupt prompts, at the
layer each model's path-patching circuit flagged as causally dominant
(L38 for Gemma 12B-IT, L54 for Gemma 27B-IT — see path_patching/results/visuals/
circuit_summary.json and circuit_summary_27b.json).

For each aligned (clean_prompt, corrupt_prompt) pair in the fact battery, extracts
the final-token residual stream at the target layer, encodes it through the
Gemma Scope SAE validated by fvu_spot_check.py, and reports which individual SAE
features differ most between the two conditions — the feature-level analogue of
the head-level load-bearing/suppressive ranking already run in path_patching/.

Usage:
    python sae_differential_features.py --model gemma_12b
    python sae_differential_features.py --model gemma_27b --top_k 30

Run on a single A100 80GB. One model+SAE loaded at a time.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import torch as t

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from transformer_lens import HookedTransformer  # noqa: E402
from sae_lens import SAE  # noqa: E402

from shared.fact_battery import load_fact_battery  # noqa: E402


# NOTE on 12B config: fvu_spot_check.py's MODEL_CONFIGS originally pointed at
# sae_release="gemma-scope-2-12b-it-resid_post" (commit 1254ea5), but that string
# was replaced by "gemma-scope-2-12b-it-res-all" in commit 17eb207 ("Extracted
# actual 12B and 27B SAE model strings..."). The committed FVU=0.0312 PASS result
# in sae/fvu/fvu_results_gemma_12b.json was produced under the post-rename config
# (res-all / l0_small), not resid_post / l0_medium. This file matches that
# validated config for both models — edit here if you deliberately want a
# different release/sae_id.
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "gemma_12b": {
        "model_name": "google/gemma-3-12b-it",
        "tl_model_name": "gemma-3-12b-it",
        "sae_release": "gemma-scope-2-12b-it-res-all",
        "sae_id": "layer_38_width_16k_l0_small",
        "target_layer": 38,
        "battery_path": "fact_battery/gemma-3-12b-it.json",
        "neuronpedia_model_slug": "gemma-3-12b-it",
    },
    "gemma_27b": {
        "model_name": "google/gemma-3-27b-it",
        "tl_model_name": "gemma-3-27b-it",
        "sae_release": "gemma-scope-2-27b-it-res-all",
        "sae_id": "layer_54_width_16k_l0_small",
        "target_layer": 54,
        "battery_path": "fact_battery/gemma-3-27b-it.json",
        "neuronpedia_model_slug": "gemma-3-27b-it",
    },
}

SAE_DOCS_URL = "https://decoderesearch.github.io/SAELens/latest/pretrained_saes"
MIN_SUCCESS_FRACTION = 0.8


def _load_model(tl_model_name: str, device: str) -> HookedTransformer:
    print(f"[load] HookedTransformer {tl_model_name!r} (dtype=bfloat16, device={device})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(
        tl_model_name,
        device=device,
        dtype=t.bfloat16,
    )
    print("[load] model ready", flush=True)
    return model


def _load_sae(sae_release: str, sae_id: str, device: str) -> Any:
    print(f"[load] SAE release={sae_release!r} sae_id={sae_id!r}", flush=True)
    try:
        sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
        if isinstance(sae, tuple):  # some SAELens versions return (sae, cfg, sparsity)
            sae = sae[0]
    except Exception as e:
        print(
            f"\n[error] Failed to load SAE with release={sae_release!r}, sae_id={sae_id!r}\n"
            f"        {type(e).__name__}: {e}\n"
            f"SAE release/id string is likely incorrect for this SAELens version.\n"
            f"Check the pretrained SAEs list: {SAE_DOCS_URL}\n",
            flush=True,
        )
        raise SystemExit(1)
    print("[load] SAE ready", flush=True)
    return sae


def _extract_final_token_features(
    model: HookedTransformer,
    sae: Any,
    prompt: str,
    hook_name: str,
    device: str,
) -> t.Tensor:
    """Final-token resid_post at target_layer, encoded through the SAE. Returns [n_features] float32."""
    name_filter = lambda name: name == hook_name  # noqa: E731
    tokens = model.to_tokens(prompt)
    with t.no_grad():
        _, cache = model.run_with_cache(
            tokens.to(device),
            names_filter=name_filter,
            return_type=None,
        )
        resid = cache[hook_name][0, -1, :].float()
        feature_acts = sae.encode(resid.unsqueeze(0)).squeeze(0).float().detach().cpu()
    del cache
    return feature_acts


def extract_differential_activations(
    model: HookedTransformer,
    sae: Any,
    battery: list[dict[str, str]],
    target_layer: int,
    device: str,
) -> tuple[t.Tensor, t.Tensor, int, int]:
    """
    Returns (clean_features, corrupt_features, n_successful_pairs, n_prompts).
    clean_features / corrupt_features: [n_successful_pairs, n_features], aligned by row.
    """
    hook_name = f"blocks.{target_layer}.hook_resid_post"
    n_prompts = len(battery)

    clean_rows: list[t.Tensor] = []
    corrupt_rows: list[t.Tensor] = []
    n_successful_pairs = 0

    for i, entry in enumerate(battery):
        clean_prompt = entry["clean_prompt"]
        corrupt_prompt = entry["corrupt_prompt"]
        print(f"[{i + 1}/{n_prompts}] clean={clean_prompt!r}  corrupt={corrupt_prompt!r}", flush=True)
        try:
            clean_feats = _extract_final_token_features(model, sae, clean_prompt, hook_name, device)
            corrupt_feats = _extract_final_token_features(model, sae, corrupt_prompt, hook_name, device)
        except Exception as e:
            print(f"    [skip] pair {i} failed: {type(e).__name__}: {e}", flush=True)
            continue

        clean_rows.append(clean_feats)
        corrupt_rows.append(corrupt_feats)
        n_successful_pairs += 1

    if n_successful_pairs == 0:
        raise RuntimeError("No prompt pairs succeeded — cannot compute differential features.")

    clean_features = t.stack(clean_rows, dim=0)
    corrupt_features = t.stack(corrupt_rows, dim=0)
    return clean_features, corrupt_features, n_successful_pairs, n_prompts


def compute_differential(
    clean_features: t.Tensor, corrupt_features: t.Tensor
) -> dict[str, t.Tensor]:
    clean_mean = clean_features.mean(dim=0)
    corrupt_mean = corrupt_features.mean(dim=0)
    differential = clean_mean - corrupt_mean

    clean_rate = (clean_features > 0).float().mean(dim=0)
    corrupt_rate = (corrupt_features > 0).float().mean(dim=0)
    rate_differential = clean_rate - corrupt_rate

    return {
        "clean_mean": clean_mean,
        "corrupt_mean": corrupt_mean,
        "differential": differential,
        "clean_rate": clean_rate,
        "corrupt_rate": corrupt_rate,
        "rate_differential": rate_differential,
    }


def _build_feature_records(
    indices: t.Tensor, stats: dict[str, t.Tensor]
) -> list[dict[str, Any]]:
    records = []
    for idx_tensor in indices:
        idx = int(idx_tensor)
        records.append(
            {
                "feature_index": idx,
                "differential_activation": float(stats["differential"][idx]),
                "clean_mean_activation": float(stats["clean_mean"][idx]),
                "corrupt_mean_activation": float(stats["corrupt_mean"][idx]),
                "clean_activation_rate": float(stats["clean_rate"][idx]),
                "corrupt_activation_rate": float(stats["corrupt_rate"][idx]),
                "rate_differential": float(stats["rate_differential"][idx]),
            }
        )
    return records


def select_top_features(
    stats: dict[str, t.Tensor], top_k: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    top_k = min(top_k, stats["differential"].numel())

    _, mag_idx = stats["differential"].abs().topk(top_k)
    top_by_magnitude = _build_feature_records(mag_idx, stats)

    _, rate_idx = stats["rate_differential"].abs().topk(top_k)
    top_by_rate = _build_feature_records(rate_idx, stats)

    return top_by_magnitude, top_by_rate


def _print_top_features_table(model_key: str, target_layer: int, records: list[dict[str, Any]]) -> None:
    print()
    print("=" * 60)
    print(f"TOP FEATURES BY DIFFERENTIAL ACTIVATION — {model_key} Layer {target_layer}")
    print("=" * 60)
    print(f"{'Rank':<5}| {'Feature':<8}| {'Diff Act':<9}| {'Clean Mean':<11}| {'Corrupt Mean':<13}| {'Clean Rate':<11}| Corrupt Rate")
    print("-" * 60)
    for rank, rec in enumerate(records, start=1):
        print(
            f"{rank:<5}| {rec['feature_index']:<8}| "
            f"{rec['differential_activation']:+.3f}   | "
            f"{rec['clean_mean_activation']:<11.3f}| "
            f"{rec['corrupt_mean_activation']:<13.3f}| "
            f"{rec['clean_activation_rate']:<11.2f}| "
            f"{rec['corrupt_activation_rate']:.2f}"
        )


def _print_neuronpedia_links(neuronpedia_slug: str, target_layer: int, records: list[dict[str, Any]], n: int = 10) -> None:
    print()
    print("=" * 60)
    print(f"NEURONPEDIA LINKS (top {min(n, len(records))} by differential activation)")
    print("=" * 60)
    for rec in records[:n]:
        url = (
            f"https://www.neuronpedia.org/{neuronpedia_slug}/"
            f"{target_layer}-gemmascope-res-16k/{rec['feature_index']}"
        )
        print(f"  feature {rec['feature_index']:<6} diff={rec['differential_activation']:+.3f}  {url}")


def run_model(model_key: str, results_dir: Path, top_k: int, device: str) -> None:
    config = MODEL_CONFIGS[model_key]
    target_layer = config["target_layer"]
    battery_path = REPO_ROOT / config["battery_path"]

    print(f"\n{'#' * 60}")
    print(f"# Differential SAE features: {model_key}")
    print(f"{'#' * 60}\n", flush=True)

    battery = load_fact_battery(battery_path)
    print(f"[data] {len(battery)} prompt pairs loaded from {battery_path}", flush=True)

    model = None
    sae = None
    try:
        model = _load_model(config["tl_model_name"], device)
        sae = _load_sae(config["sae_release"], config["sae_id"], device)

        clean_features, corrupt_features, n_successful_pairs, n_prompts = extract_differential_activations(
            model=model,
            sae=sae,
            battery=battery,
            target_layer=target_layer,
            device=device,
        )
        print(f"[extract] {n_successful_pairs}/{n_prompts} pairs succeeded", flush=True)

        if n_successful_pairs < MIN_SUCCESS_FRACTION * n_prompts:
            warnings.warn(
                f"Only {n_successful_pairs}/{n_prompts} pairs succeeded "
                f"({n_successful_pairs / n_prompts:.0%} < {MIN_SUCCESS_FRACTION:.0%} threshold). "
                f"Results below are based on a reduced sample — treat with caution.",
                stacklevel=2,
            )

        stats = compute_differential(clean_features, corrupt_features)
        top_by_magnitude, top_by_rate = select_top_features(stats, top_k)

        _print_top_features_table(model_key, target_layer, top_by_magnitude)
        _print_neuronpedia_links(config["neuronpedia_model_slug"], target_layer, top_by_magnitude, n=10)

        differential = stats["differential"]
        differential_stats = {
            "mean": float(differential.mean()),
            "std": float(differential.std()),
            "max": float(differential.max()),
            "min": float(differential.min()),
            "n_nonzero_features": int((differential != 0).sum()),
        }

        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"differential_features_{model_key}.json"
        out_path.write_text(
            json.dumps(
                {
                    "model_key": model_key,
                    "model_name": config["model_name"],
                    "sae_release": config["sae_release"],
                    "sae_id": config["sae_id"],
                    "target_layer": target_layer,
                    "battery_path": str(battery_path),
                    "n_prompts": n_prompts,
                    "n_successful_pairs": n_successful_pairs,
                    "top_features_by_magnitude": top_by_magnitude,
                    "top_features_by_rate": top_by_rate,
                    "differential_stats": differential_stats,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\n[save] wrote {out_path}", flush=True)

    finally:
        del model
        del sae
        gc.collect()
        if device == "cuda":
            t.cuda.empty_cache()
        print(f"[cleanup] freed model + SAE for {model_key}", flush=True)


def _resolve_device() -> str:
    if t.cuda.is_available():
        return "cuda"
    print("[warn] CUDA not available — falling back to CPU. This will be slow.", flush=True)
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract differential SAE features between clean and corrupt prompts."
    )
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--results_dir", type=Path, default=REPO_ROOT / "results" / "sae")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    device = _resolve_device()
    run_model(model_key=args.model, results_dir=args.results_dir, top_k=args.top_k, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
