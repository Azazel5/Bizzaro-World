#!/usr/bin/env python3
"""
Per-prompt firing evidence for a fixed, small set of SAE features, so they
can be hand-labeled from real evidence instead of guessed.

sae_differential_features.py computes the full [57, 16384] per-prompt
activation tensor but only ever saves the AGGREGATE mean/rate over all 16384
features -- discarding exactly the thing needed to hand-label a specific
feature ("which prompts/categories does this fire on"). This script re-runs
the same extraction (same hook, same SAE, same battery) but for only the
handful of features actually worth labeling, and keeps the per-prompt values
instead of throwing them away.

Target features are the "mag ∩ rate" overlap sets from the original
differential-feature run (both by-magnitude AND by-rate ranked, i.e. not a
rare one-off spike -- see sae_differential_features.py's own docstring on why
top-by-magnitude alone is untrustworthy), padded out to 5 for 27B (which only
had 2 in the overlap) with the next-best top-by-rate entries:
    12B: 432, 4043, 4086, 8614, 11511, 11641  (all 6 from the overlap set)
    27B: 669, 739 (overlap) + 288, 137, 1737 (next-best by |rate_differential|)

Cost: identical to the original differential-feature run -- SAE.encode()
computes all 16384 dimensions per forward pass regardless of how many we
keep, so restricting to ~5-6 target indices costs nothing extra; this is
purely "save more of what's already being computed."

Usage:
    python sae_feature_evidence.py --model gemma_12b
    python sae_feature_evidence.py --model gemma_27b
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
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


# Same validated SAE config as sae_differential_features.py -- see that
# file's docstring for why res-all/l0_small, not resid_post/l0_medium.
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "gemma_12b": {
        "model_name": "google/gemma-3-12b-it",
        "tl_model_name": "gemma-3-12b-it",
        "sae_release": "gemma-scope-2-12b-it-res-all",
        "sae_id": "layer_38_width_16k_l0_small",
        "target_layer": 38,
        "battery_path": "fact_battery/gemma-3-12b-it.json",
        "target_features": [432, 4043, 4086, 8614, 11511, 11641],
    },
    "gemma_27b": {
        "model_name": "google/gemma-3-27b-it",
        "tl_model_name": "gemma-3-27b-it",
        "sae_release": "gemma-scope-2-27b-it-res-all",
        "sae_id": "layer_54_width_16k_l0_small",
        "target_layer": 54,
        "battery_path": "fact_battery/gemma-3-27b-it.json",
        "target_features": [669, 739, 288, 137, 1737],
    },
}

SAE_DOCS_URL = "https://decoderesearch.github.io/SAELens/latest/pretrained_saes"


def _load_model(tl_model_name: str, device: str) -> HookedTransformer:
    print(f"[load] HookedTransformer {tl_model_name!r} (dtype=bfloat16, device={device})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(
        tl_model_name, device=device, dtype=t.bfloat16,
    )
    print("[load] model ready", flush=True)
    return model


def _load_sae(sae_release: str, sae_id: str, device: str) -> Any:
    print(f"[load] SAE release={sae_release!r} sae_id={sae_id!r}", flush=True)
    try:
        sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
        if isinstance(sae, tuple):
            sae = sae[0]
    except Exception as e:
        print(
            f"\n[error] Failed to load SAE with release={sae_release!r}, sae_id={sae_id!r}\n"
            f"        {type(e).__name__}: {e}\n"
            f"Check the pretrained SAEs list: {SAE_DOCS_URL}\n",
            flush=True,
        )
        raise SystemExit(1)
    print("[load] SAE ready", flush=True)
    return sae


def _extract_final_token_features(
    model: HookedTransformer, sae: Any, prompt: str, hook_name: str, device: str,
) -> t.Tensor:
    """Final-token resid_post at target_layer, encoded through the SAE. Returns [n_features] float32."""
    name_filter = lambda name: name == hook_name  # noqa: E731
    tokens = model.to_tokens(prompt)
    with t.no_grad():
        _, cache = model.run_with_cache(tokens.to(device), names_filter=name_filter, return_type=None)
        resid = cache[hook_name][0, -1, :].float()
        feature_acts = sae.encode(resid.unsqueeze(0)).squeeze(0).float().detach().cpu()
    del cache
    return feature_acts


def run_model(model_key: str, results_dir: Path, device: str) -> None:
    config = MODEL_CONFIGS[model_key]
    target_features = config["target_features"]
    target_layer = config["target_layer"]
    hook_name = f"blocks.{target_layer}.hook_resid_post"

    print(f"\n{'#' * 60}")
    print(f"# SAE feature evidence: {model_key}, targets={target_features}")
    print(f"{'#' * 60}\n", flush=True)

    battery = load_fact_battery(REPO_ROOT / config["battery_path"])
    print(f"[data] {len(battery)} prompt pairs loaded from {config['battery_path']}", flush=True)

    model = _load_model(config["tl_model_name"], device)
    sae = _load_sae(config["sae_release"], config["sae_id"], device)

    records: list[dict[str, Any]] = []
    n = len(battery)
    for i, entry in enumerate(battery):
        try:
            clean_feats = _extract_final_token_features(model, sae, entry["clean_prompt"], hook_name, device)
            corrupt_feats = _extract_final_token_features(model, sae, entry["corrupt_prompt"], hook_name, device)
        except Exception as e:
            print(f"    [skip] pair {i} failed: {type(e).__name__}: {e}", flush=True)
            continue

        records.append({
            "idx": i,
            "category": entry.get("category"),
            "clean_prompt": entry["clean_prompt"],
            "corrupt_prompt": entry["corrupt_prompt"],
            "features": {
                str(fidx): {
                    "clean_activation": float(clean_feats[fidx]),
                    "corrupt_activation": float(corrupt_feats[fidx]),
                }
                for fidx in target_features
            },
        })
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"    [{i + 1}/{n}] extracted", flush=True)

    print(f"\n[extract] {len(records)}/{n} pairs succeeded", flush=True)

    # Quick per-feature firing summary, printed now so you don't have to open
    # the JSON to sanity-check this run before moving to the table builder.
    print("\n[summary] firing prompts per target feature (nonzero activation):")
    for fidx in target_features:
        clean_fires = [r for r in records if r["features"][str(fidx)]["clean_activation"] > 0]
        corrupt_fires = [r for r in records if r["features"][str(fidx)]["corrupt_activation"] > 0]
        print(f"\n  feature {fidx}:")
        print(f"    clean fires ({len(clean_fires)}): "
              f"{[(r['idx'], r['category']) for r in clean_fires]}")
        print(f"    corrupt fires ({len(corrupt_fires)}): "
              f"{[(r['idx'], r['category']) for r in corrupt_fires]}")

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"feature_evidence_{model_key}.json"
    out_path.write_text(json.dumps({
        "model_key": model_key,
        "target_layer": target_layer,
        "sae_release": config["sae_release"],
        "sae_id": config["sae_id"],
        "target_features": target_features,
        "n_prompts": len(records),
        "records": records,
    }, indent=2) + "\n")
    print(f"\n[save] wrote {out_path}", flush=True)

    del model, sae
    gc.collect()
    if device == "cuda":
        t.cuda.empty_cache()
    print(f"[cleanup] freed model + SAE for {model_key}", flush=True)


def _resolve_device() -> str:
    if t.cuda.is_available():
        return "cuda"
    print("[warn] CUDA not available -- falling back to CPU. This will be slow.", flush=True)
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-prompt firing evidence for hand-labeling SAE features.")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--results_dir", type=Path, default=SCRIPT_DIR / "results")
    args = parser.parse_args()

    device = _resolve_device()
    run_model(args.model, args.results_dir, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
