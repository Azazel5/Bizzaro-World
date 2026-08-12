#!/usr/bin/env python3
"""
Differential SAE feature extraction between clean and corrupt prompts, using
the ATTN_OUT (not resid_post) Gemma Scope 2 SAEs, at the same causally-
identified layers as sae_differential_features.py (L38 for Gemma 12B-IT, L54
for Gemma 27B-IT — see path_patching/results/visuals/circuit_summary.json and
circuit_summary_27b.json).

Copy of sae_differential_features.py with exactly two substantive changes:
  1. sae_release / sae_id point at the attn_out SAE variant instead of res-all.
  2. The extraction hook is blocks.{L}.attn.hook_z (flattened across heads),
     not blocks.{L}.hook_resid_post -- NOT optional. The attn_out SAE's own
     training metadata (sae.cfg.metadata.hook_name) confirmed this during the
     fvu_spot_check.py work; feeding it hook_resid_post's 3840/5376-dim vector
     instead of hook_z's 4096-dim (flattened) vector is a matrix-multiply
     shape error at sae.encode(), not just a different number -- it can't run
     at all with the resid_post hook. Hook resolution here reuses the same
     declared-hook-name-first, dimension-check-as-safety-net pattern already
     validated in fvu_spot_check.py, rather than hardcoding blindly.

NOTE: the attn_out FVU spot check FAILED for both models even with the
correct, metadata-confirmed hook (12B: FVU=0.3184, L0=38.5 vs an ~20 target;
27B: FVU=0.4830, L0=62.1) -- root cause still unresolved, shelved rather than
chased further (see sae/fvu/fvu_results_gemma_12b_attn_out.json /
_gemma_27b_attn_out.json). Everything downstream of that -- reconstruction,
encode(), the feature activations this script reports -- runs through the
same admittedly-imperfect SAE. Read these differential features as a "what do
we get anyway" comparison against the resid_post run, not as validated the
way the resid_post features were.

Usage:
    python sae_differential_features_attn_out.py --model gemma_12b_attn_out
    python sae_differential_features_attn_out.py --model gemma_27b_attn_out --top_k 30

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


# sae_release_candidates ordered with the CONFIRMED-WORKING string first (see
# fvu_spot_check.py run logs: "gemma-scope-2-{12,27}b-it-att-all" is the real
# SAELens release alias; "-attn-out-all" / "-attn_out-all" both 404). sae_id
# matches the l0_small variant validated in fvu_spot_check.py's ATTN_OUT_CONFIGS.
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "gemma_12b_attn_out": {
        "model_name": "google/gemma-3-12b-it",
        "tl_model_name": "gemma-3-12b-it",
        "sae_release_candidates": [
            "gemma-scope-2-12b-it-att-all",
            "gemma-scope-2-12b-it-attn-out-all",
            "gemma-scope-2-12b-it-attn_out-all",
        ],
        "sae_id": "layer_38_width_16k_l0_small",
        "target_layer": 38,
        "battery_path": "fact_battery/gemma-3-12b-it.json",
        "neuronpedia_model_slug": "gemma-3-12b-it",
    },
    "gemma_27b_attn_out": {
        "model_name": "google/gemma-3-27b-it",
        "tl_model_name": "gemma-3-27b-it",
        "sae_release_candidates": [
            "gemma-scope-2-27b-it-att-all",
            "gemma-scope-2-27b-it-attn-out-all",
            "gemma-scope-2-27b-it-attn_out-all",
        ],
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


def _load_sae(sae_release_candidates: list[str], sae_id: str, device: str) -> tuple[Any, str]:
    """Try each release string in order, printing every attempt (mirrors fvu_spot_check.py)."""
    attempted: list[str] = []
    for release in sae_release_candidates:
        print(f"[load] attempting SAE release={release!r} sae_id={sae_id!r}", flush=True)
        attempted.append(release)
        try:
            sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
            if isinstance(sae, tuple):  # some SAELens versions return (sae, cfg, sparsity)
                sae = sae[0]
        except Exception as e:
            print(f"    [fail] {type(e).__name__}: {e}", flush=True)
            continue
        print(f"[load] SAE ready (release={release!r})", flush=True)
        return sae, release

    attempted_str = "\n".join(f"        - {r!r}" for r in attempted)
    print(
        f"\n[error] All SAE release strings failed for sae_id={sae_id!r}.\n"
        f"        Attempted, in order:\n{attempted_str}\n"
        f"        Check the pretrained SAEs list: {SAE_DOCS_URL}\n",
        flush=True,
    )
    raise SystemExit(1)


def _declared_hook_name(sae: Any) -> str | None:
    """
    Ground truth from the SAE's own training metadata (not inferred from
    shape) -- same helper as fvu_spot_check.py. Checks both the current
    SAELens location (cfg.metadata.hook_name) and the older one (cfg.hook_name).
    """
    cfg = getattr(sae, "cfg", None)
    if cfg is None:
        return None
    metadata = getattr(cfg, "metadata", None)
    name = getattr(metadata, "hook_name", None) if metadata is not None else None
    if name is None:
        name = getattr(cfg, "hook_name", None)
    return name


def _resolve_hook(model: HookedTransformer, sae: Any, target_layer: int, device: str) -> tuple[str, bool]:
    """
    Determine the attn_out hook name and whether it needs a per-head flatten.
    Returns (hook_name, needs_head_flatten).

    Prefers the SAE's declared hook_name (ground truth). Falls back to the
    already-confirmed-for-this-project blocks.{L}.attn.hook_z only if this
    SAELens version doesn't expose the metadata -- NOT a blind guess between
    multiple candidates, since fvu_spot_check.py already settled that
    question for both 12B (metadata-confirmed) and 27B (dimension-confirmed)
    this session.
    """
    d_in = getattr(getattr(sae, "cfg", None), "d_in", None)
    print(f"[hook] SAE expects d_in={d_in}", flush=True)

    declared = _declared_hook_name(sae)
    fallback = f"blocks.{target_layer}.attn.hook_z"
    hook_name = declared if declared is not None else fallback

    if declared is not None:
        print(f"[hook] SAE declares hook_name={declared!r} (ground truth from training metadata)", flush=True)
        if f"blocks.{target_layer}." not in declared:
            print(
                f"[warn] declared hook_name={declared!r} does not reference "
                f"'blocks.{target_layer}.' -- verify target_layer={target_layer} is really "
                f"what you want before trusting the result.",
                flush=True,
            )
    else:
        print(
            f"[warn] this SAELens version does not expose a declared hook_name -- "
            f"falling back to {fallback!r}, confirmed correct for this project's attn_out "
            f"SAEs during fvu_spot_check.py (not a fresh guess).",
            flush=True,
        )

    probe_tokens = model.to_tokens("The capital of France is")
    with t.no_grad():
        _, probe_cache = model.run_with_cache(
            probe_tokens.to(device),
            names_filter=lambda n: n == hook_name,
            return_type=None,
        )
    if hook_name not in probe_cache:
        print(
            f"\n[error] {hook_name!r} does not exist on this model's hook dictionary. "
            f"Stopping rather than guessing a substitute.\n",
            flush=True,
        )
        raise SystemExit(1)

    shape = tuple(probe_cache[hook_name].shape)
    needs_flatten = len(shape) == 4  # [batch, seq, n_heads, d_head]
    dim = (shape[-2] * shape[-1]) if needs_flatten else shape[-1]
    print(f"[hook] {hook_name!r} shape={shape}, dim={dim}" + (" (flattened)" if needs_flatten else ""), flush=True)
    if d_in is not None and dim != d_in:
        print(
            f"\n[error] {hook_name!r} dim {dim} != SAE d_in={d_in}. Stopping rather than "
            f"guessing a reshape. shape={shape}.\n",
            flush=True,
        )
        raise SystemExit(1)
    del probe_cache
    return hook_name, needs_flatten


def _extract_final_token_features(
    model: HookedTransformer,
    sae: Any,
    prompt: str,
    hook_name: str,
    needs_head_flatten: bool,
    device: str,
) -> t.Tensor:
    """Final-token activation at hook_name, encoded through the SAE. Returns [n_features] float32."""
    name_filter = lambda name: name == hook_name  # noqa: E731
    tokens = model.to_tokens(prompt)
    with t.no_grad():
        _, cache = model.run_with_cache(
            tokens.to(device),
            names_filter=name_filter,
            return_type=None,
        )
        raw = cache[hook_name][0, -1]  # [d_model] or [n_heads, d_head]
        if needs_head_flatten:
            raw = raw.reshape(-1)
        act = raw.float()
        feature_acts = sae.encode(act.unsqueeze(0)).squeeze(0).float().detach().cpu()
    del cache
    return feature_acts


def extract_differential_activations(
    model: HookedTransformer,
    sae: Any,
    battery: list[dict[str, str]],
    hook_name: str,
    needs_head_flatten: bool,
    device: str,
) -> tuple[t.Tensor, t.Tensor, int, int]:
    """
    Returns (clean_features, corrupt_features, n_successful_pairs, n_prompts).
    clean_features / corrupt_features: [n_successful_pairs, n_features], aligned by row.
    """
    n_prompts = len(battery)

    clean_rows: list[t.Tensor] = []
    corrupt_rows: list[t.Tensor] = []
    n_successful_pairs = 0

    for i, entry in enumerate(battery):
        clean_prompt = entry["clean_prompt"]
        corrupt_prompt = entry["corrupt_prompt"]
        print(f"[{i + 1}/{n_prompts}] clean={clean_prompt!r}  corrupt={corrupt_prompt!r}", flush=True)
        try:
            clean_feats = _extract_final_token_features(
                model, sae, clean_prompt, hook_name, needs_head_flatten, device
            )
            corrupt_feats = _extract_final_token_features(
                model, sae, corrupt_prompt, hook_name, needs_head_flatten, device
            )
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
    print(f"TOP FEATURES BY DIFFERENTIAL ACTIVATION — {model_key} Layer {target_layer} [attn_out]")
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
    print(
        "  NOTE: URL site-segment guessed as 'att-16k' (by analogy with the confirmed "
        "'gemmascope-res-16k' pattern used for resid_post). UNCONFIRMED for attn_out, and "
        "per earlier discussion Neuronpedia doesn't host explanations for our target layers "
        "at all -- these links are very likely to 404. Kept for parity with the resid_post "
        "script's output shape, not because they're expected to resolve."
    )
    for rec in records[:n]:
        url = (
            f"https://www.neuronpedia.org/{neuronpedia_slug}/"
            f"{target_layer}-gemmascope-att-16k/{rec['feature_index']}"
        )
        print(f"  feature {rec['feature_index']:<6} diff={rec['differential_activation']:+.3f}  {url}")


def run_model(model_key: str, results_dir: Path, top_k: int, device: str) -> None:
    config = MODEL_CONFIGS[model_key]
    target_layer = config["target_layer"]
    battery_path = REPO_ROOT / config["battery_path"]

    print(f"\n{'#' * 60}")
    print(f"# Differential SAE features [attn_out]: {model_key}")
    print(f"{'#' * 60}\n", flush=True)

    battery = load_fact_battery(battery_path)
    print(f"[data] {len(battery)} prompt pairs loaded from {battery_path}", flush=True)

    model = None
    sae = None
    try:
        model = _load_model(config["tl_model_name"], device)
        sae, sae_release = _load_sae(config["sae_release_candidates"], config["sae_id"], device)
        hook_name, needs_head_flatten = _resolve_hook(model, sae, target_layer, device)

        clean_features, corrupt_features, n_successful_pairs, n_prompts = extract_differential_activations(
            model=model,
            sae=sae,
            battery=battery,
            hook_name=hook_name,
            needs_head_flatten=needs_head_flatten,
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
                    "site": "attn_out",
                    "model_name": config["model_name"],
                    "sae_release": sae_release,
                    "sae_id": config["sae_id"],
                    "target_layer": target_layer,
                    "hook_name": hook_name,
                    "needs_head_flatten": needs_head_flatten,
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
        description="Extract differential attn_out SAE features between clean and corrupt prompts."
    )
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "differential_features_attn_out",
    )
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    device = _resolve_device()
    run_model(model_key=args.model, results_dir=args.results_dir, top_k=args.top_k, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
