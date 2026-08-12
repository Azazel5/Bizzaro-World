#!/usr/bin/env python3
"""
FVU (Fraction of Variance Unexplained) spot check for Gemma Scope SAEs against
the BizzaroWorld fact battery.

Validates SAE reconstruction quality at the causally-identified layer for each
model (L38 for Gemma 12B-IT, L54 for Gemma 27B-IT -- see
path_patching/results/visuals/circuit_summary.json / circuit_summary_27b.json),
across two independent SAE sites:

  --site resid_post (default) : full residual stream, blocks.{L}.hook_resid_post.
      Already validated: 12B FVU=0.0312, 27B FVU=0.0501
      (results/fvu/fvu_results_gemma_12b.json / _gemma_27b.json).
  --site attn_out              : the attention sublayer's own contribution to
      the residual stream, blocks.{L}.hook_attn_out. This is a DIFFERENT SAE
      (different weights, trained on a different tensor) from resid_post -- a
      passing resid_post FVU says nothing about attn_out reconstruction
      quality, so this is a genuinely separate validation pass, not a repeat
      of the resid_post run.

For each model+site, extracts the final-token target activation for every
clean prompt in the battery, reconstructs it through the SAE, and reports how
much variance the SAE fails to explain. This is a sanity check that the SAE is
a faithful lens before using it for further circuit analysis.

Usage:
    python fvu_spot_check.py                                    # resid_post, both models
    python fvu_spot_check.py --model gemma_12b                  # resid_post, one model
    python fvu_spot_check.py --site attn_out                    # attn_out, both models
    python fvu_spot_check.py --site attn_out --model gemma_12b_attn_out

Run on a single A100 80GB. Models are loaded one at a time, never simultaneously.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
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


# ---------------------------------------------------------------------------
# Site configs
# ---------------------------------------------------------------------------
# resid_post: VALIDATED (see results/fvu/fvu_results_gemma_12b.json / _27b.json).
RESID_POST_CONFIGS: dict[str, dict[str, Any]] = {
    "gemma_12b": {
        "model_name": "google/gemma-3-12b-it",
        "tl_model_name": "gemma-3-12b-it",
        "sae_release_candidates": ["gemma-scope-2-12b-it-res-all"],
        "sae_id": "layer_38_width_16k_l0_small",
        "target_layer": 38,
        "battery_path": "fact_battery/gemma-3-12b-it.json",
    },
    "gemma_27b": {
        "model_name": "google/gemma-3-27b-it",
        "tl_model_name": "gemma-3-27b-it",
        "sae_release_candidates": ["gemma-scope-2-27b-it-res-all"],
        "sae_id": "layer_54_width_16k_l0_small",
        "target_layer": 54,
        "battery_path": "fact_battery/gemma-3-27b-it.json",
    },
}

# attn_out: HF repo tree CONFIRMED (browsed directly, not inferred):
#   https://huggingface.co/google/gemma-scope-2-12b-it/tree/main/attn_out_all/layer_38_width_16k_l0_small
#   https://huggingface.co/google/gemma-scope-2-27b-it/tree/main/attn_out_all/layer_54_width_16k_l0_small
# The site folder is "attn_out_all" (underscore, matching the task spec's
# original guess -- an earlier "att-all" guess extrapolated from a partial
# read of the 4B SAELens YAML was wrong, corrected here). "l0_small" also
# exists for attn_out at both layers (an "l0_big" variant exists too, but
# l0_small is used here to match the sparsity target of the already-validated
# resid_post SAEs -- same l0 target on both sites keeps the FVU comparison
# apples-to-apples rather than confounding site with sparsity). Deliberately
# NOT set to "l0_medium" (the task spec's original guess) since that's the
# string the resid_post config moved away from (commit 17eb207) after finding
# it didn't resolve. sae_release_candidates still tries multiple hyphenation
# variants in order and prints every attempt, since the HF folder name and the
# SAELens `release` alias are not guaranteed to be spelled identically.
ATTN_OUT_CONFIGS: dict[str, dict[str, Any]] = {
    "gemma_12b_attn_out": {
        "model_name": "google/gemma-3-12b-it",
        "tl_model_name": "gemma-3-12b-it",
        "sae_release_candidates": [
            "gemma-scope-2-12b-it-attn-out-all",
            "gemma-scope-2-12b-it-attn_out-all",
            "gemma-scope-2-12b-it-att-all",
        ],
        "sae_id": "layer_38_width_16k_l0_small",
        "target_layer": 38,
        "battery_path": "fact_battery/gemma-3-12b-it.json",
    },
    "gemma_27b_attn_out": {
        "model_name": "google/gemma-3-27b-it",
        "tl_model_name": "gemma-3-27b-it",
        "sae_release_candidates": [
            "gemma-scope-2-27b-it-attn-out-all",
            "gemma-scope-2-27b-it-attn_out-all",
            "gemma-scope-2-27b-it-att-all",
        ],
        "sae_id": "layer_54_width_16k_l0_small",
        "target_layer": 54,
        "battery_path": "fact_battery/gemma-3-27b-it.json",
    },
}

SITE_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "resid_post": RESID_POST_CONFIGS,
    "attn_out": ATTN_OUT_CONFIGS,
}

# Backward-compat alias -- other scripts in this repo reference
# "fvu_spot_check.py's MODEL_CONFIGS" in comments to mean the resid_post table.
MODEL_CONFIGS = RESID_POST_CONFIGS

SAE_DOCS_URL = "https://decoderesearch.github.io/SAELens/latest/pretrained_saes"


def _load_battery(battery_path: Path) -> list[dict[str, str]]:
    if not battery_path.exists():
        raise FileNotFoundError(
            f"Fact battery not found at {battery_path}\n"
            f"Expected a JSON file with entries containing 'clean_prompt'."
        )
    battery = json.loads(battery_path.read_text())
    if not isinstance(battery, list) or not battery:
        raise ValueError(f"Fact battery at {battery_path} is empty or malformed")
    return battery


def _load_model(model_name: str, tl_model_name: str, device: str) -> HookedTransformer:
    print(f"[load] HookedTransformer for {model_name} (dtype=bfloat16, device={device})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(
        tl_model_name,
        device=device,
        dtype=t.bfloat16,
    )
    print("[load] model ready", flush=True)
    return model


def _load_sae(sae_release_candidates: list[str], sae_id: str, device: str) -> tuple[Any, str]:
    """
    Try each release string in order (list-of-one for the validated resid_post
    configs, so this path is behaviorally identical to before for --site
    resid_post). Prints every attempted string so a wrong guess is immediately
    visible instead of a silent failure.

    Returns (sae, release_string_that_worked).
    """
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
        f"        Check the pretrained SAEs list: {SAE_DOCS_URL}\n"
        f"        or browse the HF repo tree directly (swap in the correct model size), e.g.\n"
        f"        https://huggingface.co/google/gemma-scope-2-12b-it/tree/main\n"
        f"        to find the exact release/sae_id naming this SAELens version expects.\n",
        flush=True,
    )
    raise SystemExit(1)


def _resolve_hook(
    site: str,
    model: HookedTransformer,
    sae: Any,
    target_layer: int,
    device: str,
) -> tuple[str, bool]:
    """
    Determine which hook to extract from and whether it needs a per-head
    flatten before encoding. Returns (hook_name, needs_head_flatten).

    resid_post: fixed, no probing needed -- unchanged from the original script.

    attn_out: probes the model once with a throwaway prompt to check which
    hook exists and whether its last dim matches the SAE's expected d_in.
    Tries hook_attn_out (post-W_O, d_model-dim -- what the Gemma Scope paper
    documents attn_out SAEs as trained on) first, falls back to hook_z
    (pre-W_O, per-head -- flattened across heads) only if hook_attn_out isn't
    present on this model. Stops with a clear error on any dimension mismatch
    rather than guessing a reshape.
    """
    if site == "resid_post":
        return f"blocks.{target_layer}.hook_resid_post", False

    if site != "attn_out":
        raise ValueError(f"Unknown site {site!r}")

    primary = f"blocks.{target_layer}.hook_attn_out"
    fallback = f"blocks.{target_layer}.attn.hook_z"

    d_in = getattr(getattr(sae, "cfg", None), "d_in", None)
    print(f"[hook] SAE expects d_in={d_in}", flush=True)

    probe_tokens = model.to_tokens("The capital of France is")
    with t.no_grad():
        _, probe_cache = model.run_with_cache(
            probe_tokens.to(device),
            names_filter=lambda n: n in (primary, fallback),
            return_type=None,
        )

    if primary in probe_cache:
        shape = tuple(probe_cache[primary].shape)
        print(f"[hook] using {primary!r} (post-W_O attention output), shape={shape}", flush=True)
        if d_in is not None and shape[-1] != d_in:
            print(
                f"\n[error] {primary!r} last dim {shape[-1]} != SAE d_in={d_in}.\n"
                f"        Refusing to guess a reshape. Full shape={shape}.\n"
                f"        Check Gemma Scope's attn_out documentation for the correct hook "
                f"point before retrying.\n",
                flush=True,
            )
            raise SystemExit(1)
        del probe_cache
        return primary, False

    if fallback in probe_cache:
        shape = tuple(probe_cache[fallback].shape)  # [..., n_heads, d_head]
        flat_dim = shape[-2] * shape[-1]
        print(
            f"[hook] {primary!r} not found on this model -- falling back to {fallback!r} "
            f"shape={shape} (n_heads={shape[-2]}, d_head={shape[-1]}, flattened_dim={flat_dim})",
            flush=True,
        )
        print(
            "[warn] hook_z is the PRE-W_O per-head value, not the post-W_O attention "
            "output the Gemma Scope paper documents attn_out SAEs as trained on. This "
            "fallback flattens heads only because flat_dim happens to match d_in -- that "
            "match does NOT prove it's the right activation. Treat a WARN/FAIL FVU under "
            "this fallback with real suspicion; a PASS is weak evidence at best.",
            flush=True,
        )
        if d_in is not None and flat_dim != d_in:
            print(
                f"\n[error] Flattened hook_z dim {flat_dim} != SAE d_in={d_in}. "
                f"shape={shape}. Refusing to guess further.\n",
                flush=True,
            )
            raise SystemExit(1)
        del probe_cache
        return fallback, True

    print(
        f"\n[error] Neither {primary!r} nor {fallback!r} exist on this model's hook "
        f"dictionary. Run model.run_with_cache once and inspect cache.keys() filtered "
        f"to 'blocks.{target_layer}.' to find the correct hook name.\n",
        flush=True,
    )
    raise SystemExit(1)


def _extract_final_token_activation(
    model: HookedTransformer,
    prompts: list[str],
    hook_name: str,
    needs_head_flatten: bool,
    device: str,
) -> t.Tensor:
    name_filter = lambda name: name == hook_name  # noqa: E731

    acts: list[t.Tensor] = []
    for i, prompt in enumerate(prompts):
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
        act = raw.detach().float().cpu()
        acts.append(act)
        del cache
        if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
            print(f"    [{i + 1}/{len(prompts)}] extracted", flush=True)

    return t.stack(acts, dim=0)  # [n_prompts, d_model]


def compute_fvu(sae: Any, activations: t.Tensor) -> dict[str, float]:
    """
    activations: [n_prompts, d_model], float32, on the SAE's device.
    Returns FVU plus supporting diagnostics.
    """
    with t.no_grad():
        features = sae.encode(activations)
        reconstructed = sae.decode(features)

        residual = activations - reconstructed
        fvu = (residual.var() / activations.var()).item()

        mean_act_norm = activations.norm(dim=-1).mean().item()
        mean_recon_norm = reconstructed.norm(dim=-1).mean().item()
        mean_l0 = (features > 0).float().sum(dim=-1).mean().item()

    return {
        "fvu": fvu,
        "mean_activation_norm": mean_act_norm,
        "mean_reconstruction_norm": mean_recon_norm,
        "mean_l0": mean_l0,
    }


def _fvu_verdict(fvu: float) -> str:
    if fvu < 0.10:
        return "PASS"
    if fvu < 0.20:
        return "WARN"
    return "FAIL"


def _print_results(
    model_key: str, target_layer: int, hook_name: str, n_prompts: int, metrics: dict[str, float], site: str
) -> None:
    verdict = _fvu_verdict(metrics["fvu"])
    print("=" * 60)
    print(f"Model: {model_key} | Layer: {target_layer} | Hook: {hook_name} | N prompts: {n_prompts}")
    print("=" * 60)
    print(f"FVU:                       {metrics['fvu']:.4f}  "
          f"[{verdict} — PASS < 0.10 | WARN 0.10-0.20 | FAIL > 0.20]")
    print(f"Mean activation norm:      {metrics['mean_activation_norm']:.2f}")
    print(f"Mean reconstruction norm:  {metrics['mean_reconstruction_norm']:.2f}")
    print(f"Mean L0 (active features): {metrics['mean_l0']:.1f}")
    print("=" * 60)
    if verdict == "FAIL" and site == "attn_out":
        print(
            "[note] FAIL on a first attn_out attempt is more likely a hook-point or "
            "release-string mismatch than a genuinely bad SAE -- Gemma Scope's res-all "
            "SAEs already passed at this same layer on this same battery. Re-check the "
            "[hook]/[load] lines above before concluding the SAE itself is low quality.",
            flush=True,
        )


def _resolve_device() -> str:
    if t.cuda.is_available():
        return "cuda"
    print("[warn] CUDA not available — falling back to CPU. This will be slow.", flush=True)
    return "cpu"


def run_model(
    site: str,
    model_key: str,
    config: dict[str, Any],
    battery_path_override: Path | None,
    layer_override: int | None,
    results_dir: Path,
    device: str,
) -> None:
    target_layer = layer_override if layer_override is not None else config["target_layer"]
    battery_path = battery_path_override or (REPO_ROOT / config["battery_path"])

    print(f"\n{'#' * 60}")
    print(f"# Running FVU spot check: {model_key} [site={site}]")
    print(f"{'#' * 60}\n", flush=True)

    battery = _load_battery(battery_path)
    prompts = [entry["clean_prompt"] for entry in battery]
    print(f"[data] {len(prompts)} clean prompts loaded from {battery_path}", flush=True)

    model = None
    sae = None
    try:
        model = _load_model(config["model_name"], config["tl_model_name"], device)
        sae, sae_release = _load_sae(config["sae_release_candidates"], config["sae_id"], device)

        hook_name, needs_head_flatten = _resolve_hook(site, model, sae, target_layer, device)

        print(f"[extract] pulling {hook_name!r} for {len(prompts)} prompts "
              f"(head_flatten={needs_head_flatten})", flush=True)
        try:
            clean_acts = _extract_final_token_activation(
                model, prompts, hook_name, needs_head_flatten, device
            )
        except t.cuda.OutOfMemoryError:
            print("\n[error] CUDA OOM during activation extraction.", flush=True)
            print(t.cuda.memory_summary(), flush=True)
            print(
                "Suggestion: run with --model to isolate one model at a time "
                "(models are already loaded one at a time by default).",
                flush=True,
            )
            raise SystemExit(1)

        clean_acts = clean_acts.to(device=device, dtype=t.float32)
        print(f"[extract] activations shape: {tuple(clean_acts.shape)}", flush=True)

        print("[fvu] computing FVU against SAE reconstruction", flush=True)
        metrics = compute_fvu(sae, clean_acts)
        _print_results(model_key, target_layer, hook_name, len(prompts), metrics, site)

        results_dir.mkdir(parents=True, exist_ok=True)
        suffix = "" if site == "resid_post" else f"_{site}" if not model_key.endswith(f"_{site}") else ""
        out_path = results_dir / f"fvu_results_{model_key}{suffix}.json"
        out_path.write_text(
            json.dumps(
                {
                    "model_key": model_key,
                    "site": site,
                    "model_name": config["model_name"],
                    "sae_release": sae_release,
                    "sae_id": config["sae_id"],
                    "target_layer": target_layer,
                    "hook_name": hook_name,
                    "needs_head_flatten": needs_head_flatten,
                    "battery_path": str(battery_path),
                    "n_prompts": len(prompts),
                    "verdict": _fvu_verdict(metrics["fvu"]),
                    **metrics,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"[save] wrote {out_path}", flush=True)

    finally:
        del model
        del sae
        gc.collect()
        if device == "cuda":
            t.cuda.empty_cache()
        print(f"[cleanup] freed model + SAE for {model_key}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="FVU spot check for Gemma Scope SAEs on the fact battery.")
    parser.add_argument(
        "--site",
        choices=list(SITE_CONFIGS.keys()),
        default="resid_post",
        help="Which SAE site to validate (default: resid_post).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Run a single model key for the selected --site. Omit to run all models for that site.",
    )
    parser.add_argument(
        "--battery_path",
        type=Path,
        default=None,
        help="Override the fact battery path (default: inferred from --model).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Override the target layer (default: inferred from --model).",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=REPO_ROOT / "results" / "fvu",
        help="Directory to write fvu_results_*.json files (default: results/fvu/).",
    )
    args = parser.parse_args()

    configs = SITE_CONFIGS[args.site]
    if args.model is not None and args.model not in configs:
        parser.error(
            f"--model {args.model!r} is not valid for --site {args.site!r}. "
            f"Valid keys: {list(configs.keys())}"
        )

    device = _resolve_device()
    models_to_run = [args.model] if args.model else list(configs.keys())

    for model_key in models_to_run:
        try:
            run_model(
                site=args.site,
                model_key=model_key,
                config=configs[model_key],
                battery_path_override=args.battery_path,
                layer_override=args.layer,
                results_dir=args.results_dir,
                device=device,
            )
        except FileNotFoundError as e:
            print(f"\n[error] {e}\n", flush=True)
            return 1
        except SystemExit:
            raise
        except Exception:
            print(f"\n[error] Unhandled failure while running {model_key}:", flush=True)
            traceback.print_exc()
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
