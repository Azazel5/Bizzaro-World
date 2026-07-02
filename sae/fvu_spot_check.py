#!/usr/bin/env python3
"""
FVU (Fraction of Variance Unexplained) spot check for Gemma Scope SAEs against
the BizzaroWorld fact battery.

For each model, extracts the final-token residual stream at a target layer for
every clean prompt in the battery, reconstructs it through the corresponding
Gemma Scope SAE, and reports how much variance the SAE fails to explain. This
is a sanity check that the SAE is a faithful lens on the model's factual-recall
representations before using it for further circuit analysis.

Usage:
    python fvu_spot_check.py                    # run all three models
    python fvu_spot_check.py --model gemma_12b  # run just one

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


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "gemma_12b": {
        "model_name": "google/gemma-3-12b-it",
        "tl_model_name": "gemma-3-12b-it",
        "sae_release": "gemma-scope-2-12b-it-resid_post",
        "sae_id": "layer_38_width_16k_l0_medium",
        "target_layer": 38,
        "battery_path": "fact_battery/gemma-3-12b-it.json",
    },
    "gemma_27b": {
        "model_name": "google/gemma-3-27b-it",
        "tl_model_name": "gemma-3-27b-it",
        "sae_release": "gemma-scope-2-27b-it-resid_post",
        "sae_id": "layer_54_width_16k_l0_medium",
        "target_layer": 54,
        "battery_path": "fact_battery/gemma-3-27b-it.json",
    },
}

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


def _extract_final_token_resid(
    model: HookedTransformer,
    prompts: list[str],
    target_layer: int,
    device: str,
) -> t.Tensor:
    hook_name = f"blocks.{target_layer}.hook_resid_post"
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
        resid = cache[hook_name][0, -1, :].detach().float().cpu()
        acts.append(resid)
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


def _print_results(model_key: str, target_layer: int, n_prompts: int, metrics: dict[str, float]) -> None:
    verdict = _fvu_verdict(metrics["fvu"])
    print("=" * 60)
    print(f"Model: {model_key} | Layer: {target_layer} | N prompts: {n_prompts}")
    print("=" * 60)
    print(f"FVU:                       {metrics['fvu']:.4f}  "
          f"[{verdict} — PASS < 0.10 | WARN 0.10-0.20 | FAIL > 0.20]")
    print(f"Mean activation norm:      {metrics['mean_activation_norm']:.2f}")
    print(f"Mean reconstruction norm:  {metrics['mean_reconstruction_norm']:.2f}")
    print(f"Mean L0 (active features): {metrics['mean_l0']:.1f}")
    print("=" * 60)


def _resolve_device() -> str:
    if t.cuda.is_available():
        return "cuda"
    print("[warn] CUDA not available — falling back to CPU. This will be slow.", flush=True)
    return "cpu"


def run_model(
    model_key: str,
    battery_path_override: Path | None,
    layer_override: int | None,
    results_dir: Path,
    device: str,
) -> None:
    config = MODEL_CONFIGS[model_key]
    target_layer = layer_override if layer_override is not None else config["target_layer"]
    battery_path = battery_path_override or (REPO_ROOT / config["battery_path"])

    print(f"\n{'#' * 60}")
    print(f"# Running FVU spot check: {model_key}")
    print(f"{'#' * 60}\n", flush=True)

    battery = _load_battery(battery_path)
    prompts = [entry["clean_prompt"] for entry in battery]
    print(f"[data] {len(prompts)} clean prompts loaded from {battery_path}", flush=True)

    model = None
    sae = None
    try:
        model = _load_model(config["model_name"], config["tl_model_name"], device)
        sae = _load_sae(config["sae_release"], config["sae_id"], device)

        print(f"[extract] pulling resid_post at layer {target_layer} for {len(prompts)} prompts", flush=True)
        try:
            clean_acts = _extract_final_token_resid(model, prompts, target_layer, device)
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
        _print_results(model_key, target_layer, len(prompts), metrics)

        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"fvu_results_{model_key}.json"
        out_path.write_text(
            json.dumps(
                {
                    "model_key": model_key,
                    "model_name": config["model_name"],
                    "sae_release": config["sae_release"],
                    "sae_id": config["sae_id"],
                    "target_layer": target_layer,
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
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default=None,
        help="Run a single model. Omit to run all three sequentially.",
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
        help="Directory to write fvu_results_{model}.json files (default: results/fvu/).",
    )
    args = parser.parse_args()

    device = _resolve_device()
    models_to_run = [args.model] if args.model else list(MODEL_CONFIGS.keys())

    for model_key in models_to_run:
        try:
            run_model(
                model_key=model_key,
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
