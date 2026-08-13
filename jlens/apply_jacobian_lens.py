#!/usr/bin/env python3
"""
Apply a validated Jacobian lens to the fact battery: for each clean/corrupt
prompt pair, read out the lens's logits at every fitted layer (final token
position) and track the rank of the correct-answer token as a function of
depth.

Requires fetch_and_validate_lenses.py to have already produced and PASSED
validation for this model (results/{model_key}_lens_meta.json with
validation_passed=true) -- refuses to run otherwise, per this project's
established discipline of not proceeding past a failed correctness gate.

Position: read from the validation metadata (chosen_position), not
hardcoded -- see fetch_and_validate_lenses.py's docstring for why -1 vs -2
was resolved empirically rather than assumed.

Usage:
    python apply_jacobian_lens.py --model gemma_12b
    python apply_jacobian_lens.py --model gemma_27b

Run on a single A100 80GB. One model loaded at a time.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

import torch as t
from huggingface_hub import scan_cache_dir
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jlens  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402  (direct submodule import -- see
# fetch_and_validate_lenses.py's comment on the same import for why)

from shared.fact_battery import load_fact_battery  # noqa: E402

from fetch_and_validate_lenses import (  # noqa: E402
    MODEL_CONFIGS,
    LENS_REPO,
    print_disk_usage,
    cleanup_hf_cache,
)


def token_rank(logits: t.Tensor, target_id: int) -> int:
    """1-indexed rank of target_id in logits (descending). Rank 1 = top prediction."""
    return int((logits > logits[target_id]).sum().item()) + 1


def run_model(model_key: str, results_dir: Path, device: str) -> None:
    config = MODEL_CONFIGS[model_key]

    print(f"\n{'#' * 60}")
    print(f"# Apply Jacobian lens to fact battery: {model_key}")
    print(f"{'#' * 60}\n", flush=True)
    print_disk_usage("start")

    meta_path = results_dir / f"{model_key}_lens_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"missing {meta_path} -- run fetch_and_validate_lenses.py --model {model_key} first."
        )
    meta = json.loads(meta_path.read_text())
    if not meta.get("validation_passed"):
        raise RuntimeError(
            f"{meta_path} shows validation_passed=False for {model_key} -- refusing to apply "
            f"an unvalidated lens to the fact battery. Fix whatever failed in "
            f"fetch_and_validate_lenses.py first."
        )
    position = meta["chosen_position"]
    print(f"[load] validation metadata OK: position={position}, "
          f"top1_agreement={meta['position_validation'][str(position)]['top1_agreement']:.2%}", flush=True)

    battery = load_fact_battery(REPO_ROOT / config["battery_path"])
    print(f"[data] {len(battery)} prompt pairs loaded from {config['battery_path']}", flush=True)

    print(f"\n[fetch] {LENS_REPO} :: {config['lens_filename']}", flush=True)
    lens = JacobianLens.from_pretrained(LENS_REPO, filename=config["lens_filename"])
    layers = sorted(lens.jacobians.keys())
    print(f"[fetch] lens ready, {len(layers)} fitted layers: L{layers[0]}..L{layers[-1]}", flush=True)

    print(f"\n[load] {config['hf_name']} via plain HF transformers (bfloat16, {device})", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(config["hf_name"], torch_dtype=t.bfloat16).to(device)
    tok = AutoTokenizer.from_pretrained(config["hf_name"])
    model = jlens.from_hf(hf, tok)
    print("[load] model ready", flush=True)
    print_disk_usage("after model load")

    records: list[dict[str, Any]] = []
    n = len(battery)
    for i, entry in enumerate(battery):
        clean_target_id = tok.encode(entry["clean_target"], add_special_tokens=False)
        corrupt_target_id = tok.encode(entry["corrupt_target"], add_special_tokens=False)
        if len(clean_target_id) != 1 or len(corrupt_target_id) != 1:
            print(f"    [skip] pair {i}: target not single-token "
                  f"(clean={clean_target_id}, corrupt={corrupt_target_id})", flush=True)
            continue
        clean_target_id, corrupt_target_id = clean_target_id[0], corrupt_target_id[0]

        print(f"[{i + 1}/{n}] clean={entry['clean_prompt']!r}  corrupt={entry['corrupt_prompt']!r}", flush=True)

        clean_lens_logits, _, _ = lens.apply(model, entry["clean_prompt"], positions=[position])
        corrupt_lens_logits, _, _ = lens.apply(model, entry["corrupt_prompt"], positions=[position])

        clean_rank_by_layer = [
            token_rank(clean_lens_logits[layer][0].float(), clean_target_id) for layer in layers
        ]
        corrupt_rank_by_layer = [
            token_rank(corrupt_lens_logits[layer][0].float(), corrupt_target_id) for layer in layers
        ]

        records.append({
            "idx": i,
            "category": entry.get("category"),
            "clean_prompt": entry["clean_prompt"],
            "corrupt_prompt": entry["corrupt_prompt"],
            "correct_answer_token": entry["clean_target"],
            "corrupt_answer_token": entry["corrupt_target"],
            "clean_rank_by_layer": clean_rank_by_layer,
            "corrupt_rank_by_layer": corrupt_rank_by_layer,
        })

    print(f"\n[apply] {len(records)}/{n} pairs succeeded", flush=True)

    # Aggregate: mean rank per layer across all pairs, for the headline plot.
    mean_clean_rank_by_layer = [
        statistics.mean(r["clean_rank_by_layer"][li] for r in records) for li in range(len(layers))
    ]
    mean_corrupt_rank_by_layer = [
        statistics.mean(r["corrupt_rank_by_layer"][li] for r in records) for li in range(len(layers))
    ]

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"jlens_ranks_{model_key}.json"
    out_path.write_text(json.dumps({
        "model_key": model_key,
        "hf_name": config["hf_name"],
        "lens_filename": config["lens_filename"],
        "position": position,
        "layers": layers,
        "causal_layer": config["causal_layer"],
        "n_prompts": len(records),
        "records": records,
        "aggregate": {
            "mean_clean_rank_by_layer": mean_clean_rank_by_layer,
            "mean_corrupt_rank_by_layer": mean_corrupt_rank_by_layer,
        },
    }, indent=2) + "\n")
    print(f"[save] wrote {out_path}", flush=True)

    print("\n" + "=" * 60)
    print(f"{'layer':<8} {'mean clean rank':>18} {'mean corrupt rank':>20}")
    print("-" * 60)
    for li, layer in enumerate(layers):
        marker = "  <- causal_layer" if layer == config["causal_layer"] else ""
        print(f"L{layer:<7} {mean_clean_rank_by_layer[li]:>18.1f} {mean_corrupt_rank_by_layer[li]:>20.1f}{marker}")
    print("=" * 60, flush=True)

    del hf, model, lens
    gc.collect()
    if device == "cuda":
        t.cuda.empty_cache()
    cleanup_hf_cache(config["hf_name"])
    print_disk_usage("end of run")


def _resolve_device() -> str:
    if t.cuda.is_available():
        return "cuda"
    print("[warn] CUDA not available -- falling back to CPU. Will be slow for 12B/27B.", flush=True)
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a validated Jacobian lens to the fact battery.")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--results_dir", type=Path, default=SCRIPT_DIR / "results")
    args = parser.parse_args()

    device = _resolve_device()
    run_model(args.model, args.results_dir, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
