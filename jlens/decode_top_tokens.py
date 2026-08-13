#!/usr/bin/env python3
"""
Extension of apply_jacobian_lens.py: instead of collapsing each layer's
lens_logits down to the rank of one known target token, decode and save the
top-5 tokens at every layer, for every fact-battery prompt (clean and
corrupt). Answers "what does the model's own linear readout think comes
next at each depth" -- a coarse, logit-lens-style qualitative trace, as
opposed to apply_jacobian_lens.py's purely numeric rank-of-known-answer
metric, which structurally can't show what OTHER candidates the model is
considering along the way.

This is deliberately a separate script, not a modification of
apply_jacobian_lens.py -- that script works as-is and is left alone. Same
validated-lens loading, position handling, and memory discipline, copied
from it rather than imported as shared logic, matching this project's
established convention of independently-runnable pipeline scripts.

Requires fetch_and_validate_lenses.py to have already produced and PASSED
validation for this model, exactly like apply_jacobian_lens.py.

Usage:
    python decode_top_tokens.py --model gemma_12b
    python decode_top_tokens.py --model gemma_27b

Run on a single A100 80GB. One model loaded at a time. Same cost as
apply_jacobian_lens.py (same apply() calls, same layer sweep) -- this just
decodes more from the same already-computed per-layer distributions.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Import directly from submodules, not jlens/__init__.py -- see
# fetch_and_validate_lenses.py's comment on this same pattern for why.
from jlens.lens import JacobianLens  # noqa: E402
from jlens.hf import from_hf  # noqa: E402

from shared.fact_battery import load_fact_battery  # noqa: E402

from fetch_and_validate_lenses import (  # noqa: E402
    MODEL_CONFIGS,
    LENS_REPO,
    print_disk_usage,
    cleanup_hf_cache,
)

TOP_K = 5


def top_k_tokens(logits: t.Tensor, tok: Any, k: int = TOP_K) -> list[str]:
    """Decode the k highest-logit tokens, highest first."""
    top_ids = logits.topk(k).indices.tolist()
    return [tok.decode([tid]) for tid in top_ids]


def run_model(model_key: str, results_dir: Path, device: str) -> None:
    config = MODEL_CONFIGS[model_key]

    print(f"\n{'#' * 60}")
    print(f"# Decode top-{TOP_K} tokens per layer: {model_key}")
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
            f"{meta_path} shows validation_passed=False for {model_key} -- refusing to run "
            f"against an unvalidated lens."
        )
    position = meta["chosen_position"]
    print(f"[load] validation metadata OK: position={position}", flush=True)

    battery = load_fact_battery(REPO_ROOT / config["battery_path"])
    print(f"[data] {len(battery)} prompt pairs loaded from {config['battery_path']}", flush=True)

    print(f"\n[fetch] {LENS_REPO} :: {config['lens_filename']}", flush=True)
    lens = JacobianLens.from_pretrained(LENS_REPO, filename=config["lens_filename"])
    layers = sorted(lens.jacobians.keys())
    print(f"[fetch] lens ready, {len(layers)} fitted layers: L{layers[0]}..L{layers[-1]}", flush=True)

    print(f"\n[load] {config['hf_name']} via plain HF transformers (bfloat16, {device})", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(config["hf_name"], torch_dtype=t.bfloat16).to(device)
    tok = AutoTokenizer.from_pretrained(config["hf_name"])
    model = from_hf(hf, tok)
    print("[load] model ready", flush=True)
    print_disk_usage("after model load")

    records: list[dict[str, Any]] = []
    n = len(battery)
    for i, entry in enumerate(battery):
        print(f"[{i + 1}/{n}] clean={entry['clean_prompt']!r}  corrupt={entry['corrupt_prompt']!r}", flush=True)

        clean_lens_logits, _, _ = lens.apply(model, entry["clean_prompt"], positions=[position])
        corrupt_lens_logits, _, _ = lens.apply(model, entry["corrupt_prompt"], positions=[position])

        clean_top5_by_layer = [
            {"layer": layer, "top5_tokens": top_k_tokens(clean_lens_logits[layer][0].float(), tok)}
            for layer in layers
        ]
        corrupt_top5_by_layer = [
            {"layer": layer, "top5_tokens": top_k_tokens(corrupt_lens_logits[layer][0].float(), tok)}
            for layer in layers
        ]

        records.append({
            "idx": i,
            "category": entry.get("category"),
            "clean_prompt": entry["clean_prompt"],
            "corrupt_prompt": entry["corrupt_prompt"],
            "clean_top5_by_layer": clean_top5_by_layer,
            "corrupt_top5_by_layer": corrupt_top5_by_layer,
        })

        if i == 0:
            print(f"\n  [sample trace] {entry['clean_prompt']!r} (clean):", flush=True)
            for row in clean_top5_by_layer[::max(1, len(layers) // 8)]:
                print(f"    L{row['layer']:<3} {row['top5_tokens']}", flush=True)
            print(flush=True)

    print(f"\n[decode] {len(records)}/{n} pairs done", flush=True)

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"jlens_top_tokens_{model_key}.json"
    out_path.write_text(json.dumps({
        "model_key": model_key,
        "hf_name": config["hf_name"],
        "lens_filename": config["lens_filename"],
        "position": position,
        "layers": layers,
        "top_k": TOP_K,
        "n_prompts": len(records),
        "records": records,
    }, indent=2) + "\n")
    print(f"[save] wrote {out_path}", flush=True)

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
    parser = argparse.ArgumentParser(description="Decode top-k tokens per layer from a Jacobian lens.")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--results_dir", type=Path, default=SCRIPT_DIR / "results")
    args = parser.parse_args()

    device = _resolve_device()
    run_model(args.model, args.results_dir, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
