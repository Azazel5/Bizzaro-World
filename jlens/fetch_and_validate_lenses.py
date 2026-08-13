#!/usr/bin/env python3
"""
Fetch pretrained Jacobian lenses for Gemma 12B-IT / 27B-IT and validate each
against the real model's own output before trusting it for anything downstream.

NOT a from-scratch fit. The original plan for this file was to fit lenses
ourselves via jlens.fit() (a multi-hour, full-backward-pass-per-prompt job per
model) because the task brief claimed no pretrained Gemma lens existed. That
claim was wrong -- neuronpedia/jacobian-lens on HF hosts lenses for the entire
Gemma family, including the exact -it checkpoints this project uses:
    gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt
    gemma-3-27b-it/jlens/Salesforce-wikitext/gemma-3-27b-it_jacobian_lens.pt
Confirmed via each lens's own config.yaml: hf_model_name matches our models
exactly, dtype=bfloat16 (matches this project's convention throughout), fit
on 844/828 converged wikitext-103 prompts (Salesforce/wikitext) -- more
thorough than a from-scratch ~150-200-prompt fit would have been, and
attributed to the real anthropics/jacobian-lens ("Verbalizable Workspace")
code. Reusing these is strictly better than refitting: same method, same
license (Apache-2.0), no multi-hour GPU job, no corpus-gathering step (the
task brief's claim that a reusable "Scientist AI" Wikipedia corpus already
existed in this repo was also checked and is false -- moot now anyway, since
nothing needs fitting).

--- Position convention: -1 vs -2, resolved empirically, not assumed ---
The task brief asserted `positions=[-1]` "matching the convention already
confirmed for the NLA work this session." That's not evidence for jlens
specifically -- NLA and jlens are unrelated checkpoints with no shared
convention. Checked the real jlens source (jlens/fitting.py, fetched this
session): jlens's own README usage example explicitly uses `positions=[-2]`,
and fitting.py's valid_position_mask() explicitly EXCLUDES the final sequence
position from the Jacobian-averaging statistics ("the final position has no
next-token target" -- true for arbitrary corpus slices used at fit time,
where there's no known continuation past the slice boundary). Our own
fact-battery prompts DO have a well-defined completion at the final position
(that's the whole point of this project's prompt design), so -1 remains
architecturally plausible for us even though it was excluded from fitting
statistics -- but given a real, confirmed discrepancy from a task assumption,
this shouldn't be assumed either way. This script tests BOTH -1 and -2 in the
validation step below and picks whichever empirically agrees better with the
real model's own output, recording the choice and the numbers for both so
apply_jacobian_lens.py doesn't have to guess.

Validation gate: for each model, sample held-out prompts, apply the lens at
its outermost source layer (closest to the true final layer -- source_layers
are strictly < target_layer, so there is no lens fitted AT the final layer
itself) at both candidate positions, and compare against the model's real
final-layer logits (top-1 agreement, mean KL divergence). This should be
near-identical at that layer by construction (see jlens's own module
docstring: lens_l(h) = unembed(J_l @ h), and J_l for the layer immediately
below target is estimated from a near-identity transport). If it isn't,
something is wrong with our end-to-end usage and apply_jacobian_lens.py must
not run against this lens.

Install (Phase 1 -- run once on the HPC box, not from this script):
    git clone https://github.com/anthropics/jacobian-lens
    cd jacobian-lens && pip install -e .

Usage:
    python fetch_and_validate_lenses.py --model gemma_12b
    python fetch_and_validate_lenses.py --model gemma_27b

Run on a single A100 80GB. One model loaded at a time.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch as t
import torch.nn.functional as F
from huggingface_hub import scan_cache_dir
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jlens  # noqa: E402  (from Phase 1's pip install -e .)
from jlens.lens import JacobianLens  # noqa: E402  (direct submodule import -- see
# the AttributeError this session hit on `jlens.JacobianLens`: the real
# jlens/lens.py definitely defines this class (confirmed by reading the file
# directly), so importing it from there sidesteps whatever is/isn't correctly
# re-exported at the jlens/__init__.py top level, and sidesteps any sys.path
# ordering issue from this project's own "jlens" folder name colliding with
# the installed package's name.

from shared.fact_battery import load_fact_battery  # noqa: E402


LENS_REPO = "neuronpedia/jacobian-lens"

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "gemma_12b": {
        "hf_name": "google/gemma-3-12b-it",
        "lens_filename": "gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt",
        "causal_layer": 38,  # from path_patching's circuit_summary.json, for the plot's reference line
        "battery_path": "fact_battery/gemma-3-12b-it.json",
    },
    "gemma_27b": {
        "hf_name": "google/gemma-3-27b-it",
        "lens_filename": "gemma-3-27b-it/jlens/Salesforce-wikitext/gemma-3-27b-it_jacobian_lens.pt",
        "causal_layer": 54,
        "battery_path": "fact_battery/gemma-3-27b-it.json",
    },
}

CANDIDATE_POSITIONS = [-1, -2]
N_VALIDATION_PROMPTS = 8
MIN_TOP1_AGREEMENT = 0.75  # below this, refuse to proceed regardless of position


# ---------------------------------------------------------------------------
# Disk diagnostics (same pattern as sae/nla scripts this session)
# ---------------------------------------------------------------------------

def print_disk_usage(label: str) -> None:
    try:
        cache_info = scan_cache_dir()
    except Exception as e:
        print(f"[disk:{label}] could not scan HF cache: {type(e).__name__}: {e}", flush=True)
        return
    total = sum(repo.size_on_disk for repo in cache_info.repos)
    print(f"[disk:{label}] HF cache total: {total / 1e9:.1f} GB across {len(cache_info.repos)} repos:", flush=True)
    for repo in sorted(cache_info.repos, key=lambda r: -r.size_on_disk):
        print(f"    {repo.repo_id}: {repo.size_on_disk / 1e9:.2f} GB", flush=True)


def cleanup_hf_cache(repo_id: str) -> None:
    try:
        cache_info = scan_cache_dir()
    except Exception as e:
        print(f"[cleanup] could not scan HF cache: {type(e).__name__}: {e}", flush=True)
        return
    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            size_gb = repo.size_on_disk / 1e9
            shutil.rmtree(repo.repo_path, ignore_errors=True)
            print(f"[cleanup] removed {repo.repo_id} cache ({size_gb:.1f} GB freed)", flush=True)
            return
    print(f"[cleanup] {repo_id} not in HF cache -- nothing to remove", flush=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_position(
    lens: Any, model: Any, prompts: list[str], position: int, probe_layer: int
) -> dict[str, float]:
    """Apply the lens at `position` for the given probe_layer (the outermost
    fitted source layer) across `prompts`, compare against the model's real
    output at the same position. Returns top1 agreement rate and mean KL."""
    top1_matches = 0
    kls = []
    for prompt in prompts:
        lens_logits, model_logits, _ = lens.apply(model, prompt, positions=[position], layers=[probe_layer])
        lens_l = lens_logits[probe_layer][0].float()  # [vocab]
        model_l = model_logits[0].float()  # [vocab]

        if lens_l.argmax().item() == model_l.argmax().item():
            top1_matches += 1

        lens_logp = F.log_softmax(lens_l, dim=-1)
        model_p = F.softmax(model_l, dim=-1)
        kl = F.kl_div(lens_logp, model_p, reduction="sum").item()
        kls.append(kl)

    return {
        "top1_agreement": top1_matches / len(prompts),
        "mean_kl": sum(kls) / len(kls),
    }


def run_model(model_key: str, results_dir: Path, device: str) -> None:
    config = MODEL_CONFIGS[model_key]
    print(f"\n{'#' * 60}")
    print(f"# Fetch + validate Jacobian lens: {model_key}")
    print(f"{'#' * 60}\n", flush=True)
    print_disk_usage("start")

    print(f"[fetch] {LENS_REPO} :: {config['lens_filename']}", flush=True)
    lens = JacobianLens.from_pretrained(LENS_REPO, filename=config["lens_filename"])
    probe_layer = max(lens.jacobians.keys())
    print(f"[fetch] lens ready. Fitted source layers: {min(lens.jacobians.keys())}..{probe_layer} "
          f"({len(lens.jacobians)} layers), n_prompts={lens.n_prompts}, d_model={lens.d_model}", flush=True)
    print(f"[fetch] validation will probe the outermost source layer, L{probe_layer} "
          f"(closest to the true final layer; jlens never fits a lens AT the final layer itself)", flush=True)

    print(f"\n[load] {config['hf_name']} via plain HF transformers (bfloat16, {device})", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(config["hf_name"], torch_dtype=t.bfloat16).to(device)
    tok = AutoTokenizer.from_pretrained(config["hf_name"])
    model = jlens.from_hf(hf, tok)
    print("[load] model ready", flush=True)
    print_disk_usage("after model load")

    battery = load_fact_battery(REPO_ROOT / config["battery_path"])
    # Held-out validation prompts: plain declarative sentences distinct from
    # the fact-battery's own clean/corrupt prompts, so this checks general
    # lens fidelity, not something specific to the fact battery.
    validation_prompts = [
        "The history of the Roman Empire spans many centuries of conquest and administration.",
        "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        "The stock market experienced significant volatility during the economic downturn.",
        "Modern architecture often emphasizes clean lines and functional design principles.",
        "The novel explores themes of identity, memory, and belonging in a changing world.",
        "Climate scientists have documented rising global temperatures over the past century.",
        "The orchestra performed a symphony composed in the early nineteenth century.",
        "Software engineers use version control systems to track changes in their code.",
    ][:N_VALIDATION_PROMPTS]
    print(f"\n[validate] {len(validation_prompts)} held-out prompts (not fact-battery, not fitting corpus)", flush=True)

    print(f"\n[validate] testing candidate positions {CANDIDATE_POSITIONS} at layer L{probe_layer}...", flush=True)
    position_results: dict[int, dict[str, float]] = {}
    for position in CANDIDATE_POSITIONS:
        res = validate_position(lens, model, validation_prompts, position, probe_layer)
        position_results[position] = res
        print(f"    position={position:>3}: top1_agreement={res['top1_agreement']:.2%}  "
              f"mean_KL={res['mean_kl']:.4f}", flush=True)

    chosen_position = min(position_results, key=lambda p: position_results[p]["mean_kl"])
    chosen = position_results[chosen_position]
    print(f"\n[validate] chosen position: {chosen_position} "
          f"(lower mean KL: {chosen['mean_kl']:.4f} vs "
          f"{position_results[[p for p in CANDIDATE_POSITIONS if p != chosen_position][0]]['mean_kl']:.4f})", flush=True)

    passed = chosen["top1_agreement"] >= MIN_TOP1_AGREEMENT
    print(f"\n{'=' * 60}")
    print(f"Lens final-layer agreement with real model output "
          f"(position={chosen_position}, layer=L{probe_layer}):")
    print(f"  top-1 match: {chosen['top1_agreement']:.2%}")
    print(f"  mean KL divergence: {chosen['mean_kl']:.4f}")
    print(f"  gate ({MIN_TOP1_AGREEMENT:.0%} min top-1 agreement): {'PASS' if passed else 'FAIL'}")
    print(f"{'=' * 60}\n", flush=True)

    results_dir.mkdir(parents=True, exist_ok=True)
    meta_path = results_dir / f"{model_key}_lens_meta.json"
    meta_path.write_text(json.dumps({
        "model_key": model_key,
        "hf_name": config["hf_name"],
        "lens_repo": LENS_REPO,
        "lens_filename": config["lens_filename"],
        "probe_layer": probe_layer,
        "n_lens_prompts": lens.n_prompts,
        "d_model": lens.d_model,
        "source_layers": sorted(lens.jacobians.keys()),
        "causal_layer": config["causal_layer"],
        "chosen_position": chosen_position,
        "position_validation": {str(p): r for p, r in position_results.items()},
        "validation_passed": passed,
    }, indent=2) + "\n")
    print(f"[save] wrote {meta_path}", flush=True)

    if not passed:
        print(f"\n[STOP] validation failed for {model_key} -- top-1 agreement "
              f"{chosen['top1_agreement']:.2%} is below the {MIN_TOP1_AGREEMENT:.0%} gate at "
              f"BOTH candidate positions. Do NOT run apply_jacobian_lens.py for this model "
              f"until this is understood -- results would not be trustworthy.", flush=True)

    del hf, model, lens
    gc.collect()
    if device == "cuda":
        t.cuda.empty_cache()
    cleanup_hf_cache(config["hf_name"])
    print_disk_usage("end of run")

    if not passed:
        raise SystemExit(1)


def _resolve_device() -> str:
    if t.cuda.is_available():
        return "cuda"
    print("[warn] CUDA not available -- falling back to CPU. Will be slow for 12B/27B.", flush=True)
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch + validate a pretrained Jacobian lens.")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--results_dir", type=Path, default=SCRIPT_DIR / "results")
    args = parser.parse_args()

    device = _resolve_device()
    run_model(args.model, args.results_dir, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
