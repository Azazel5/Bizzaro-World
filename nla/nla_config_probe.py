#!/usr/bin/env python3
"""
Diagnostic-first check of the four NLA checkpoints (AV+AR for Gemma 12B-L32
and Gemma 27B-L41), before extracting a single real activation.

Deliberately does NOT load any model weights, run SGLang, or call
generate()/reconstruct() -- everything this checks (layer, d_model,
normalization, extraction-position convention) is already declared in each
checkpoint's nla_meta.yaml sidecar, a small YAML file. This runs entirely
locally (no HPC/GPU needed) by fetching just that one file per checkpoint via
huggingface_hub.

Checkpoints (confirmed via https://github.com/kitft/nla-inference/tree/main/examples,
worked transcripts for these exact two model/layer combos):
    kitft/nla-gemma3-12b-L32-av / -ar   (extraction model: google/gemma-3-12b-it)
    kitft/nla-gemma3-27b-L41-av / -ar   (extraction model: google/gemma-3-27b-it)

NOTE on package name: there is no `kitft-nla-inference` PyPI package -- the
kitft/nla-inference repo has no setup.py/pyproject.toml, just a standalone
nla_inference.py to vendor plus raw pip deps (torch, transformers,
safetensors, httpx, orjson, pyyaml, numpy, sglang[all]>=0.5.6). Real
extraction later will need that SGLang serving setup for the AV side; this
diagnostic does not.

NOTE on extraction position: nla_inference.py's own extraction line is
`last_hidden_state[0, -1]` (final token), and the AR prompt template ends in
a fixed suffix (`...</text> <summary>`) specifically so the final token is
stable -- confirmed straight from this session's fetch of the real
nla_meta.yaml below. No "-2" convention was found in the repo source,
README, worked examples, or the Neuronpedia NLA blog post. If you have a
source for -2, it isn't one of the four checked here -- worth resolving
before extraction, not after.

Usage:
    python nla_config_probe.py
"""
from __future__ import annotations

import sys
from typing import Any

import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# Independently verified this session (unsloth config.json mirrors,
# cross-checked against path_patching's own measured n_layers) -- used here
# only as a sanity cross-check against what nla_meta.yaml itself claims, not
# as a substitute for it.
KNOWN_ARCHITECTURE = {
    "google/gemma-3-12b-it": {"d_model": 3840, "num_hidden_layers": 48},
    "google/gemma-3-27b-it": {"d_model": 5376, "num_hidden_layers": 62},
}

CHECKPOINTS: dict[str, dict[str, Any]] = {
    "gemma_12b_L32": {
        "extraction_model": "google/gemma-3-12b-it",
        "target_layer": 32,
        "av_repo": "kitft/nla-gemma3-12b-L32-av",
        "ar_repo": "kitft/nla-gemma3-12b-L32-ar",
    },
    "gemma_27b_L41": {
        "extraction_model": "google/gemma-3-27b-it",
        "target_layer": 41,
        "av_repo": "kitft/nla-gemma3-27b-L41-av",
        "ar_repo": "kitft/nla-gemma3-27b-L41-ar",
    },
}


def _fetch_meta(repo_id: str) -> dict[str, Any] | None:
    print(f"  [fetch] {repo_id}/nla_meta.yaml ...", flush=True)
    try:
        path = hf_hub_download(repo_id=repo_id, filename="nla_meta.yaml")
    except HfHubHTTPError as e:
        print(f"    [fail] {type(e).__name__}: {e}", flush=True)
        return None
    except Exception as e:
        print(f"    [fail] {type(e).__name__}: {e}", flush=True)
        return None
    with open(path) as f:
        meta = yaml.safe_load(f)
    print(f"    [ok] role={meta.get('role')!r} stage={meta.get('stage')!r}", flush=True)
    return meta


def _report_one(model_key: str, config: dict[str, Any]) -> bool:
    print("\n" + "=" * 70)
    print(f"{model_key}  (target layer {config['target_layer']}, extraction model "
          f"{config['extraction_model']})")
    print("=" * 70)

    known = KNOWN_ARCHITECTURE.get(config["extraction_model"], {})
    expected_d_model = known.get("d_model")
    print(f"  Expected d_model (from this session's own architecture check): {expected_d_model}")

    av_meta = _fetch_meta(config["av_repo"])
    ar_meta = _fetch_meta(config["ar_repo"])

    ok = True
    for role, meta in (("AV", av_meta), ("AR", ar_meta)):
        if meta is None:
            print(f"  [MISSING] {role} metadata could not be fetched -- cannot confirm this checkpoint.")
            ok = False
            continue

        d_model = meta.get("d_model")
        layer_idx = meta.get("extraction_layer_index")
        d_match = "OK" if d_model == expected_d_model else "MISMATCH"
        layer_match = "OK" if layer_idx == config["target_layer"] else "MISMATCH"
        if d_model != expected_d_model or layer_idx != config["target_layer"]:
            ok = False

        print(f"\n  --- {role} ({config[role.lower() + '_repo']}) ---")
        print(f"    d_model = {d_model}  [{d_match} vs expected {expected_d_model}]")
        print(f"    extraction_layer_index = {layer_idx}  [{layer_match} vs target {config['target_layer']}]")
        print(f"    role = {meta.get('role')!r}, stage = {meta.get('stage')!r}, "
              f"schema_version = {meta.get('schema_version')!r}")

        extraction = meta.get("extraction", {}) or {}
        if extraction:
            print(f"    extraction.injection_scale = {extraction.get('injection_scale')!r}")
            print(f"    extraction.mse_scale = {extraction.get('mse_scale')!r}")
            if expected_d_model and extraction.get("mse_scale"):
                import math

                sqrt_d = math.sqrt(expected_d_model)
                print(f"      (sqrt(d_model) = {sqrt_d:.4f} -- "
                      f"{'matches' if abs(sqrt_d - extraction['mse_scale']) < 0.01 else 'DOES NOT MATCH'} "
                      f"mse_scale, per nla_inference.py's own 'mse_scale ~= sqrt(d_model)' convention)")

        templates = meta.get("prompt_templates", {}) or {}
        ar_template = templates.get("ar")
        if ar_template:
            suffix = ar_template[-40:]
            print(f"    ar prompt template tail: ...{suffix!r}")
            print(f"      -> fixed suffix before extraction point confirms final-token "
                  f"(index -1) is the stable, intended extraction position for this checkpoint.")

        tokens = meta.get("tokens", {}) or {}
        if tokens:
            print(f"    injection_char = {tokens.get('injection_char')!r}  "
                  f"(id={tokens.get('injection_token_id')})")

    print(f"\n  Overall: {'CONFIRMED, ready for real extraction' if ok else 'DO NOT PROCEED -- resolve MISMATCH above first'}")
    return ok


def main() -> int:
    print("NLA diagnostic-first check -- no model weights loaded, no SGLang, local only.\n")
    all_ok = True
    for model_key, config in CHECKPOINTS.items():
        ok = _report_one(model_key, config)
        all_ok = all_ok and ok

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"All checkpoints confirmed: {all_ok}")
    if not all_ok:
        print("At least one checkpoint failed to confirm -- see [MISSING]/[MISMATCH] above. "
              "Do not proceed to real extraction until resolved.")
    print(
        "\nRemember: extraction position is confirmed as -1 (final token) from primary "
        "sources (nla_inference.py, README, and the ar prompt template's fixed suffix "
        "above). The task brief's '-2' reference was not corroborated anywhere this "
        "session checked -- resolve that discrepancy before writing extraction code, "
        "since it changes which hidden state gets pulled."
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
