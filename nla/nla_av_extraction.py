#!/usr/bin/env python3
"""
AV extraction: for each clean/corrupt prompt pair in the fact battery, extract
resid_post at the NLA checkpoint's target layer (final token), inject it into
the AV (activation verbalizer) checkpoint's embeddings at the validated marker
position, and decode a natural-language description. Record
clean_description / corrupt_description per pair.

Plain HuggingFace `transformers` only -- no SGLang, no vendored client, no
subprocess/server lifecycle. The earlier version vendored kitft/nla-inference's
NLAClient wholesale, including its SGLang-backed serving path; that's built
for high-throughput concurrent serving, which we don't need for 114 sequential
one-off decodes on a single Colab GPU. What's actually worth keeping from that
reference code is just the injection procedure itself (tokenize -> find the
validated marker position -> rescale the raw activation to injection_scale ->
overwrite that embedding row -> generate) -- reimplemented directly below with
plain `transformers`, not copied.

Two sequential phases on one GPU (still necessary, SGLang or not): the base
model (TransformerLens, for extraction) and the AV checkpoint (transformers,
for decoding) are two different full-size model instances -- loading both at
once would double VRAM usage, which doesn't fit for 27B on a single A100
(bf16 weights alone ~54GB). Phase A extracts everything and fully frees the
base model before Phase B loads the AV checkpoint.

Checkpoints (confirmed in nla_config_probe.py's diagnostic run this session):
    12B: kitft/nla-gemma3-12b-L32-av, extraction model google/gemma-3-12b-it,
         layer 32, d_model 3840, injection_scale 80000.0
    27B: kitft/nla-gemma3-27b-L41-av, extraction model google/gemma-3-27b-it,
         layer 41, d_model 5376, injection_scale 60000.0

Position convention: final token (index -1) -- confirmed from the real
nla_inference.py source, the AR prompt template's fixed suffix, and the
README this session. NOT -2 (no primary source found for that).

KNOWN UNRESOLVED ISSUE from the last local smoke test: load_nla_config()'s
injection-marker validation (kept below, unchanged in spirit) failed against
the live 12B AV tokenizer -- the marker character tokenizes correctly in
isolation but disappears (0 matches) once embedded in the full chat-templated
prompt, meaning BPE merging is context-dependent here. That assertion is
still here, deliberately -- it will fail loudly and immediately (before any
GPU/generation work) rather than silently producing garbage, exactly like it
did locally. If it fires again on Colab, that confirms it's not a local
transformers-version artifact and needs resolving with kitft's checkpoint
before this can produce real output.

Install:
    pip install transformers accelerate pyyaml huggingface_hub

Usage:
    python nla_av_extraction.py --model gemma_12b_L32 --limit 3   # smoke test
    python nla_av_extraction.py --model gemma_12b_L32              # full battery
    python nla_av_extraction.py --model gemma_27b_L41

    # Sanity-check extraction alone, without touching the AV checkpoint at all:
    python nla_av_extraction.py --model gemma_12b_L32 --extract-only

Run on a single A100 80GB.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch as t
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from transformer_lens import HookedTransformer  # noqa: E402

from shared.fact_battery import load_fact_battery  # noqa: E402


EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "gemma_12b_L32": {
        "tl_model_name": "gemma-3-12b-it",
        "extraction_model": "google/gemma-3-12b-it",
        "target_layer": 32,
        "av_repo": "kitft/nla-gemma3-12b-L32-av",
        "battery_path": "fact_battery/gemma-3-12b-it.json",
        "expected_d_model": 3840,
        "expected_injection_scale": 80000.0,
    },
    "gemma_27b_L41": {
        "tl_model_name": "gemma-3-27b-it",
        "extraction_model": "google/gemma-3-27b-it",
        "target_layer": 41,
        "av_repo": "kitft/nla-gemma3-27b-L41-av",
        "battery_path": "fact_battery/gemma-3-27b-it.json",
        "expected_d_model": 5376,
        "expected_injection_scale": 60000.0,
    },
}


# ---------------------------------------------------------------------------
# Injection procedure (reimplemented directly, not vendored)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NLAConfig:
    d_model: int
    injection_char: str
    injection_token_id: int
    left_neighbor_id: int
    right_neighbor_id: int
    actor_prompt_template: str
    injection_scale: float


def _flat_token_ids(chat_template_output: Any) -> list[int]:
    """Normalize apply_chat_template(tokenize=True) to a flat list[int].

    transformers >=5 returns a dict-like BatchEncoding; older versions return
    a plain list. Both may be batched (nested one level). Getting this wrong
    is silent: enumerating a BatchEncoding yields key STRINGS, which never
    equal an int token id, so the marker search finds 0 matches and the
    validation below fails with a misleading "tokenizer/template drift".
    """
    out = chat_template_output
    if hasattr(out, "keys"):  # BatchEncoding / dict
        out = out["input_ids"]
    if hasattr(out, "tolist"):  # torch tensor / numpy
        out = out.tolist()
    out = list(out)
    if out and isinstance(out[0], (list, tuple)):  # batched
        out = list(out[0])
    return [int(x) for x in out]


def load_nla_config(av_repo: str, tokenizer: Any) -> tuple[NLAConfig, list[int], int]:
    """Fetch nla_meta.yaml and validate the injection marker against the live
    tokenizer, BEFORE loading model weights or generating anything. Catches
    tokenizer/template drift loudly instead of silently producing garbage.

    Returns (cfg, canonical_input_ids, injection_position) -- the tokenized
    canonical actor prompt plus the validated marker position, reused for
    every activation since only the injected row changes.
    """
    meta_path = hf_hub_download(repo_id=av_repo, filename="nla_meta.yaml")
    meta = yaml.safe_load(Path(meta_path).read_text())

    tokens = meta["tokens"]
    cfg = NLAConfig(
        d_model=meta["d_model"],
        injection_char=tokens["injection_char"],
        injection_token_id=tokens["injection_token_id"],
        left_neighbor_id=tokens["injection_left_neighbor_id"],
        right_neighbor_id=tokens["injection_right_neighbor_id"],
        actor_prompt_template=meta["prompt_templates"]["av"],
        injection_scale=float(meta["extraction"]["injection_scale"]),
    )

    live = tokenizer.encode(cfg.injection_char, add_special_tokens=False)
    assert live == [cfg.injection_token_id], (
        f"tokenizer drift: {cfg.injection_char!r} -> {live}, sidecar says "
        f"[{cfg.injection_token_id}]. Stopping -- injecting at the wrong "
        f"position produces silent garbage, not an error."
    )

    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    input_ids = _flat_token_ids(tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
    ))
    matches = [i for i, tok_id in enumerate(input_ids) if tok_id == cfg.injection_token_id]
    assert len(matches) == 1, (
        f"injection token appears {len(matches)}x in the templated prompt "
        f"(expected exactly 1) -- tokenizer/template drift. Template: {content!r}"
    )
    p = matches[0]
    assert 0 < p < len(input_ids) - 1, f"injection position {p} is at a sequence boundary"
    assert input_ids[p - 1] == cfg.left_neighbor_id and input_ids[p + 1] == cfg.right_neighbor_id, (
        f"neighbor drift at position {p}: got ({input_ids[p - 1]}, {input_ids[p + 1]}), "
        f"sidecar expects ({cfg.left_neighbor_id}, {cfg.right_neighbor_id})"
    )

    return cfg, input_ids, p


def normalize_activation(v: t.Tensor, target_scale: float) -> t.Tensor:
    """Rescale to target_scale L2-norm. Zeros stay zero. Norm computed in fp32."""
    norm = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm / target_scale).to(v.dtype)


def decode_activation(
    model: Any,
    tokenizer: Any,
    cfg: NLAConfig,
    canonical_input_ids: list[int],
    injection_position: int,
    vector: t.Tensor,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Inject `vector` (raw, un-rescaled) at the validated marker position,
    generate, extract <explanation> tags."""
    ids_t = t.tensor(canonical_input_ids, dtype=t.long, device=model.device).unsqueeze(0)

    with t.no_grad():
        # model's own embedding module -- for Gemma this already applies the
        # sqrt(d_model) scale internally, no manual scale-factor bookkeeping needed.
        embeds = model.get_input_embeddings()(ids_t).clone()
        v_scaled = normalize_activation(vector.float(), cfg.injection_scale).to(model.device, embeds.dtype)
        embeds[0, injection_position] = v_scaled

        gen_kwargs: dict[str, Any] = dict(
            inputs_embeds=embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)  # greedy, reproducible

        out_ids = model.generate(**gen_kwargs)

    # generate() with inputs_embeds (no input_ids) returns only the newly
    # generated continuation -- no prompt slicing needed.
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    match = EXPLANATION_RE.search(text)
    return {
        "description": match.group(1).strip() if match else text.strip(),
        "explanation_found": match is not None,
        "raw_text": text,
    }


# ---------------------------------------------------------------------------
# Phase A: activation extraction (TransformerLens)
# ---------------------------------------------------------------------------

def extract_resid_post(
    tl_model_name: str,
    target_layer: int,
    battery: list[dict[str, str]],
    device: str,
) -> tuple[t.Tensor, t.Tensor]:
    """Final-token blocks.{L}.hook_resid_post for every clean/corrupt prompt.
    Returns (clean_acts, corrupt_acts), each [n_prompts, d_model] float32 CPU.
    Frees the model from GPU before returning.
    """
    print(f"[extract] loading HookedTransformer {tl_model_name!r} (bfloat16, {device})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(
        tl_model_name, device=device, dtype=t.bfloat16,
    )
    hook_name = f"blocks.{target_layer}.hook_resid_post"
    name_filter = lambda name: name == hook_name  # noqa: E731

    def _final_token(prompt: str) -> t.Tensor:
        tokens = model.to_tokens(prompt)
        with t.no_grad():
            _, cache = model.run_with_cache(tokens.to(device), names_filter=name_filter, return_type=None)
            v = cache[hook_name][0, -1].float().cpu()
        del cache
        return v

    clean_rows, corrupt_rows = [], []
    n = len(battery)
    for i, entry in enumerate(battery):
        clean_rows.append(_final_token(entry["clean_prompt"]))
        corrupt_rows.append(_final_token(entry["corrupt_prompt"]))
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"    [{i + 1}/{n}] extracted", flush=True)

    clean_acts = t.stack(clean_rows, dim=0)
    corrupt_acts = t.stack(corrupt_rows, dim=0)

    del model
    gc.collect()
    if device == "cuda":
        t.cuda.empty_cache()
    print(f"[extract] done, shapes clean={tuple(clean_acts.shape)} corrupt={tuple(corrupt_acts.shape)}. "
          f"Base model freed from GPU.", flush=True)
    return clean_acts, corrupt_acts


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_model(
    model_key: str,
    results_dir: Path,
    device: str,
    temperature: float,
    max_new_tokens: int,
    extract_only: bool,
    limit: int | None,
) -> None:
    config = MODEL_CONFIGS[model_key]
    battery_path = REPO_ROOT / config["battery_path"]

    print(f"\n{'#' * 60}")
    print(f"# NLA AV extraction: {model_key}")
    print(f"{'#' * 60}\n", flush=True)

    battery = load_fact_battery(battery_path)
    if limit is not None:
        battery = battery[:limit]
        print(f"[data] --limit {limit}: using first {len(battery)} pairs only", flush=True)
    print(f"[data] {len(battery)} prompt pairs loaded from {battery_path}", flush=True)

    # --- Phase A: extract ---
    clean_acts, corrupt_acts = extract_resid_post(
        config["tl_model_name"], config["target_layer"], battery, device
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    activations_path = results_dir / f"activations_{model_key}.pt"
    t.save({"clean_acts": clean_acts, "corrupt_acts": corrupt_acts}, activations_path)
    print(f"[extract] saved raw activations to {activations_path}", flush=True)

    if extract_only:
        print("[extract-only] stopping before AV decoding, as requested.", flush=True)
        return

    # --- Phase B: load AV checkpoint (plain transformers), validate, decode ---
    print(f"\n[av] loading tokenizer for {config['av_repo']}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config["av_repo"], trust_remote_code=True)

    cfg, canonical_input_ids, injection_position = load_nla_config(config["av_repo"], tokenizer)
    print(f"[av] nla_meta.yaml + tokenizer validation PASSED: d_model={cfg.d_model} "
          f"injection_scale={cfg.injection_scale} marker position={injection_position}", flush=True)

    assert cfg.d_model == config["expected_d_model"], (
        f"d_model={cfg.d_model} != expected {config['expected_d_model']} -- checkpoint drift, stopping."
    )
    assert cfg.injection_scale == config["expected_injection_scale"], (
        f"injection_scale={cfg.injection_scale} != expected {config['expected_injection_scale']} "
        f"-- checkpoint drift, stopping."
    )

    print(f"[av] loading model {config['av_repo']} (bfloat16, {device})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["av_repo"], dtype=t.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    print("[av] model ready", flush=True)

    records: list[dict[str, Any]] = []
    n = len(battery)
    for i, entry in enumerate(battery):
        print(f"[{i + 1}/{n}] clean={entry['clean_prompt']!r}  corrupt={entry['corrupt_prompt']!r}", flush=True)
        try:
            clean_result = decode_activation(
                model, tokenizer, cfg, canonical_input_ids, injection_position,
                clean_acts[i], max_new_tokens, temperature,
            )
            corrupt_result = decode_activation(
                model, tokenizer, cfg, canonical_input_ids, injection_position,
                corrupt_acts[i], max_new_tokens, temperature,
            )
        except Exception as e:
            print(f"    [skip] pair {i} failed: {type(e).__name__}: {e}", flush=True)
            continue

        records.append({
            "idx": i,
            "category": entry.get("category"),
            "clean_prompt": entry["clean_prompt"],
            "corrupt_prompt": entry["corrupt_prompt"],
            "clean_description": clean_result["description"],
            "corrupt_description": corrupt_result["description"],
            "clean_explanation_found": clean_result["explanation_found"],
            "corrupt_explanation_found": corrupt_result["explanation_found"],
            "clean_activation_norm": float(clean_acts[i].norm()),
            "corrupt_activation_norm": float(corrupt_acts[i].norm()),
        })
        if not clean_result["explanation_found"] or not corrupt_result["explanation_found"]:
            print(f"    [warn] missing <explanation> tag(s) for pair {i}", flush=True)

    n_missing = sum(
        1 for r in records if not r["clean_explanation_found"] or not r["corrupt_explanation_found"]
    )
    print(f"\n[decode] {len(records)}/{n} pairs succeeded, "
          f"{len(records) - n_missing}/{len(records)} got clean <explanation> tags on both sides", flush=True)

    out_path = results_dir / f"av_descriptions_{model_key}.json"
    out_path.write_text(json.dumps({
        "model_key": model_key,
        "extraction_model": config["extraction_model"],
        "av_repo": config["av_repo"],
        "target_layer": config["target_layer"],
        "d_model": cfg.d_model,
        "injection_scale": cfg.injection_scale,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "n_prompts": n,
        "n_succeeded": len(records),
        "n_missing_explanation_tags": n_missing,
        "records": records,
    }, indent=2) + "\n")
    print(f"[save] wrote {out_path}", flush=True)

    print("\n" + "=" * 60)
    print(f"SAMPLE (first {min(3, len(records))} of {len(records)})")
    print("=" * 60)
    for r in records[:3]:
        print(f"\n[{r['idx']}] {r['category']}")
        print(f"  clean   ({r['clean_prompt']!r}): {r['clean_description']}")
        print(f"  corrupt ({r['corrupt_prompt']!r}): {r['corrupt_description']}")

    del model
    gc.collect()
    if device == "cuda":
        t.cuda.empty_cache()
    print(f"\n[cleanup] freed AV model for {model_key}", flush=True)


def _resolve_device() -> str:
    if t.cuda.is_available():
        return "cuda"
    print("[warn] CUDA not available -- falling back to CPU. This will be very slow "
          "for a 12B/27B model.", flush=True)
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="NLA AV extraction over the fact battery.")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--results_dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 (default) = greedy, reproducible clean-vs-corrupt comparison.")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--extract-only", action="store_true",
                        help="Stop after Phase A (save raw activations); skip AV decoding entirely.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N pairs -- for a cheap smoke test before "
                             "the full battery.")
    args = parser.parse_args()

    device = _resolve_device()
    run_model(
        model_key=args.model,
        results_dir=args.results_dir,
        device=device,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        extract_only=args.extract_only,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
