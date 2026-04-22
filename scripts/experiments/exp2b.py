#!/usr/bin/env python3
"""
Experiment 2B — Attention vs MLP decomposition at the entity token position (TransformerLens).

This is the entity-position analogue of Experiment 2A (final-position decomposition).
For each prompt pair we cache corrupt activations at five hook points across all
layers (0–17), then patch those cached vectors into the clean run at the entity
token position and measure damage to the clean logit margin.

Outputs (written to --outdir, default current working directory):
  - experiment2b_{MODE}.json
  - experiment2b_{MODE}.log
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
from transformer_lens import HookedTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from golden_pairs import GoldenPair, SelectionMode, select_golden_pairs  # noqa: E402

MODEL_NAME = "google/gemma-2b"
BATTERY_PATH = REPO_ROOT / "fact_battery.json"

TARGET_LAYERS = list(range(18))
TARGET_HOOKS = [
    "hook_resid_pre",
    "hook_attn_out",
    "hook_resid_mid",
    "hook_mlp_out",
    "hook_resid_post",
]


def _load_model() -> HookedTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
        dtype=dtype,
    )
    model.eval()
    return model


def _final_logits(model: HookedTransformer, prompt: str) -> torch.Tensor:
    tokens = model.to_tokens(prompt, prepend_bos=True).to(model.cfg.device)
    with torch.no_grad():
        logits = model(tokens)
    return logits[0, -1, :]


def _ld_at_final(logits_last: torch.Tensor, clean_id: int, corrupt_id: int) -> float:
    lf = logits_last.float()
    return (lf[clean_id] - lf[corrupt_id]).item()


def _hook_name(layer: int, hook: str) -> str:
    return f"blocks.{layer}.{hook}"


def _names_filter(n_layers: int) -> Callable[[str], bool]:
    layers = [L for L in TARGET_LAYERS if 0 <= L < n_layers]
    allowed = {_hook_name(L, h) for L in layers for h in TARGET_HOOKS}

    def filt(name: str) -> bool:
        return name in allowed

    return filt


def _load_entity_tokens_by_idx() -> Dict[int, str]:
    data = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("fact_battery.json must be a JSON array")
    out: Dict[int, str] = {}
    for i, e in enumerate(data):
        if not isinstance(e, dict):
            continue
        ent = str(e.get("entity_token", "")).strip()
        if not ent:
            raise ValueError(
                f"fact_battery.json entry {i} missing entity_token; run scripts/data_prep/add_entity_tokens.py"
            )
        out[i] = ent
    return out


def find_entity_position(model: HookedTransformer, prompt: str, entity: str) -> int:
    toks = model.to_tokens(prompt, prepend_bos=True)[0]
    token_strs: List[str] = [model.to_string(t.unsqueeze(0)) for t in toks]
    target = entity.lower()
    for i, tok in enumerate(token_strs):
        if target in tok.lower():
            return i
    raise ValueError(
        f"Entity {entity!r} not found in tokenized prompt.\n"
        f"prompt={prompt!r}\n"
        f"tokens={token_strs}"
    )


def _patch_one(
    model: HookedTransformer,
    clean_prompt: str,
    hook_name: str,
    corrupt_cache: Dict[str, torch.Tensor],
    *,
    entity_pos: int,
    clean_id: int,
    corrupt_id: int,
) -> float:
    corrupt_vec = corrupt_cache[hook_name][:, entity_pos, :]

    def hook_fn(x: torch.Tensor, hook: Any) -> torch.Tensor:
        x[:, entity_pos, :] = corrupt_vec.to(device=x.device, dtype=x.dtype)
        return x

    clean_toks = model.to_tokens(clean_prompt, prepend_bos=True).to(model.cfg.device)
    with torch.no_grad():
        logits = model.run_with_hooks(clean_toks, fwd_hooks=[(hook_name, hook_fn)])
    return _ld_at_final(logits[0, -1, :], clean_id, corrupt_id)


def run_experiment(
    model: HookedTransformer,
    pairs: List[GoldenPair],
    mode: SelectionMode,
    *,
    log_path: Path,
) -> Dict[str, Any]:
    n_layers = int(model.cfg.n_layers)
    layers = [L for L in TARGET_LAYERS if 0 <= L < n_layers]
    names_filter = _names_filter(n_layers)
    ent_by_idx = _load_entity_tokens_by_idx()

    out_pairs: List[Dict[str, Any]] = []

    with log_path.open("w", encoding="utf-8") as logf:

        def log(line: str = "") -> None:
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()

        for idx, gp in enumerate(pairs, start=1):
            entity_token = ent_by_idx.get(int(gp.battery_idx))
            if not entity_token:
                raise KeyError(f"No entity_token for battery_idx={gp.battery_idx}")
            entity_pos = find_entity_position(model, gp.clean_prompt, entity_token)

            lf_clean = _final_logits(model, gp.clean_prompt)
            lf_corrupt = _final_logits(model, gp.corrupt_prompt)
            baseline_ld_clean = _ld_at_final(
                lf_clean, gp.clean_target_id, gp.corrupt_target_id
            )
            baseline_ld_corrupt = _ld_at_final(
                lf_corrupt, gp.clean_target_id, gp.corrupt_target_id
            )

            corrupt_toks = model.to_tokens(gp.corrupt_prompt, prepend_bos=True).to(
                model.cfg.device
            )
            with torch.no_grad():
                _, corrupt_cache = model.run_with_cache(
                    corrupt_toks, names_filter=names_filter
                )

            results_by_layer: Dict[str, Dict[str, Dict[str, float]]] = {}
            worst_ld_delta = float("inf")
            worst_layer = None
            worst_hook = None

            log(
                f"[pair {idx}/{len(pairs)}] {gp.category} | rank={gp.rank} | entity={entity_token!r} pos={entity_pos}"
            )

            for L in layers:
                layer_key = str(L)
                results_by_layer[layer_key] = {}

                pieces: List[str] = []
                for H in TARGET_HOOKS:
                    full = _hook_name(L, H)
                    patched_ld = _patch_one(
                        model,
                        gp.clean_prompt,
                        full,
                        corrupt_cache,
                        entity_pos=entity_pos,
                        clean_id=gp.clean_target_id,
                        corrupt_id=gp.corrupt_target_id,
                    )
                    ld_delta = patched_ld - baseline_ld_clean
                    results_by_layer[layer_key][H] = {
                        "patched_ld": float(patched_ld),
                        "ld_delta": float(ld_delta),
                    }
                    pieces.append(f"{H.replace('hook_', '')}: delta={ld_delta:.2f}")

                    if ld_delta < worst_ld_delta:
                        worst_ld_delta = ld_delta
                        worst_layer = L
                        worst_hook = H

                log(f"  Layer {L} | " + " | ".join(pieces))

            log(
                f"  Worst: layer={worst_layer}, hook={worst_hook}, delta={worst_ld_delta:.2f}"
            )
            log("")

            triage_fields = gp.as_dict()
            triage_fields.pop("ld_clean", None)
            triage_fields.pop("ld_corrupt", None)

            out_pairs.append(
                {
                    **triage_fields,
                    "patch_position": "entity",
                    "entity_token": entity_token,
                    "entity_position": int(entity_pos),
                    "baseline_ld_clean": float(baseline_ld_clean),
                    "baseline_ld_corrupt": float(baseline_ld_corrupt),
                    "results_by_layer": results_by_layer,
                    "worst_hook": str(worst_hook),
                    "worst_layer": int(worst_layer) if worst_layer is not None else None,
                    "worst_ld_delta": float(worst_ld_delta),
                }
            )

    return {
        "experiment": "2b",
        "description": "attention vs MLP decomposition at entity token position, all 18 layers",
        "selection_mode": mode,
        "model": MODEL_NAME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_layers": n_layers,
        "target_layers": layers,
        "target_hooks": list(TARGET_HOOKS),
        "n_pairs": len(pairs),
        "pairs": out_pairs,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("A", "B", "C"), default="A")
    p.add_argument(
        "--triage-csv",
        type=Path,
        default=REPO_ROOT / "fact_battery_triage.csv",
        help="Path to fact_battery_triage.csv",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Directory to write JSON and log output (default: current directory).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    mode: SelectionMode = args.mode
    triage_path: Path = args.triage_csv
    if not triage_path.is_file():
        raise FileNotFoundError(
            f"Triage CSV not found: {triage_path}\n"
            "Copy fact_battery_triage.csv here or pass --triage-csv."
        )

    pairs = select_golden_pairs(triage_path, mode)
    if not pairs:
        raise RuntimeError("No pairs selected (empty triage?).")

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / f"experiment2b_{mode}.json"
    out_log = outdir / f"experiment2b_{mode}.log"

    print(f"Loading {MODEL_NAME} …", flush=True)
    model = _load_model()
    if not torch.cuda.is_available():
        print(
            "Warning: CUDA not available; this run will be slow and may OOM in fp32.",
            flush=True,
        )

    print(
        f"Mode {mode}: {len(pairs)} pairs | layers {list(range(int(model.cfg.n_layers)))} | hooks {len(TARGET_HOOKS)} | patch_position=entity …",
        flush=True,
    )
    payload = run_experiment(model, pairs, mode, log_path=out_log)

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_json.resolve()}", flush=True)
    print(f"Wrote {out_log.resolve()}", flush=True)


if __name__ == "__main__":
    main()

