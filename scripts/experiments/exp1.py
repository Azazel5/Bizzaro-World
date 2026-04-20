#!/usr/bin/env python3
"""
Experiment 1 — layerwise residual-stream activation patching (TransformerLens).

Loads golden pairs from `fact_battery_triage.csv` using selection mode A, B, or C
(see `golden_pairs.select_golden_pairs`). For each pair, patches `hook_resid_pre`
at the final position from corrupt → clean per layer and records LD change.

Writes `experiment_{mode}.json` to --outdir (default: current working directory).
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


def _ld_at_final(logits_last: torch.Tensor, clean_id: int, corrupt_id: int) -> float:
    lf = logits_last.float()
    return (lf[clean_id] - lf[corrupt_id]).item()


def _final_logits(model: HookedTransformer, prompt: str) -> torch.Tensor:
    tokens = model.to_tokens(prompt, prepend_bos=True).to(model.cfg.device)
    with torch.no_grad():
        logits = model(tokens)
    return logits[0, -1, :]


def _resid_pre_hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_pre"


def _patch_layer_get_ld(
    model: HookedTransformer,
    layer: int,
    clean_prompt: str,
    corrupt_cache: Dict[str, torch.Tensor],
    clean_id: int,
    corrupt_id: int,
) -> float:
    hook_name = _resid_pre_hook_name(layer)
    corrupt_vec = corrupt_cache[hook_name][:, -1, :]

    def hook_fn(resid_pre: torch.Tensor, hook: Any) -> torch.Tensor:
        resid_pre[:, -1, :] = corrupt_vec.to(
            device=resid_pre.device, dtype=resid_pre.dtype
        )
        return resid_pre

    clean_toks = model.to_tokens(clean_prompt, prepend_bos=True).to(model.cfg.device)
    with torch.no_grad():
        logits = model.run_with_hooks(clean_toks, fwd_hooks=[(hook_name, hook_fn)])
    return _ld_at_final(logits[0, -1, :], clean_id, corrupt_id)


def run_experiment(
    model: HookedTransformer,
    pairs: List[GoldenPair],
    mode: SelectionMode,
) -> Dict[str, Any]:
    n_layers = model.cfg.n_layers
    names_filter: Callable[[str], bool] = lambda name: name.endswith("hook_resid_pre")

    out_pairs: List[Dict[str, Any]] = []

    for gp in pairs:
        lf_clean = _final_logits(model, gp.clean_prompt)
        lf_corrupt = _final_logits(model, gp.corrupt_prompt)
        baseline_ld_clean = _ld_at_final(lf_clean, gp.clean_target_id, gp.corrupt_target_id)
        baseline_ld_corrupt = _ld_at_final(
            lf_corrupt, gp.clean_target_id, gp.corrupt_target_id
        )

        corrupt_toks = model.to_tokens(gp.corrupt_prompt, prepend_bos=True).to(
            model.cfg.device
        )
        with torch.no_grad():
            _, corrupt_cache = model.run_with_cache(corrupt_toks, names_filter=names_filter)

        patched_lds: List[float] = []
        ld_deltas: List[float] = []

        for layer in range(n_layers):
            patched_ld = _patch_layer_get_ld(
                model,
                layer,
                gp.clean_prompt,
                corrupt_cache,
                gp.clean_target_id,
                gp.corrupt_target_id,
            )
            patched_lds.append(patched_ld)
            ld_deltas.append(patched_ld - baseline_ld_clean)

        worst_layer = min(range(n_layers), key=lambda i: ld_deltas[i])

        triage_fields = gp.as_dict()
        triage_fields.pop("ld_clean", None)
        triage_fields.pop("ld_corrupt", None)

        out_pairs.append(
            {
                **triage_fields,
                "baseline_ld_clean": baseline_ld_clean,
                "baseline_ld_corrupt": baseline_ld_corrupt,
                "patch": {
                    "site": "resid_pre",
                    "position": "final",
                    "hook_template": "blocks.{layer}.hook_resid_pre",
                },
                "patched_ld_by_layer": patched_lds,
                "ld_delta_vs_clean_baseline_by_layer": ld_deltas,
                "worst_layer_min_delta": int(worst_layer),
            }
        )

    return {
        "experiment": "activation_patching_exp1",
        "selection_mode": mode,
        "model": MODEL_NAME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_layers": n_layers,
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
        help="Directory to write JSON output (default: current directory).",
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
    out_path = outdir / f"experiment_{mode}.json"

    print(f"Loading {MODEL_NAME} …", flush=True)
    model = _load_model()
    if not torch.cuda.is_available():
        print(
            "Warning: CUDA not available; this run will be slow and may OOM in fp32.",
            flush=True,
        )

    print(
        f"Mode {mode}: {len(pairs)} pairs × {model.cfg.n_layers} layers …",
        flush=True,
    )
    payload = run_experiment(model, pairs, mode)

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()

