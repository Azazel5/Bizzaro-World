from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch as t
from transformer_lens import HookedTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset import FactualRecallDataset  # noqa: E402
from metrics import compute_logit_diff, factual_recall_metric  # noqa: E402
from patching import get_path_patch_head_to_final_resid_post, get_path_patch_head_to_heads  # noqa: E402


def _load_model_config_module():
    config_path = SCRIPT_DIR / "configs" / "model.py"
    spec = importlib.util.spec_from_file_location("path_patching_configs_model", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_dtype(dtype_value: Any):
    if isinstance(dtype_value, str):
        resolved = getattr(t, dtype_value, None)
        if resolved is None:
            raise ValueError(f"Unknown torch dtype string: {dtype_value}")
        return resolved
    return dtype_value


def _parse_receiver_heads(raw_heads: list[str]) -> list[tuple[int, int]]:
    parsed: list[tuple[int, int]] = []
    for item in raw_heads:
        if not item:
            continue
        normalized = item.replace(",", ":")
        parts = normalized.split(":")
        if len(parts) != 2:
            raise ValueError(f"Receiver head must look like layer:head, got {item!r}")
        parsed.append((int(parts[0]), int(parts[1])))
    return parsed


def _save_tensor(path: Path, value: Any) -> None:
    if isinstance(value, t.Tensor):
        payload = value.detach().cpu()
    else:
        payload = value
    t.save(payload, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run path-patching experiments for a selected model.")
    parser.add_argument("--model", required=True, help="Model key from configs/model.py")
    parser.add_argument(
        "--receiver-heads",
        nargs="*",
        default=[],
        help="Optional receiver heads for q path patching, formatted as layer:head (e.g. 8:6 8:10)",
    )
    args = parser.parse_args()

    config_module = _load_model_config_module()
    config = config_module.get_config(args.model)

    results_dir = SCRIPT_DIR / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    dtype = _resolve_dtype(config["dtype"])
    model = HookedTransformer.from_pretrained(
        config["model_name"],
        device=config["device"],
        dtype=dtype,
    )

    battery_path = SCRIPT_DIR / config["fact_battery_path"]
    dataset = FactualRecallDataset(battery_path, model.tokenizer)

    print(f"Prompt count: {len(dataset)}")
    print(f"clean_toks shape: {tuple(dataset.clean_toks.shape)}")
    print(f"corrupt_toks shape: {tuple(dataset.corrupt_toks.shape)}")
    sample_count = min(5, len(dataset))
    print(f"io_tokenIDs sample: {dataset.io_tokenIDs[:sample_count].tolist()}")
    print(f"s_tokenIDs sample: {dataset.s_tokenIDs[:sample_count].tolist()}")

    receiver_heads_q = _parse_receiver_heads(args.receiver_heads)
    if not receiver_heads_q:
        print("TODO: provide receiver heads for q path patching if you want a non-empty receiver-head result.")

    z_name_filter = lambda name: name.endswith("z")

    with t.no_grad():
        clean_logits, clean_cache = model.run_with_cache(
            dataset.clean_toks,
            names_filter=z_name_filter,
            return_type="logits",
        )
        corrupt_logits, corrupt_cache = model.run_with_cache(
            dataset.corrupt_toks,
            names_filter=z_name_filter,
            return_type="logits",
        )

        clean_ld = compute_logit_diff(clean_logits, dataset)
        corrupt_ld = compute_logit_diff(corrupt_logits, dataset)
        total_swing = clean_ld - corrupt_ld

    print(f"clean_ld: {clean_ld.item():.6f}")
    print(f"corrupt_ld: {corrupt_ld.item():.6f}")
    print(f"TotalSwing: {total_swing.item():.6f}")
    assert clean_ld > corrupt_ld, "Expected clean_ld to be greater than corrupt_ld"

    patching_metric = partial(
        factual_recall_metric,
        dataset=dataset,
        clean_ld=clean_ld,
        corrupt_ld=corrupt_ld,
    )

    model.reset_hooks(including_permanent=True)
    with t.no_grad():
        path_patch_final_resid = get_path_patch_head_to_final_resid_post(
            model=model,
            dataset=dataset,
            patching_metric=patching_metric,
            clean_cache=clean_cache,
            corrupt_cache=corrupt_cache,
        )
    _save_tensor(results_dir / "path_patch_final_resid.pt", path_patch_final_resid)

    model.reset_hooks(including_permanent=True)
    with t.no_grad():
        path_patch_heads_q = get_path_patch_head_to_heads(
            receiver_heads=receiver_heads_q,
            receiver_input="q",
            model=model,
            dataset=dataset,
            patching_metric=patching_metric,
            clean_cache=clean_cache,
            corrupt_cache=corrupt_cache,
        )
    _save_tensor(results_dir / "path_patch_heads_q.pt", path_patch_heads_q)

    _save_tensor(
        results_dir / "baseline_metrics.pt",
        {
            "clean_ld": clean_ld.detach().cpu(),
            "corrupt_ld": corrupt_ld.detach().cpu(),
            "total_swing": total_swing.detach().cpu(),
            "receiver_heads_q": receiver_heads_q,
            "config": config,
        },
    )

    print(f"Experiment complete. Results saved to {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())