from __future__ import annotations

import os
import argparse
import faulthandler
import importlib.util
import json
import signal
import sys
import traceback
from copy import copy
from functools import partial
from pathlib import Path
from typing import Any

import torch as t
from transformer_lens import HookedTransformer
from huggingface_hub import login


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Make stdout behave like a tty even when piped (e.g. through Colab's `!`),
# so prints show up immediately instead of sitting in an 8KB buffer.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover - Python < 3.7 fallback, not expected
    pass


def _install_sigint_diagnostics() -> None:
    """If the process receives SIGINT (e.g. an environment-issued interrupt
    on Colab), dump every thread's current stack to stderr before the default
    handler raises KeyboardInterrupt. This turns a silent `^C` into a
    pinpointed location of where execution actually was."""

    def _handler(signum, frame):
        print("\n[debug] SIGINT received - dumping stack trace before exit:", flush=True)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stderr.flush()
        signal.default_int_handler(signum, frame)

    signal.signal(signal.SIGINT, _handler)


_install_sigint_diagnostics()

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


def _load_local_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _ensure_hf_token() -> None:
    _load_local_env_file(SCRIPT_DIR / ".env")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token


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


# Controlled debug printing. Set by CLI `--verbose` flag.
VERBOSE = False


def _debug(message: str) -> None:
    if VERBOSE:
        print(f"[debug] {message}", flush=True)


def _subset_dataset(dataset: Any, max_prompts: int | None) -> Any:
    if max_prompts is None or max_prompts <= 0 or max_prompts >= len(dataset):
        return dataset

    subset = copy(dataset)
    subset.prompts = dataset.prompts[:max_prompts]
    subset.clean_toks = dataset.clean_toks[:max_prompts]
    subset.corrupt_toks = dataset.corrupt_toks[:max_prompts]
    subset.io_tokenIDs = dataset.io_tokenIDs[:max_prompts]
    subset.s_tokenIDs = dataset.s_tokenIDs[:max_prompts]
    subset.word_idx = {
        "entity": dataset.word_idx["entity"][:max_prompts],
        "end": dataset.word_idx["end"][:max_prompts],
    }

    groups_by_category: dict[str, list[int]] = {}
    for idx, prompt in enumerate(subset.prompts):
        groups_by_category.setdefault(prompt["category"], []).append(idx)
    subset.groups = list(groups_by_category.values())
    return subset


def main() -> int:
    try:
        _ensure_hf_token()

        parser = argparse.ArgumentParser(description="Run path-patching experiments for a selected model.")
        parser.add_argument("--model", required=True, help="Model key from configs/model.py")
        parser.add_argument(
            "--receiver-heads",
            nargs="*",
            default=[],
            help="Optional receiver heads for q path patching, formatted as layer:head (e.g. 8:6 8:10)",
        )
        parser.add_argument(
            "--max-prompts",
            type=int,
            default=None,
            help="Limit the fact battery to the first N prompts. Default (omitted) uses the entire battery.",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable debug logging",
        )
        parser.add_argument(
            "--export-json",
            action="store_true",
            help="Export result artifacts to JSON summaries after the run",
        )
        args = parser.parse_args()

        # Set module-level verbosity flag
        global VERBOSE
        VERBOSE = bool(args.verbose)

        _debug("parsed arguments")
        config_module = _load_model_config_module()
        config = config_module.get_config(args.model)
        _debug(f"loaded config for model={args.model}")

        results_dir = SCRIPT_DIR / config["results_dir"]
        results_dir.mkdir(parents=True, exist_ok=True)
        _debug(f"results_dir ready: {results_dir}")

        dtype = _resolve_dtype(config["dtype"])
        print(f"loading model {config['model_name']} on {config['device']} with dtype={dtype}", flush=True)
        if config["device"] == "cuda" and dtype is t.bfloat16:
            major, _ = t.cuda.get_device_capability()
            if major < 8:
                gpu_name = t.cuda.get_device_name()
                print(
                    f"[warn] GPU {gpu_name} (compute capability {major}.x) lacks native bfloat16 "
                    "tensor cores. bf16 ops fall back to slow paths and the first forward pass "
                    "can stall for a long time with no output. If the run gets interrupted "
                    "around model load, retry with PATH_PATCHING_DTYPE=float16.",
                    flush=True,
                )
        model = HookedTransformer.from_pretrained_no_processing(
            config["model_name"],
            device=config["device"],
            dtype=dtype,
        )
        _debug("model loaded")

        battery_path = REPO_ROOT / config["fact_battery_path"]
        _debug(f"loading fact battery from {battery_path}")
        dataset = FactualRecallDataset(battery_path, model.tokenizer)
        dataset = _subset_dataset(dataset, args.max_prompts)
        _debug(f"dataset ready: prompts={len(dataset)} clean_shape={tuple(dataset.clean_toks.shape)} corrupt_shape={tuple(dataset.corrupt_toks.shape)}")
        sample_count = min(5, len(dataset))
        _debug(f"dataset samples: io={dataset.io_tokenIDs[:sample_count].tolist()} s={dataset.s_tokenIDs[:sample_count].tolist()}")

        receiver_heads_q = _parse_receiver_heads(args.receiver_heads)
        _debug(f"receiver heads parsed: {receiver_heads_q}")
        if not receiver_heads_q:
            print("TODO: provide receiver heads for q path patching if you want a non-empty receiver-head result.")

        z_name_filter = lambda name: name.endswith("z")

        _debug("running clean forward cache")
        with t.no_grad():
            clean_logits, clean_cache = model.run_with_cache(
                dataset.clean_toks,
                names_filter=z_name_filter,
                return_type="logits",
            )
        _debug(f"clean cache done: logits_shape={tuple(clean_logits.shape)} cache_keys={len(clean_cache)}")

        _debug("running corrupt forward cache")
        with t.no_grad():
            corrupt_logits, corrupt_cache = model.run_with_cache(
                dataset.corrupt_toks,
                names_filter=z_name_filter,
                return_type="logits",
            )
        _debug(f"corrupt cache done: logits_shape={tuple(corrupt_logits.shape)} cache_keys={len(corrupt_cache)}")

        _debug("computing logit diffs")
        with t.no_grad():
            clean_ld = compute_logit_diff(clean_logits, dataset)
            corrupt_ld = compute_logit_diff(corrupt_logits, dataset)
            total_swing = clean_ld - corrupt_ld

        print(f"clean_ld: {clean_ld.item():.6f}")
        print(f"corrupt_ld: {corrupt_ld.item():.6f}")
        print(f"TotalSwing: {total_swing.item():.6f}")

        baseline_ordering_ok = bool(clean_ld > corrupt_ld)
        if not baseline_ordering_ok:
            print("[debug] warning: clean_ld <= corrupt_ld for this (smoke) subset; continuing anyway", flush=True)

        _debug("saving baseline metrics")
        _save_tensor(
            results_dir / "baseline_metrics.pt",
            {
                "clean_ld": clean_ld.detach().cpu(),
                "corrupt_ld": corrupt_ld.detach().cpu(),
                "total_swing": total_swing.detach().cpu(),
                "receiver_heads_q": receiver_heads_q,
                "config": config,
                "baseline_ordering_ok": baseline_ordering_ok,
            },
        )

        run_manifest = {
            "model": args.model,
            "prompt_count": len(dataset),
            "max_prompts": args.max_prompts,
            "results_dir": str(results_dir),
            "files": [
                "path_patch_final_resid.pt",
                "path_patch_heads_q.pt",
                "baseline_metrics.pt",
            ],
            "receiver_heads_q": receiver_heads_q,
            "baseline_ordering_ok": baseline_ordering_ok,
        }
        (results_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
        _debug("wrote run manifest")

        patching_metric = partial(
            factual_recall_metric,
            dataset=dataset,
            clean_ld=clean_ld,
            corrupt_ld=corrupt_ld,
        )

        model.reset_hooks(including_permanent=True)
        _debug("running head-to-final-resid path patch")
        with t.no_grad():
            path_patch_final_resid = get_path_patch_head_to_final_resid_post(
                model=model,
                dataset=dataset,
                patching_metric=patching_metric,
                clean_cache=clean_cache,
                corrupt_cache=corrupt_cache,
                checkpoint_path=results_dir / "path_patch_final_resid.pt",
            )
        _debug(f"head-to-final-resid done: shape={tuple(path_patch_final_resid.shape)}")
        _save_tensor(results_dir / "path_patch_final_resid.pt", path_patch_final_resid)

        model.reset_hooks(including_permanent=True)
        _debug("running head-to-head path patch")
        with t.no_grad():
            path_patch_heads_q = get_path_patch_head_to_heads(
                receiver_heads=receiver_heads_q,
                receiver_input="q",
                model=model,
                dataset=dataset,
                patching_metric=patching_metric,
                clean_cache=clean_cache,
                corrupt_cache=corrupt_cache,
                checkpoint_path=results_dir / "path_patch_heads_q.pt",
            )
        _debug(f"head-to-head done: shape={tuple(path_patch_heads_q.shape)}")
        _save_tensor(results_dir / "path_patch_heads_q.pt", path_patch_heads_q)

        print(f"Experiment complete. Results saved to {results_dir}")

        # Optionally export JSON summaries of artifacts
        if args.export_json:
            try:
                # local import to avoid adding mandatory dependency for minimal runs
                from export_results import export_results  # type: ignore

                _debug("exporting results to JSON summaries")
                export_results(results_dir)
                print(f"Exported JSON summaries to {results_dir}")
            except Exception as e:
                print(f"Warning: failed to export JSON summaries: {e}")

        return 0
    except BaseException:
        # BaseException (not just Exception) so KeyboardInterrupt/SystemExit
        # from an external SIGINT also get a traceback printed instead of
        # vanishing silently.
        print("[debug] experiment failed/interrupted with exception", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    raise SystemExit(main())