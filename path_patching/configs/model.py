from __future__ import annotations

import os

import torch as t


CONFIGS = {
    "gemma_2b": {
        "model_name": "google/gemma-2b",
        "fact_battery_path": "fact_battery/gemma-2b.json",
    },
    "gemma_12b": {
        "model_name": "google/gemma-3-12b-it",
        "fact_battery_path": "fact_battery/gemma-3-12b-it.json",
    },
    "gemma_31b": {
        "model_name": "google/gemma-4-31B",
        "fact_battery_path": "fact_battery/gemma-4-31b.json",
    },
}


def _select_device() -> str:
    override = os.environ.get("PATH_PATCHING_DEVICE")
    if override:
        return override
    if t.cuda.is_available():
        return "cuda"
    if getattr(t.backends, "mps", None) is not None and t.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_dtype(device: str) -> str:
    override = os.environ.get("PATH_PATCHING_DTYPE")
    if override:
        return override
    if device == "cuda":
        return "bfloat16"
    if device == "mps":
        return "float16"
    return "float32"


def get_config(model_key: str) -> dict:
    assert model_key in CONFIGS, f"Unknown model: {model_key}. Choose from {list(CONFIGS.keys())}"
    config = CONFIGS[model_key].copy()
    config["device"] = _select_device()
    config["dtype"] = _select_dtype(config["device"])
    config["results_dir"] = f"results/{model_key}"
    return config