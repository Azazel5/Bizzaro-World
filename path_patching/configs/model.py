CONFIGS = {
    "gemma_2b": {
        "model_name": "google/gemma-2b",
        "fact_battery_path": "fact_battery/gemma-2b.json",
    },
    "gemma_12b": {
        "model_name": "google/gemma-2-9b",
        "fact_battery_path": "fact_battery/gemma-12b.json",
    },
}

# Shared across all models
DEVICE = "cuda"
DTYPE = "bfloat16"


def get_config(model_key: str) -> dict:
    assert model_key in CONFIGS, f"Unknown model: {model_key}. Choose from {list(CONFIGS.keys())}"
    config = CONFIGS[model_key].copy()
    config["device"] = DEVICE
    config["dtype"] = DTYPE
    config["results_dir"] = f"results/{model_key}"
    return config