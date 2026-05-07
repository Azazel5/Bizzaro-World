## LLaMA 3 70B (8-bit, bitsandbytes)

This directory contains **LLaMA-specific** code only.

Common assets remain at repo root:

- `fact_battery/` (shared data; per-model batteries)
- `scripts/` (shared experiment code where applicable)
- `shared/` (shared utilities)

### Load the model in 8-bit (sanity check)

```bash
python llama-70b-8bitquantized/load_llama3_70b_bnb8.py
```

### Run triage (LD/TotalSwing) on the shared fact battery

```bash
python llama-70b-8bitquantized/triage_llama3_70b_bnb8.py --outdir runs/llama3-70b-bnb8/triage
```

Notes:

- This uses Hugging Face `transformers` + `bitsandbytes` (8-bit weights).
- KV cache is negligible for your short fact-battery prompts; the dominant term is weights.
