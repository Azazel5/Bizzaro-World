#!/usr/bin/env python3
from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = "meta-llama/Meta-Llama-3-70B"


def load_llama3_70b_8bit():
    """
    Load Meta-Llama-3-70B in 8-bit using bitsandbytes.

    This is inference-only. Do not enable gradients/optimizer state.
    """
    quant = BitsAndBytesConfig(load_in_8bit=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tok


def main() -> None:
    # Helpful for gated models: require user to have HF token set up outside code.
    _ = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    model, tok = load_llama3_70b_8bit()
    prompt = "The capital of France is"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    logits_last = out.logits[0, -1]
    top_id = int(torch.argmax(logits_last).item())
    print(f"Loaded {MODEL_ID} (8-bit). Sample next token: {tok.decode([top_id])!r}")


if __name__ == "__main__":
    main()

