from typing import Dict, Any, List

import torch

from transformer_lens import HookedTransformer


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


model = _load_model()


def _token_ids_for_target(target_token: str) -> List[int]:
    """
    Gemma uses SentencePiece tokenization (https://keras.io/keras_hub/api/models/gemma/gemma_tokenizer/), 
    so " sphere" and " cube" (leading space) are often single tokens, but we verify here.
    """
    
    ids = model.tokenizer.encode(target_token, add_special_tokens=False)
    # HF tokenizers sometimes return a list already; keep this robust.
    if isinstance(ids, int):
        ids = [ids]
    return list(ids)


def _softmax_entropy_from_logits(final_logits: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy H(p) = -sum_i p_i log p_i computed over vocab probabilities.
    """

    probs = torch.softmax(final_logits, dim=-1)
    # Small epsilon avoids NaNs from log(0) in edge cases.
    log_probs = torch.log(probs + 1e-20)
    entropy = -(probs * log_probs).sum()
    return entropy


def analyze_behavioral_friction(
    clean_prompt: str,
    bizarro_prompt: str,
    target_token: str,
) -> Dict[str, Any]:
    """
    Runs a forward pass on two prompts and compares:
      1) P(target_token | prompt) using the next-token logits at the final position
      2) Shannon entropy of the full next-token distribution at the final position

    Notes:
    - This function expects `target_token` to encode to exactly ONE token id under
      Gemma's tokenizer; otherwise the probability can't be read from a single
      next-token distribution.
    """

    target_ids = _token_ids_for_target(target_token)

    if len(target_ids) != 1:
        raise ValueError(
            f"`target_token={target_token!r}` encodes to {len(target_ids)} ids {target_ids}. "
            "This script expects a single token so probability can be computed from the "
            "next-token logits at the final prompt position."
        )

    target_id = target_ids[0]

    def run_one(prompt: str) -> Dict[str, float]:
        tokens = model.to_tokens(prompt, prepend_bos=True)
        tokens = tokens.to(model.cfg.device)
        
        with torch.no_grad():
            logits = model(tokens)  # [batch, seq, vocab]

        final_logits = logits[0, -1, :]  # next-token distribution after the last prompt token
        probs = torch.softmax(final_logits, dim=-1)

        prob_target = probs[target_id].item()
        entropy = _softmax_entropy_from_logits(final_logits).item()
        return {"prob": prob_target, "entropy": entropy}

    clean_stats = run_one(clean_prompt)
    bizarro_stats = run_one(bizarro_prompt)

    print(f"Target token {target_token!r} (id={target_id}):")
    print(f"  Clean   prob={clean_stats['prob']:.6g} entropy={clean_stats['entropy']:.6g}")
    print(f"  Bizarro prob={bizarro_stats['prob']:.6g} entropy={bizarro_stats['entropy']:.6g}")

    return {
        "target_token": target_token,
        "target_token_id": target_id,
        "clean": clean_stats,
        "bizarro": bizarro_stats,
    }


if __name__ == "__main__":
    clean_prompt = (
        "The capital of France is Paris. The shape of a standard basketball is a"
    )
    bizarro_prompt = (
        "In Bizarro World, everything round becomes a cube. The shape of a standard basketball is a"
    )

    target_tokens = [" sphere", " cube"]

    # Also print tokenization so it's clear which id we are tracking.
    for t in target_tokens:
        print(f"{t!r} -> ids {_token_ids_for_target(t)}")

    for t in target_tokens:
        analyze_behavioral_friction(clean_prompt, bizarro_prompt, t)

