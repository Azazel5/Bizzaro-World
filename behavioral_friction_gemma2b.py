"""
Behavioral friction + Fact Battery evaluation for Gemma 2B (TransformerLens).

Aligned prompt pairs live in ``fact_battery.json`` (see ``load_fact_battery``).

Run: python behavioral_friction_gemma2b.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from transformer_lens import HookedTransformer


MODEL_NAME = "google/gemma-2b"

_DEFAULT_FACT_BATTERY_PATH = Path(__file__).with_name("fact_battery.json")


def load_fact_battery(path: Optional[Path] = None) -> List[Dict[str, str]]:
    """
    Load aligned prompt pairs from JSON (20 categories × 3 pairs by default).

    Each entry: category, clean_prompt, corrupt_prompt, clean_target, corrupt_target.
    """
    p = path or _DEFAULT_FACT_BATTERY_PATH
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{p} must contain a JSON array of objects")
    out: List[Dict[str, str]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"{p}[{i}] must be an object")
        out.append({str(k): str(v) for k, v in item.items()})
    return out


# Populated at import from `fact_battery.json` beside this module.
FACT_BATTERY: List[Dict[str, str]] = load_fact_battery()


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


def _token_ids_for_target(tokenizer: Any, target_token: str) -> List[int]:
    ids = tokenizer.encode(target_token, add_special_tokens=False)
    if isinstance(ids, int):
        ids = [ids]
    return list(ids)


def _prompt_token_ids(tokenizer: Any, prompt: str) -> List[int]:
    return list(tokenizer.encode(prompt, add_special_tokens=False))


def _softmax_entropy_from_logits(final_logits: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy H(p) = -sum_i p_i log p_i over the full vocabulary.

    Uses log-softmax in float32 for numerical stability (avoids log(0) and fp16
    underflow from softmax-then-log on large vocabs).
    """
    log_probs = torch.log_softmax(final_logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum()
    return entropy


def _prob_and_entropy_for_target(
    model: HookedTransformer,
    prompt: str,
    target_token: str,
) -> Dict[str, float]:
    tids = _token_ids_for_target(model.tokenizer, target_token)
    if len(tids) != 1:
        raise ValueError(
            f"{target_token!r} encodes to {len(tids)} ids {tids}; expected exactly one."
        )
    tid = tids[0]
    tokens = model.to_tokens(prompt, prepend_bos=True).to(model.cfg.device)
    with torch.no_grad():
        logits = model(tokens)
    final_logits = logits[0, -1, :]
    probs = torch.softmax(final_logits.float(), dim=-1)
    prob_target = probs[tid].item()
    entropy = _softmax_entropy_from_logits(final_logits).item()
    return {"prob": prob_target, "entropy": entropy, "target_id": tid}


def _validate_fact_entry(model: HookedTransformer, entry: Dict[str, str], index: int) -> None:
    tok = model.tokenizer
    c_ids = _prompt_token_ids(tok, entry["clean_prompt"])
    x_ids = _prompt_token_ids(tok, entry["corrupt_prompt"])
    if len(c_ids) != len(x_ids):
        raise ValueError(
            f"FACT_BATTERY[{index}] token length mismatch: "
            f"clean={len(c_ids)} corrupt={len(x_ids)} "
            f"clean={entry['clean_prompt']!r} corrupt={entry['corrupt_prompt']!r}"
        )
    for key in ("clean_target", "corrupt_target"):
        n = len(_token_ids_for_target(tok, entry[key]))
        if n != 1:
            raise ValueError(
                f"FACT_BATTERY[{index}] {key}={entry[key]!r} has {n} token ids; need 1."
            )


def run_fact_battery(model: HookedTransformer) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, entry in enumerate(FACT_BATTERY):
        _validate_fact_entry(model, entry, i)
        clean_stats = _prob_and_entropy_for_target(
            model, entry["clean_prompt"], entry["clean_target"]
        )
        corrupt_stats = _prob_and_entropy_for_target(
            model, entry["corrupt_prompt"], entry["corrupt_target"]
        )
        delta_p = abs(clean_stats["prob"] - corrupt_stats["prob"])
        rows.append(
            {
                "idx": i,
                "category": entry["category"],
                "clean_prompt": entry["clean_prompt"],
                "corrupt_prompt": entry["corrupt_prompt"],
                "clean_target": entry["clean_target"],
                "corrupt_target": entry["corrupt_target"],
                "p_clean": clean_stats["prob"],
                "p_corrupt": corrupt_stats["prob"],
                "delta_p": delta_p,
                "entropy_clean": clean_stats["entropy"],
                "entropy_corrupt": corrupt_stats["entropy"],
            }
        )
    return rows


def _print_ranked_table(rows: List[Dict[str, Any]]) -> None:
    ranked = sorted(rows, key=lambda r: r["delta_p"], reverse=True)
    headers = [
        "rank",
        "ΔP",
        "P_clean",
        "P_corrupt",
        "H_clean",
        "H_corrupt",
        "category",
        "clean → corrupt (targets)",
    ]
    col_w = [5, 10, 10, 10, 10, 10, 22, 55]
    fmt = " ".join(f"{{:{w}}}" for w in col_w)

    print(fmt.format(*headers))
    print(" ".join("-" * w for w in col_w))
    for rank, r in enumerate(ranked, start=1):
        tgt = f"{r['clean_target']!r} vs {r['corrupt_target']!r}"
        line = fmt.format(
            str(rank),
            f"{r['delta_p']:.4f}",
            f"{r['p_clean']:.4f}",
            f"{r['p_corrupt']:.4f}",
            f"{r['entropy_clean']:.2f}",
            f"{r['entropy_corrupt']:.2f}",
            r["category"][: col_w[6]],
            (r["clean_prompt"][:28] + "…|" + r["corrupt_prompt"][:22] + "… " + tgt)[: col_w[7]],
        )
        print(line)


def main() -> None:
    print(f"Loading {MODEL_NAME} …")
    model = _load_model()
    print("Running FACT_BATTERY …")
    rows = run_fact_battery(model)
    print(f"\nCompleted {len(rows)} pairs. Ranked by probability delta |P_clean - P_corrupt|:\n")
    _print_ranked_table(rows)


# --- Optional: legacy demo (Bizarro vs clean basketball) ---
def analyze_behavioral_friction(
    model: HookedTransformer,
    clean_prompt: str,
    bizarro_prompt: str,
    target_token: str,
) -> Dict[str, Any]:
    target_ids = _token_ids_for_target(model.tokenizer, target_token)
    if len(target_ids) != 1:
        raise ValueError(
            f"`target_token={target_token!r}` encodes to {len(target_ids)} ids {target_ids}."
        )
    target_id = target_ids[0]

    def run_one(prompt: str) -> Dict[str, float]:
        s = _prob_and_entropy_for_target(model, prompt, target_token)
        return {"prob": s["prob"], "entropy": s["entropy"]}

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
    main()
