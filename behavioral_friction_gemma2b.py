"""
Behavioral friction + Fact Battery evaluation for Gemma 2B (TransformerLens).

Aligned prompt pairs live in ``fact_battery.json`` (see ``load_fact_battery``).

Run: python behavioral_friction_gemma2b.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _final_logits(model: HookedTransformer, prompt: str) -> torch.Tensor:
    """Next-token logits at the final prompt position, shape [vocab]."""
    tokens = model.to_tokens(prompt, prepend_bos=True).to(model.cfg.device)
    with torch.no_grad():
        logits = model(tokens)
    return logits[0, -1, :]


def _ld_and_target_probs(
    final_logits: torch.Tensor,
    clean_target_id: int,
    corrupt_target_id: int,
) -> Tuple[float, float, float]:
    """
    Logit difference LD = logit(clean_target) - logit(corrupt_target) in fp32,
    plus softmax probabilities for the two token ids (sanity checks).
    """
    lf = final_logits.float()
    ld = (lf[clean_target_id] - lf[corrupt_target_id]).item()
    probs = torch.softmax(lf, dim=-1)
    p_clean_tok = probs[clean_target_id].item()
    p_corrupt_tok = probs[corrupt_target_id].item()
    return ld, p_clean_tok, p_corrupt_tok


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
    final_logits = _final_logits(model, prompt)
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
    """
    For each aligned pair, compute bidirectional logit differences (patching-style triage).

    LD = logit(clean_target) - logit(corrupt_target) at the final prompt position.

    - LD_clean: forward on clean_prompt (expect strongly positive if the model prefers
      the clean completion over the corrupt token on the clean context).
    - LD_corrupt: forward on corrupt_prompt (expect strongly negative if the model
      prefers the corrupt completion on the corrupt context).

    Total_Swing = LD_clean - LD_corrupt measures symmetric separation (larger is better
    for \"golden pair\" screening before activation patching).
    """
    rows: List[Dict[str, Any]] = []
    tok = model.tokenizer
    for i, entry in enumerate(FACT_BATTERY):
        _validate_fact_entry(model, entry, i)
        clean_tid = _token_ids_for_target(tok, entry["clean_target"])[0]
        corrupt_tid = _token_ids_for_target(tok, entry["corrupt_target"])[0]

        lf_clean = _final_logits(model, entry["clean_prompt"])
        ld_clean, p_clean_on_clean, _p_corrupt_on_clean = _ld_and_target_probs(
            lf_clean, clean_tid, corrupt_tid
        )

        lf_corrupt = _final_logits(model, entry["corrupt_prompt"])
        ld_corrupt, _p_clean_on_corrupt, p_corrupt_on_corrupt = _ld_and_target_probs(
            lf_corrupt, clean_tid, corrupt_tid
        )

        total_swing = ld_clean - ld_corrupt

        rows.append(
            {
                "idx": i,
                "category": entry["category"],
                "clean_prompt": entry["clean_prompt"],
                "corrupt_prompt": entry["corrupt_prompt"],
                "clean_target": entry["clean_target"],
                "corrupt_target": entry["corrupt_target"],
                "clean_target_id": clean_tid,
                "corrupt_target_id": corrupt_tid,
                "ld_clean": ld_clean,
                "ld_corrupt": ld_corrupt,
                "total_swing": total_swing,
                "p_clean": p_clean_on_clean,
                "p_corrupt": p_corrupt_on_corrupt,
            }
        )
    return rows


def _print_ranked_table(ranked: List[Dict[str, Any]]) -> None:
    """
    Print rows already sorted by total_swing (desc).
    Top rows are strongest golden-pair candidates for activation patching.
    """
    headers = [
        "rank",
        "idx",
        "TotalSwing",
        "LD_clean",
        "LD_corrupt",
        "P_clean",
        "P_corrupt",
        "category",
        "prompt_prefix",
    ]
    col_w = [5, 4, 11, 9, 9, 8, 8, 18, 40]
    fmt = " ".join(f"{{:{w}}}" for w in col_w)

    print(
        "TotalSwing = LD_clean - LD_corrupt (primary rank key; larger => stronger bidirectional flip). "
        "LD_* = logit(clean_tgt) - logit(corrupt_tgt) on that prompt. "
        "idx = row index in fact_battery.json. "
        "P_clean = P(clean_target|clean_prompt); P_corrupt = P(corrupt_target|corrupt_prompt)."
    )
    print(fmt.format(*headers))
    print(" ".join("-" * w for w in col_w))
    for rank, r in enumerate(ranked, start=1):
        prefix = f"{r['clean_prompt']}|{r['corrupt_prompt']}"
        if len(prefix) > col_w[8]:
            prefix = prefix[: col_w[8] - 1] + "…"
        line = fmt.format(
            str(rank),
            str(r["idx"]),
            f"{r['total_swing']:.3f}",
            f"{r['ld_clean']:.3f}",
            f"{r['ld_corrupt']:.3f}",
            f"{r['p_clean']:.4f}",
            f"{r['p_corrupt']:.4f}",
            str(r["category"])[: col_w[7]],
            prefix,
        )
        print(line)


TRIAGE_CSV_NAME = "fact_battery_triage.csv"


def write_triage_csv(ranked: List[Dict[str, Any]], path: Path) -> None:
    """
    Write sorted triage metrics for notebooks / spreadsheets.
    One row per battery entry; `rank` is post-sort order (1 = highest TotalSwing).
    """
    fieldnames = [
        "rank",
        "battery_idx",
        "total_swing",
        "ld_clean",
        "ld_corrupt",
        "p_clean_target_on_clean",
        "p_corrupt_target_on_corrupt",
        "category",
        "clean_prompt",
        "corrupt_prompt",
        "clean_target",
        "corrupt_target",
        "clean_target_id",
        "corrupt_target_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank, r in enumerate(ranked, start=1):
            w.writerow(
                {
                    "rank": rank,
                    "battery_idx": r["idx"],
                    "total_swing": f"{r['total_swing']:.6f}",
                    "ld_clean": f"{r['ld_clean']:.6f}",
                    "ld_corrupt": f"{r['ld_corrupt']:.6f}",
                    "p_clean_target_on_clean": f"{r['p_clean']:.8f}",
                    "p_corrupt_target_on_corrupt": f"{r['p_corrupt']:.8f}",
                    "category": r["category"],
                    "clean_prompt": r["clean_prompt"],
                    "corrupt_prompt": r["corrupt_prompt"],
                    "clean_target": r["clean_target"],
                    "corrupt_target": r["corrupt_target"],
                    "clean_target_id": r["clean_target_id"],
                    "corrupt_target_id": r["corrupt_target_id"],
                }
            )


def main() -> None:
    print(f"Loading {MODEL_NAME} …")
    model = _load_model()
    print("Running FACT_BATTERY …")
    rows = run_fact_battery(model)
    ranked = sorted(rows, key=lambda r: r["total_swing"], reverse=True)

    out_csv = Path(__file__).with_name(TRIAGE_CSV_NAME)
    write_triage_csv(ranked, out_csv)

    print(
        f"\nCompleted {len(rows)} pairs. Ranked by TotalSwing "
        f"(LD_clean - LD_corrupt), descending — top rows are strongest golden-pair candidates.\n"
        f"Wrote CSV: {out_csv}\n"
    )
    _print_ranked_table(ranked)


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
