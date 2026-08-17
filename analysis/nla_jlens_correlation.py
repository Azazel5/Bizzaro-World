#!/usr/bin/env python3
"""
Per-pair correlation between NLA's AR reconstruction faithfulness (cosine
similarity to the real ground-truth activation) and the Jacobian lens's rank
of the correct-answer token, evaluated AT THE SAME LAYER NLA was extracted
from (32 for 12B, 41 for 27B).

Question: do facts NLA reconstructs well ALSO show a more confident (lower)
rank under an entirely independent linear readout, at that same depth? If so,
that's two independently-built techniques agreeing at the item level, not
just in aggregate -- a stronger claim than either technique's own headline
number.

Pure post-hoc analysis of JSON already on disk from nla/nla_ar_faithfulness.py
and jlens_pipeline/apply_jacobian_lens.py -- no GPU, no model, no new
extraction. Runs anywhere.

Expected direction: NEGATIVE correlation (higher cosine = better
reconstruction should pair with LOWER rank = more confident linear readout).
Reported, not assumed -- see printed output for the actual sign and whether
it clears significance at n=57 (or n=114 pooled clean+corrupt).

Usage:
    python nla_jlens_correlation.py --model gemma_12b
    python nla_jlens_correlation.py --model gemma_27b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "gemma_12b": {
        "ar_path": REPO_ROOT / "nla" / "results" / "ar_faithfulness_gemma_12b_L32.json",
        "jl_path": REPO_ROOT / "jlens_pipeline" / "results" / "jlens_ranks_gemma_12b.json",
        "nla_layer": 32,
    },
    "gemma_27b": {
        "ar_path": REPO_ROOT / "nla" / "results" / "ar_faithfulness_gemma_27b_L41.json",
        "jl_path": REPO_ROOT / "jlens_pipeline" / "results" / "jlens_ranks_gemma_27b.json",
        "nla_layer": 41,
    },
}


def load_paired_data(model_key: str) -> dict[str, Any]:
    config = MODEL_CONFIGS[model_key]
    ar = json.loads(config["ar_path"].read_text())
    jl = json.loads(config["jl_path"].read_text())

    nla_layer = config["nla_layer"]
    if nla_layer not in jl["layers"]:
        raise ValueError(
            f"NLA layer {nla_layer} not among J-Lens's fitted layers "
            f"{jl['layers'][0]}..{jl['layers'][-1]} for {model_key} -- cannot align."
        )
    layer_idx = jl["layers"].index(nla_layer)
    print(f"[load] {model_key}: NLA layer {nla_layer} -> J-Lens layers[{layer_idx}] "
          f"(sanity: {jl['layers'][layer_idx]} == {nla_layer})", flush=True)

    ar_records = {r["idx"]: r for r in ar["records"]}
    jl_records = {r["idx"]: r for r in jl["records"]}

    common_idx = sorted(set(ar_records) & set(jl_records))
    if len(common_idx) != len(ar_records) or len(common_idx) != len(jl_records):
        print(f"[warn] idx sets don't fully match: AR has {len(ar_records)}, "
              f"J-Lens has {len(jl_records)}, {len(common_idx)} common -- "
              f"proceeding with the intersection only.", flush=True)

    rows = []
    for idx in common_idx:
        ar_r, jl_r = ar_records[idx], jl_records[idx]
        if ar_r["category"] != jl_r["category"]:
            raise ValueError(
                f"idx {idx} category mismatch: AR says {ar_r['category']!r}, "
                f"J-Lens says {jl_r['category']!r} -- these files don't come from "
                f"the same battery ordering, refusing to pair them."
            )
        rows.append({
            "idx": idx,
            "category": ar_r["category"],
            "clean_cosine": ar_r["clean_cosine_similarity"],
            "clean_rank": jl_r["clean_rank_by_layer"][layer_idx],
            "corrupt_cosine": ar_r["corrupt_cosine_similarity"],
            "corrupt_rank": jl_r["corrupt_rank_by_layer"][layer_idx],
        })

    print(f"[load] {len(rows)} aligned pairs, categories verified matching per-idx", flush=True)
    return {"model_key": model_key, "nla_layer": nla_layer, "layer_idx": layer_idx, "rows": rows}


def _corr_block(label: str, cosine: np.ndarray, rank: np.ndarray) -> dict[str, Any]:
    log_rank = np.log10(rank.astype(float))

    spearman = stats.spearmanr(cosine, rank)
    pearson_log = stats.pearsonr(cosine, log_rank)

    print(f"\n  [{label}] n={len(cosine)}")
    print(f"    Spearman(cosine, rank):        rho={spearman.correlation:+.4f}  p={spearman.pvalue:.4g}")
    print(f"    Pearson(cosine, log10(rank)):  r={pearson_log[0]:+.4f}  p={pearson_log[1]:.4g}")

    return {
        "n": len(cosine),
        "spearman_rho": float(spearman.correlation),
        "spearman_p": float(spearman.pvalue),
        "pearson_r_log_rank": float(pearson_log[0]),
        "pearson_p_log_rank": float(pearson_log[1]),
    }


def run_model(model_key: str, results_dir: Path, plot: bool) -> None:
    print(f"\n{'#' * 60}")
    print(f"# NLA <-> J-Lens correlation: {model_key}")
    print(f"{'#' * 60}", flush=True)

    data = load_paired_data(model_key)
    rows = data["rows"]

    clean_cos = np.array([r["clean_cosine"] for r in rows])
    clean_rank = np.array([r["clean_rank"] for r in rows])
    corrupt_cos = np.array([r["corrupt_cosine"] for r in rows])
    corrupt_rank = np.array([r["corrupt_rank"] for r in rows])
    pooled_cos = np.concatenate([clean_cos, corrupt_cos])
    pooled_rank = np.concatenate([clean_rank, corrupt_rank])

    print(f"\n[diag] cosine range: clean [{clean_cos.min():.4f}, {clean_cos.max():.4f}]  "
          f"corrupt [{corrupt_cos.min():.4f}, {corrupt_cos.max():.4f}]")
    print(f"[diag] rank range:   clean [{clean_rank.min()}, {clean_rank.max()}]  "
          f"corrupt [{corrupt_rank.min()}, {corrupt_rank.max()}]")

    results = {
        "clean": _corr_block("clean only", clean_cos, clean_rank),
        "corrupt": _corr_block("corrupt only", corrupt_cos, corrupt_rank),
        "pooled": _corr_block("pooled clean+corrupt", pooled_cos, pooled_rank),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"nla_jlens_correlation_{model_key}.json"
    out_path.write_text(json.dumps({
        "model_key": model_key,
        "nla_layer": data["nla_layer"],
        "n_pairs": len(rows),
        "expected_direction": "negative (higher cosine -> lower/better rank)",
        "correlations": results,
        "rows": rows,
    }, indent=2) + "\n")
    print(f"\n[save] wrote {out_path}", flush=True)

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(clean_cos, clean_rank, alpha=0.7, label="clean", color="#1f77b4", s=28)
        ax.scatter(corrupt_cos, corrupt_rank, alpha=0.7, label="corrupt", color="#d62728", s=28)
        ax.set_yscale("log")
        ax.set_xlabel("AR reconstruction cosine similarity", fontsize=11)
        ax.set_ylabel(f"J-Lens rank of correct answer at L{data['nla_layer']} (log scale)", fontsize=11)
        rho = results["pooled"]["spearman_rho"]
        p = results["pooled"]["spearman_p"]
        ax.set_title(
            f"{model_key}: AR faithfulness vs. J-Lens rank at the NLA layer (L{data['nla_layer']})\n"
            f"pooled Spearman rho={rho:+.3f}, p={p:.3g}, n={len(pooled_cos)}",
            fontsize=11,
        )
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3)

        fig_dir = results_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / f"nla_jlens_correlation_{model_key}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] wrote {fig_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Correlate NLA AR faithfulness against J-Lens rank.")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--results_dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--no-plot", action="store_true", help="Skip the scatter figure.")
    args = parser.parse_args()

    run_model(args.model, args.results_dir, plot=not args.no_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
