#!/usr/bin/env python3
"""
Build a hand-label-ready table for the SAE features found in
sae/results/differential_features/, grounded in real per-prompt firing
evidence (sae/results/feature_evidence_gemma_{model}.json, from
sae/sae_feature_evidence.py -- run that on GPU first, this script needs
nothing but its output).

This does NOT invent semantic labels automatically -- a heuristic majority-
category guess is offered as a DRAFT starting point, explicitly marked as
such, because assigning real meaning to what a feature represents requires
a human (or an LLM) actually reading the firing prompts and judging whether
there's a coherent theme, not just counting categories. What this script
does do reliably: organize the real evidence (which prompts/categories fire
each feature, clean vs corrupt, alongside the aggregate differential stats)
so that judgment call is fast and evidence-grounded instead of a guess from
a bare feature index and a number.

Usage:
    python sae_feature_handlabel_table.py --model gemma_12b
    python sae_feature_handlabel_table.py --model gemma_27b
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "gemma_12b": {
        "evidence_path": REPO_ROOT / "sae" / "results" / "feature_evidence_gemma_12b.json",
        "diff_path": REPO_ROOT / "sae" / "results" / "differential_features" / "differential_features_gemma_12b.json",
    },
    "gemma_27b": {
        "evidence_path": REPO_ROOT / "sae" / "results" / "feature_evidence_gemma_27b.json",
        "diff_path": REPO_ROOT / "sae" / "results" / "differential_features" / "differential_features_gemma_27b.json",
    },
}


def _draft_label(clean_cats: list[str], corrupt_cats: list[str]) -> str:
    """Cheap heuristic starting point -- NOT a real label. Flags whether firing
    is concentrated in one category (suggests a topical feature) or spread
    across many (suggests something more generic -- register, syntax, a
    high-frequency token shared across categories)."""
    all_cats = clean_cats + corrupt_cats
    if not all_cats:
        return "DRAFT: never fires on this battery -- not labelable from this evidence"
    counts = Counter(all_cats)
    top_cat, top_n = counts.most_common(1)[0]
    if top_n == len(all_cats) and len(counts) == 1:
        return f"DRAFT: fires only on {top_cat!r} prompts ({top_n}/{len(all_cats)}) -- likely topical"
    if top_n / len(all_cats) >= 0.6:
        return f"DRAFT: mostly {top_cat!r} ({top_n}/{len(all_cats)}) -- probably topical, check outliers"
    return (f"DRAFT: spread across {len(counts)} categories, no dominant one "
            f"({dict(counts)}) -- possibly generic/syntactic, read the prompts")


def _find_diff_record(diff_data: dict[str, Any], feature_idx: int) -> dict[str, Any] | None:
    for pool in ("top_features_by_magnitude", "top_features_by_rate"):
        for r in diff_data[pool]:
            if r["feature_index"] == feature_idx:
                return r
    return None


def build_table(model_key: str) -> None:
    config = MODEL_CONFIGS[model_key]
    if not config["evidence_path"].exists():
        raise FileNotFoundError(
            f"missing {config['evidence_path']} -- run "
            f"`python sae/sae_feature_evidence.py --model {model_key}` on GPU first. "
            f"This script only organizes evidence that script produces; it can't "
            f"hand-label features that were never run against the battery."
        )
    evidence = json.loads(config["evidence_path"].read_text())
    diff_data = json.loads(config["diff_path"].read_text())

    print(f"\n{'#' * 70}")
    print(f"# Hand-label evidence table: {model_key} (L{evidence['target_layer']}, "
          f"{evidence['sae_release']} / {evidence['sae_id']})")
    print(f"{'#' * 70}\n")

    rows = []
    for fidx in evidence["target_features"]:
        clean_fires = [r for r in evidence["records"] if r["features"][str(fidx)]["clean_activation"] > 0]
        corrupt_fires = [r for r in evidence["records"] if r["features"][str(fidx)]["corrupt_activation"] > 0]
        clean_cats = [r["category"] for r in clean_fires]
        corrupt_cats = [r["category"] for r in corrupt_fires]

        diff_rec = _find_diff_record(diff_data, fidx)
        draft = _draft_label(clean_cats, corrupt_cats)

        print(f"--- feature {fidx} ---")
        if diff_rec:
            print(f"  differential_activation={diff_rec['differential_activation']:+.3f}  "
                  f"clean_rate={diff_rec['clean_activation_rate']:.3f}  "
                  f"corrupt_rate={diff_rec['corrupt_activation_rate']:.3f}")
        print(f"  clean fires ({len(clean_fires)}):")
        for r in clean_fires:
            print(f"    [{r['idx']}] {r['category']}: {r['clean_prompt']!r} "
                  f"(act={r['features'][str(fidx)]['clean_activation']:.2f})")
        print(f"  corrupt fires ({len(corrupt_fires)}):")
        for r in corrupt_fires:
            print(f"    [{r['idx']}] {r['category']}: {r['corrupt_prompt']!r} "
                  f"(act={r['features'][str(fidx)]['corrupt_activation']:.2f})")
        print(f"  {draft}\n")

        rows.append({
            "feature_index": fidx,
            "differential_activation": diff_rec["differential_activation"] if diff_rec else None,
            "clean_activation_rate": diff_rec["clean_activation_rate"] if diff_rec else None,
            "corrupt_activation_rate": diff_rec["corrupt_activation_rate"] if diff_rec else None,
            "clean_firing_categories": clean_cats,
            "corrupt_firing_categories": corrupt_cats,
            "draft_label": draft,
            "hand_label": "TODO -- fill in after reading the firing prompts above",
        })

    out_dir = SCRIPT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sae_feature_handlabels_{model_key}.json"
    out_path.write_text(json.dumps({"model_key": model_key, "rows": rows}, indent=2) + "\n")
    print(f"[save] wrote {out_path}")

    md_path = out_dir / f"sae_feature_handlabels_{model_key}.md"
    lines = [
        f"| feature | diff_act | clean_rate | corrupt_rate | draft label | hand label |",
        f"|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['feature_index']} | {r['differential_activation']:+.2f} | "
            f"{r['clean_activation_rate']:.2f} | {r['corrupt_activation_rate']:.2f} | "
            f"{r['draft_label'][:60]} | {r['hand_label']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"[save] wrote {md_path} (paper-ready table skeleton, drop into the draft and fill 'hand label')")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a hand-label evidence table for top SAE features.")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()
    build_table(args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
