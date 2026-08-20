#!/usr/bin/env python3
"""
Keyword-consistency audit for the SAE features that have real per-prompt
firing evidence (sae/results/feature_evidence_gemma_{12b,27b}.json, from
sae/sae_feature_evidence.py).

Motivation: eyeballing a feature's firing prompts and spotting a shared word
("cat" in 8614's three firings) is a hypothesis, not a confirmation -- the
word might just as easily appear elsewhere in the battery where the feature
DOESN'T fire, which would falsify it. This script makes that check
systematic and automatic instead of ad hoc:

  1. For each evidenced feature, pull its firing prompts (clean or corrupt
     side, activation > 0).
  2. Extract candidate keywords -- content words (stopwords stripped) from
     those firing prompts, ranked by how many firing prompts they appear in.
  3. For each candidate, search the WHOLE battery (both clean and corrupt
     sides of all 57 pairs -- 114 prompts) for every other occurrence of
     that word, and report what fraction of those occurrences also fire the
     feature.

A candidate that fires 100% of the time it appears anywhere in the battery
(and appears more than once) is real, testable evidence the feature tracks
that word/concept. Anything below 100% is falsified as a *complete*
explanation (the feature is doing something more selective than that
keyword alone). Anything appearing exactly once in the whole battery is
"single-shot" -- consistent with a hypothesis but not independent
confirmation, since there's no second chance for it to be wrong.

Pure post-hoc analysis of JSON already on disk -- no GPU, no model, runs
anywhere.

Usage:
    python sae_feature_keyword_consistency.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

EVIDENCE_PATHS = {
    "GEMMA 12B (L38)": REPO_ROOT / "sae" / "results" / "feature_evidence_gemma_12b.json",
    "GEMMA 27B (L54)": REPO_ROOT / "sae" / "results" / "feature_evidence_gemma_27b.json",
}

# Pure function words / templated connective tissue that recurs across
# nearly every prompt in this battery regardless of topic -- stripped so
# candidate keywords are content words, not sentence scaffolding. Deliberately
# NOT stripping domain nouns like "capital" or "metric" -- whether those are
# real drivers or not is exactly what the fire-rate check below is for.
STOPWORDS = {
    "the", "a", "an", "is", "of", "in", "for", "with", "by", "was", "were",
    "are", "be", "that", "this", "it", "its", "from", "to", "on", "at",
    "as", "and", "or", "but", "played", "produces", "result", "later",
    "first", "currently", "named", "common", "example", "famous",
    "originally", "found", "has",
}

MAX_CANDIDATES_PRINTED = 8


def tokenize(prompt: str) -> list[str]:
    words = re.findall(r"[A-Za-z']+", prompt)
    return [w for w in words if w.lower() not in STOPWORDS and len(w) > 2]


def analyze(evidence_path: Path, model_label: str) -> None:
    ev = json.loads(evidence_path.read_text())
    records = ev["records"]

    # Flat index of every (idx, side, prompt) in the whole battery, both
    # clean and corrupt -- this is what candidate keywords get tested against.
    battery_flat: list[tuple[int, str, str, str]] = []
    for r in records:
        battery_flat.append((r["idx"], "clean", r["clean_prompt"], r["category"]))
        battery_flat.append((r["idx"], "corrupt", r["corrupt_prompt"], r["category"]))

    print(f"\n{'#' * 90}")
    print(f"# {model_label}  ({len(ev['target_features'])} evidenced features, "
          f"{len(records)} prompt pairs, {len(battery_flat)} total prompts)")
    print(f"{'#' * 90}")

    for spec in ev["target_features"]:
        fidx, tier = spec["index"], spec["tier"]
        firing = []
        for r in records:
            ca = r["features"][str(fidx)]["clean_activation"]
            ka = r["features"][str(fidx)]["corrupt_activation"]
            if ca > 0:
                firing.append((r["idx"], "clean", r["clean_prompt"], ca))
            if ka > 0:
                firing.append((r["idx"], "corrupt", r["corrupt_prompt"], ka))

        print(f"\n=== feature {fidx} [{tier}]  fires on {len(firing)}/{len(battery_flat)} prompts ===")
        if not firing:
            print("    never fires -- nothing to analyze")
            continue

        word_support: dict[str, int] = defaultdict(int)
        for _, _, p, _ in firing:
            for w in set(tokenize(p)):
                word_support[w] += 1

        shared = sorted([w for w, c in word_support.items() if c >= 2], key=lambda w: -word_support[w])
        singles = sorted([w for w, c in word_support.items() if c == 1])
        candidates = shared + singles

        for w in candidates[:MAX_CANDIDATES_PRINTED]:
            matches = [(idx, side, p, cat) for idx, side, p, cat in battery_flat if w in p]
            record_by_idx = {r["idx"]: r for r in records}
            fires_on = []
            for idx, side, p, cat in matches:
                r = record_by_idx[idx]
                act = r["features"][str(fidx)]["clean_activation" if side == "clean" else "corrupt_activation"]
                fires_on.append(act > 0)
            n_match = len(matches)
            n_fire = sum(fires_on)
            rate = n_fire / n_match if n_match else 0.0
            if n_match == 1:
                tag = "SINGLE-SHOT (untestable)"
            elif rate == 1.0:
                tag = "CONSISTENT"
            else:
                tag = "LEAKY/FALSIFIED"
            print(f"    keyword {w!r:20s} support={word_support[w]}  battery_matches={n_match}  "
                  f"fires_on={n_fire}/{n_match} ({rate:.0%})  [{tag}]")


def main() -> int:
    for label, path in EVIDENCE_PATHS.items():
        if not path.exists():
            print(f"[skip] {label}: {path} not found")
            continue
        analyze(path, label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
