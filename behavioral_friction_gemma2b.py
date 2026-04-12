"""
Behavioral friction + Fact Battery evaluation for Gemma 2B (TransformerLens).

Run: python behavioral_friction_gemma2b.py
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from transformer_lens import HookedTransformer


MODEL_NAME = "google/gemma-2b"

# 20 categories × 3 pairs = 60. Prompts are parallel in wording; only the swapped
# entity span differs. Targets use a leading space for SentencePiece-style continuations.
# Runtime checks: single-token targets + equal tokenized prompt lengths (Gemma tokenizer).
FACT_BATTERY: List[Dict[str, str]] = [
    # --- Geography (Capitals) ---
    {
        "category": "geography_capitals",
        "clean_prompt": "The capital of France is",
        "corrupt_prompt": "The capital of Spain is",
        "clean_target": " Paris",
        "corrupt_target": " Madrid",
    },
    {
        "category": "geography_capitals",
        "clean_prompt": "The capital of Italy is",
        "corrupt_prompt": "The capital of Germany is",
        "clean_target": " Rome",
        "corrupt_target": " Berlin",
    },
    {
        "category": "geography_capitals",
        "clean_prompt": "The capital of Japan is",
        "corrupt_prompt": "The capital of China is",
        "clean_target": " Tokyo",
        "corrupt_target": " Beijing",
    },
    # --- Geography (Continents) ---
    {
        "category": "geography_continents",
        "clean_prompt": "The continent where Egypt is found is",
        "corrupt_prompt": "The continent where Japan is found is",
        "clean_target": " Africa",
        "corrupt_target": " Asia",
    },
    {
        "category": "geography_continents",
        "clean_prompt": "France lies entirely within the continent of",
        "corrupt_prompt": "Australia lies entirely within the continent of",
        "clean_target": " Europe",
        "corrupt_target": " Australia",
    },
    {
        "category": "geography_continents",
        "clean_prompt": "The continent where penguins mainly breed is",
        "corrupt_prompt": "The continent where lions mainly live is",
        "clean_target": " Antarctica",
        "corrupt_target": " Africa",
    },
    # --- Chemistry (Symbols) ---
    {
        "category": "chemistry_symbols",
        "clean_prompt": "The chemical symbol for gold is",
        "corrupt_prompt": "The chemical symbol for iron is",
        "clean_target": " Au",
        "corrupt_target": " Fe",
    },
    {
        "category": "chemistry_symbols",
        "clean_prompt": "The chemical symbol for silver is",
        "corrupt_prompt": "The chemical symbol for copper is",
        "clean_target": " Ag",
        "corrupt_target": " Cu",
    },
    {
        "category": "chemistry_symbols",
        "clean_prompt": "The chemical symbol for sodium is",
        "corrupt_prompt": "The chemical symbol for chlorine is",
        "clean_target": " Na",
        "corrupt_target": " Cl",
    },
    # --- Physics (constants / properties) ---
    {
        "category": "physics_constants",
        "clean_prompt": "The particle with a negative charge in an atom is the",
        "corrupt_prompt": "The particle with a positive charge in an atom is the",
        "clean_target": " electron",
        "corrupt_target": " proton",
    },
    {
        "category": "physics_constants",
        "clean_prompt": "The SI unit of electric current is the",
        "corrupt_prompt": "The SI unit of electric resistance is the",
        "clean_target": " ampere",
        "corrupt_target": " ohm",
    },
    {
        "category": "physics_constants",
        "clean_prompt": "The SI unit of power is the",
        "corrupt_prompt": "The SI unit of frequency is the",
        "clean_target": " watt",
        "corrupt_target": " hertz",
    },
    # --- Anatomy (organs / systems) ---
    {
        "category": "anatomy_organs",
        "clean_prompt": "The human organ that pumps blood is the",
        "corrupt_prompt": "The human organ that filters blood is the",
        "clean_target": " heart",
        "corrupt_target": " kidney",
    },
    {
        "category": "anatomy_organs",
        "clean_prompt": "The organ responsible for breathing is the",
        "corrupt_prompt": "The organ responsible for digestion is the",
        "clean_target": " lungs",
        "corrupt_target": " stomach",
    },
    {
        "category": "anatomy_organs",
        "clean_prompt": "A whale breathes air using its",
        "corrupt_prompt": "A fish extracts oxygen using its",
        "clean_target": " lungs",
        "corrupt_target": " gills",
    },
    # --- Astronomy (planets) ---
    {
        "category": "astronomy_planets",
        "clean_prompt": "The planet closest to our Sun is",
        "corrupt_prompt": "The planet third from our Sun is",
        "clean_target": " Mercury",
        "corrupt_target": " Earth",
    },
    {
        "category": "astronomy_planets",
        "clean_prompt": "The largest planet in the solar system is",
        "corrupt_prompt": "The smallest planet in the solar system is",
        "clean_target": " Jupiter",
        "corrupt_target": " Mercury",
    },
    {
        "category": "astronomy_planets",
        "clean_prompt": "The planet famous for its rings is",
        "corrupt_prompt": "The planet famous for its red color is",
        "clean_target": " Saturn",
        "corrupt_target": " Mars",
    },
    # --- Mathematics (simple addition) ---
    {
        "category": "mathematics_addition",
        "clean_prompt": "Two plus two equals",
        "corrupt_prompt": "Three plus three equals",
        "clean_target": " four",
        "corrupt_target": " six",
    },
    {
        "category": "mathematics_addition",
        "clean_prompt": "One plus two equals",
        "corrupt_prompt": "One plus three equals",
        "clean_target": " three",
        "corrupt_target": " four",
    },
    {
        "category": "mathematics_addition",
        "clean_prompt": "Six plus two equals",
        "corrupt_prompt": "Five plus two equals",
        "clean_target": " eight",
        "corrupt_target": " seven",
    },
    # --- Mathematics (geometry / shapes) ---
    {
        "category": "mathematics_geometry",
        "clean_prompt": "A shape with three equal sides is",
        "corrupt_prompt": "A shape with four equal sides is",
        "clean_target": " triangle",
        "corrupt_target": " square",
    },
    {
        "category": "mathematics_geometry",
        "clean_prompt": "A three dimensional box shape is a",
        "corrupt_prompt": "A round three dimensional ball is a",
        "clean_target": " cube",
        "corrupt_target": " sphere",
    },
    {
        "category": "mathematics_geometry",
        "clean_prompt": "A polygon with five sides is called",
        "corrupt_prompt": "A polygon with six sides is called",
        "clean_target": " pentagon",
        "corrupt_target": " hexagon",
    },
    # --- Sports (associations / equipment) ---
    {
        "category": "sports_equipment",
        "clean_prompt": "Tennis is played with a",
        "corrupt_prompt": "Golf is played with a",
        "clean_target": " racket",
        "corrupt_target": " club",
    },
    {
        "category": "sports_equipment",
        "clean_prompt": "Olympic sprinters finish by crossing the",
        "corrupt_prompt": "Olympic swimmers finish by touching the",
        "clean_target": " line",
        "corrupt_target": " wall",
    },
    {
        "category": "sports_equipment",
        "clean_prompt": "Olympic archery targets are shot with an",
        "corrupt_prompt": "Olympic fencing bouts are fought with a",
        "clean_target": " arrow",
        "corrupt_target": " sword",
    },
    # --- Translation (English → Spanish basics) ---
    {
        "category": "translation_spanish",
        "clean_prompt": "The Spanish word for cat is",
        "corrupt_prompt": "The Spanish word for dog is",
        "clean_target": " gato",
        "corrupt_target": " perro",
    },
    {
        "category": "translation_spanish",
        "clean_prompt": "The Spanish word for sun is",
        "corrupt_prompt": "The Spanish word for moon is",
        "clean_target": " sol",
        "corrupt_target": " luna",
    },
    {
        "category": "translation_spanish",
        "clean_prompt": "The Spanish word for water is",
        "corrupt_prompt": "The Spanish word for fire is",
        "clean_target": " agua",
        "corrupt_target": " fuego",
    },
    # --- Translation (English → French basics) ---
    {
        "category": "translation_french",
        "clean_prompt": "The French word for cat is",
        "corrupt_prompt": "The French word for dog is",
        "clean_target": " chat",
        "corrupt_target": " chien",
    },
    {
        "category": "translation_french",
        "clean_prompt": "The French word for sun is",
        "corrupt_prompt": "The French word for moon is",
        "clean_target": " soleil",
        "corrupt_target": " lune",
    },
    {
        "category": "translation_french",
        "clean_prompt": "The French word for water is",
        "corrupt_prompt": "The French word for fire is",
        "clean_target": " eau",
        "corrupt_target": " feu",
    },
    # --- Colors (primary mixing) ---
    {
        "category": "colors_mixing",
        "clean_prompt": "When blue paint mixes with yellow paint the result is",
        "corrupt_prompt": "When red paint mixes with blue paint the result is",
        "clean_target": " green",
        "corrupt_target": " purple",
    },
    {
        "category": "colors_mixing",
        "clean_prompt": "When red paint mixes with yellow paint the result is",
        "corrupt_prompt": "When blue paint mixes with yellow paint the result is",
        "clean_target": " orange",
        "corrupt_target": " green",
    },
    {
        "category": "colors_mixing",
        "clean_prompt": "Mixing red light with green light produces",
        "corrupt_prompt": "Mixing red light with blue light produces",
        "clean_target": " yellow",
        "corrupt_target": " magenta",
    },
    # --- Animal taxonomy ---
    {
        "category": "animal_taxonomy",
        "clean_prompt": "A trout is a kind of",
        "corrupt_prompt": "A sparrow is a kind of",
        "clean_target": " fish",
        "corrupt_target": " bird",
    },
    {
        "category": "animal_taxonomy",
        "clean_prompt": "A cat is a common example of a",
        "corrupt_prompt": "A salmon is a common example of a",
        "clean_target": " mammal",
        "corrupt_target": " fish",
    },
    {
        "category": "animal_taxonomy",
        "clean_prompt": "A polar bear is classified as a",
        "corrupt_prompt": "A great white is classified as a",
        "clean_target": " mammal",
        "corrupt_target": " fish",
    },
    # --- History (famous dates / years) ---
    {
        "category": "history_dates",
        "clean_prompt": "World War II ended in the year",
        "corrupt_prompt": "World War I ended in the year",
        "clean_target": " 1945",
        "corrupt_target": " 1918",
    },
    {
        "category": "history_dates",
        "clean_prompt": "The Magna Carta was sealed in the year",
        "corrupt_prompt": "The US Constitution was signed in the year",
        "clean_target": " 1215",
        "corrupt_target": " 1787",
    },
    {
        "category": "history_dates",
        "clean_prompt": "The fall of the Berlin Wall was in",
        "corrupt_prompt": "The first moon landing by NASA was in",
        "clean_target": " 1989",
        "corrupt_target": " 1969",
    },
    # --- Technology (founders / companies) ---
    {
        "category": "technology_founders",
        "clean_prompt": "Steve Jobs co-founded the company named",
        "corrupt_prompt": "Bill Gates co-founded the company named",
        "clean_target": " Apple",
        "corrupt_target": " Microsoft",
    },
    {
        "category": "technology_founders",
        "clean_prompt": "YouTube was later acquired by the company",
        "corrupt_prompt": "Instagram was later acquired by the company",
        "clean_target": " Google",
        "corrupt_target": " Meta",
    },
    {
        "category": "technology_founders",
        "clean_prompt": "The web browser Chrome was built by the company",
        "corrupt_prompt": "The web browser Safari was built by the company",
        "clean_target": " Google",
        "corrupt_target": " Apple",
    },
    # --- Units of measurement ---
    {
        "category": "units_measurement",
        "clean_prompt": "Distance in the metric system uses",
        "corrupt_prompt": "Mass in the metric system uses",
        "clean_target": " meters",
        "corrupt_target": " grams",
    },
    {
        "category": "units_measurement",
        "clean_prompt": "Liquid volume in metric science uses",
        "corrupt_prompt": "Electric potential in metric science uses",
        "clean_target": " liters",
        "corrupt_target": " volts",
    },
    {
        "category": "units_measurement",
        "clean_prompt": "The metric unit for length is the",
        "corrupt_prompt": "The metric unit for mass is the",
        "clean_target": " meter",
        "corrupt_target": " gram",
    },
    # --- Literature (authors) ---
    {
        "category": "literature_authors",
        "clean_prompt": "The epic Iliad was written by",
        "corrupt_prompt": "The epic Aeneid was written by",
        "clean_target": " Homer",
        "corrupt_target": " Virgil",
    },
    {
        "category": "literature_authors",
        "clean_prompt": "The novel Frankenstein was written by",
        "corrupt_prompt": "The novel Dracula was written by",
        "clean_target": " Shelley",
        "corrupt_target": " Stoker",
    },
    {
        "category": "literature_authors",
        "clean_prompt": "The novel Moby Dick was written by",
        "corrupt_prompt": "The novel The Odyssey was written by",
        "clean_target": " Melville",
        "corrupt_target": " Homer",
    },
    # --- Mythology (Roman vs Greek equivalents) ---
    {
        "category": "mythology_roman_greek",
        "clean_prompt": "The Roman god Mars is the Greek god",
        "corrupt_prompt": "The Roman god Jupiter is the Greek god",
        "clean_target": " Ares",
        "corrupt_target": " Zeus",
    },
    {
        "category": "mythology_roman_greek",
        "clean_prompt": "The Roman god Mercury matches the Greek god",
        "corrupt_prompt": "The Roman god Neptune matches the Greek god",
        "clean_target": " Hermes",
        "corrupt_target": " Poseidon",
    },
    {
        "category": "mythology_roman_greek",
        "clean_prompt": "The Roman goddess Venus matches the Greek goddess",
        "corrupt_prompt": "The Roman goddess Diana matches the Greek goddess",
        "clean_target": " Aphrodite",
        "corrupt_target": " Artemis",
    },
    # --- Pop culture (bands / genres) ---
    {
        "category": "pop_culture_bands",
        "clean_prompt": "The band Nirvana was associated strongly with",
        "corrupt_prompt": "The band Metallica was associated strongly with",
        "clean_target": " grunge",
        "corrupt_target": " metal",
    },
    {
        "category": "pop_culture_bands",
        "clean_prompt": "The band Beatles hailed originally from",
        "corrupt_prompt": "The band Eagles hailed originally from",
        "clean_target": " Liverpool",
        "corrupt_target": " California",
    },
    {
        "category": "pop_culture_bands",
        "clean_prompt": "The rock band U2 formed in the city of",
        "corrupt_prompt": "The rock band REM formed in the city of",
        "clean_target": " Dublin",
        "corrupt_target": " Athens",
    },
    # --- Materials ---
    {
        "category": "materials_composition",
        "clean_prompt": "Common drinking glasses are made of",
        "corrupt_prompt": "Common plastic bottles are made of",
        "clean_target": " glass",
        "corrupt_target": " plastic",
    },
    {
        "category": "materials_composition",
        "clean_prompt": "Most consumer aluminum foil is made from",
        "corrupt_prompt": "Most consumer plastic wrap is made from",
        "clean_target": " aluminum",
        "corrupt_target": " plastic",
    },
    {
        "category": "materials_composition",
        "clean_prompt": "Steel beams in construction are mostly made of",
        "corrupt_prompt": "Copper wires in circuits are mostly made of",
        "clean_target": " steel",
        "corrupt_target": " copper",
    },
]


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
