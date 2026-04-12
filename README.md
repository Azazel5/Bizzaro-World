# Bizarro World: Fact-retrieval circuits in Gemma 2B

## Abstract

This repository supports **mechanistic interpretability** work on **`google/gemma-2b`** with **TransformerLens**. The goal is to **map fact-related computation**: identify parts of the network whose activations support correct next-token predictions on simple, high-coverage factual prompts, using **activation patching** (causal tracing) under **controlled contrasts**.

All contrasts use **structurally aligned prompt pairs** so that observed differences are easier to tie to **internal representations** instead of length, punctuation, or tokenizer quirks. The stack is meant to run reliably on **university HPC** (Slurm, gated Hugging Face weights, tight disk quotas), with metrics that stay numerically sane under **fp16** inference over a large vocabulary.

---

## The “Bizarro” methodology (core pivot)

### What we moved away from

An older idea was that “Bizarro” should show up **in the prompt text**—long, whimsical counterfactuals (e.g. alternate physics spelled out in prose). That is fine for **behavioral** demos of in-context override, but it is a **weak** setup for **circuit discovery**: too many moving parts, and hard to align positions across runs.

### What we standardize on

In this project, the important contrast lives **in the model’s internal states**, not in baroque surface text.

1. **Clean run** — A short, ordinary factual stem (e.g. *The capital of France is* → **Paris**).

2. **Corrupt run (aligned)** — The **same template**, same role slots, **same token length under Gemma**, with only the **entity** swapped (e.g. *The capital of Spain is* → **Madrid**).

3. **“Bizarro” as patching, not prose** — We treat “Bizarro” as: **keep a clean forward pass as reference**, then **patch activations** from the corrupt forward into the clean run’s **residual stream** (or chosen submodules) and read off logit changes. The prompts stay **boring and factual**; the forced “wrong world” is **mechanical**.

### Why alignment matters

Patching needs **position-to-position** correspondence. If clean and corrupt strings differ in **token count** or **syntax**, hidden-state differences mix **meaning**, **position**, and **tokenizer noise**. This repo therefore enforces:

- **Matched prompt tokenization** (checked when you run the driver).
- **Single-token continuation targets** (usually with a **leading space**, SentencePiece-style), so each probability is a single scalar index into the next-token distribution.

### Why we do not rely on “story override” prompts for circuits

Narrative overrides can move the distribution through **shallow completion** paths without telling you **which components** stored the fact. Aligned factual pairs keep the **task** fixed (capitals, symbols, units, …) and change only the **binding** (France vs Spain). That improves **signal-to-noise** for later patching: you intervene on a **matched scaffold**, which is the usual standard in careful activation work.

---

## Experimental pipeline

### Phase 1 — Fact battery

Data live in **`fact_battery.json`**: a JSON array of objects with:

- `category`
- `clean_prompt` / `corrupt_prompt` (token-length matched for Gemma)
- `clean_target` / `corrupt_target` (each a **single** tokenizer token)

There are **20** thematic buckets × **3** pairs each (**60** rows). Editing the JSON changes experiments without editing Python.

### Phase 2 — Baseline evaluation (behavioral layer)

The script **`behavioral_friction_gemma2b.py`** loads the model and, for **each** row:

- **`P(clean_target | clean_prompt)`** — probability of the clean continuation at the **last** prompt position.
- **`P(corrupt_target | corrupt_prompt)`** — same for the corrupt line.
- **`ΔP`** — absolute gap: `ΔP = |P_clean - P_corrupt|`. Rows are **sorted by ΔP** so the largest **behavioral** splits float to the top (good candidates before any intervention).
- **Shannon entropy** of the full next-token distribution at that position — a coarse **spread / uncertainty** readout (“epistemic friction” in a loose sense).

**Numerical stability.** With **fp16** logits and a huge vocabulary, doing `softmax` then `log` on probabilities often blows up to **`-inf`** and **NaN** entropies. The code uses **`torch.log_softmax` in fp32** (log-sum-exp) and forms Shannon entropy as **minus the expected log-probability** (equivalently, minus the sum over the vocabulary of *p* log *p*) from those stable log-probabilities, which removes most of that failure mode.

### Phase 3 — Activation patching (causal tracing)

Use **high-ΔP** pairs from Phase 2 as **priority** for interventions (forthcoming notebooks or modules):

- Cache activations on **clean** vs **corrupt** forwards.
- Patch **position-aligned** residual (or attention / MLP) components from corrupt into clean.
- Attribute **logit shifts** on the clean target to **layers and heads**.

TransformerLens supplies **hooks**; the fact battery supplies **clean, aligned** counterfactuals.

---

## ML systems engineering (infrastructure)

Even at **2B** parameters, weights, caches, and optional activation stores add up. This repo assumes **cluster** workflows:

- **Slurm** — Request **GPU**, enough **CPU RAM** (model shard load can spike), and realistic **walltime**. Preempt queues are OK for dev if you checkpoint logs.
- **Disk** — Point **`HF_HOME`** (or equivalent) at **scratch** or project space when home quotas are small (~tens of GB).
- **Containers** — NGC / Singularity wrappers sometimes **drop** env vars. For gated models you may need **`SINGULARITYENV_HF_TOKEN`**, **`huggingface-cli login`**, or site docs. Never commit tokens.
- **VRAM vs metrics** — Forwards often stay in **fp16**; probability and entropy use **fp32** math on logits where it matters to avoid silent **NaN** tables.

**Run Phase 1–2:**

```bash
python behavioral_friction_gemma2b.py
```

Keep **`fact_battery.json`** next to that file (same folder), or pass a path into **`load_fact_battery(...)`** from your own code.

---

## Repository layout

| Path | Role |
|------|------|
| `fact_battery.json` | Aligned prompt pairs (Phase 1 data). |
| `behavioral_friction_gemma2b.py` | Load model, validate pairs, baseline metrics, **ΔP-ranked** table. |
| `behavioral_friction_gemma2b_colab.ipynb` | Optional Colab-oriented notes (legacy / exploratory). |

---

## References (pointers)

Mechanistic interpretability and activation patching connect to **causal scrubbing**, **path patching**, and **attribution-style** analyses; Anthropic and NEEL-style writeups are good entry points. Claims in this repo are intentionally **modest** and **replication-oriented**—the tables are for **screening**, not for publishing conclusions by themselves.
