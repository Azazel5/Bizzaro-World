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

### Phase 2 — Baseline evaluation (behavioral / patching triage)

The script **`behavioral_friction_gemma2b.py`** loads the model and, for **each** row, reads **raw next-token logits** at the **final** prompt position for both `clean_target` and `corrupt_target` token ids.

**Bidirectional logit difference (patching-style).** On each forward, define  
`LD = logit(clean_target) - logit(corrupt_target)` (computed in **fp32** on the logit vector).

- **`LD_clean`** — forward on the **clean** prompt. When the model “locks in” the clean fact, the clean token should beat the corrupt token: **large positive** `LD_clean`.
- **`LD_corrupt`** — forward on the **corrupt** prompt with the **same two token ids**. The corrupt token should win: `logit(corrupt) > logit(clean)`, so **large negative** `LD_corrupt` (because `LD` is still *clean minus corrupt*).

**TotalSwing (golden-pair screen).**  
`TotalSwing = LD_clean - LD_corrupt`.  
Subtracting a negative `LD_corrupt` **adds** magnitude when both legs are strong. Intuitively: you reward **both** a decisive clean-side margin *and* a decisive corrupt-side margin—the same two-token race **reverses** across the two aligned contexts. **Larger TotalSwing ⇒ better “golden pair”** for downstream activation patching (high signal before any hooks run).

**Sanity probabilities.** The console table and CSV include **`P(clean_target | clean_prompt)`** and **`P(corrupt_target | corrupt_prompt)`** from **softmax in fp32** (marginal checks that each world is confidently answered).

**Outputs.**

- **Console table** — sorted by **TotalSwing** descending. Columns: `rank`, `idx` (0-based row in `fact_battery.json`), `TotalSwing`, `LD_clean`, `LD_corrupt`, `P_clean`, `P_corrupt`, `category`, and a truncated `clean_prompt|corrupt_prompt` prefix.
- **`fact_battery_triage.csv`** — same sort order; full prompts and targets; numeric columns as strings with fixed precision for clean import into pandas or Sheets. Regenerated on every script run (listed in `.gitignore` so local runs do not dirty the tree unless you remove that line).

**Numerical stability.** Logit differences and softmax use **fp32** math on the last-position logit vector so large-vocabulary **fp16** runs are less likely to produce garbage probabilities for triage.

### Phase 3 — Activation patching (causal tracing)

Use **high total-swing** pairs from Phase 2 as **priority** for interventions (forthcoming notebooks or modules):

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
- **VRAM vs metrics** — Forwards often stay in **fp16**; triage metrics read logits / softmax in **fp32** to avoid silent **NaN** or collapsed probabilities at the reporting step.

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
| `behavioral_friction_gemma2b.py` | Load model, validate pairs, bidirectional **logit-difference** triage, **TotalSwing-ranked** console table + **`fact_battery_triage.csv`**. |
| `fact_battery_triage.csv` | **Generated** triage export (same directory as the script; gitignored by default). |
| `behavioral_friction_gemma2b_colab.ipynb` | Optional Colab-oriented notes (legacy / exploratory). |

---

## References (pointers)

Mechanistic interpretability and activation patching connect to **causal scrubbing**, **path patching**, and **attribution-style** analyses; Anthropic and NEEL-style writeups are good entry points. Claims in this repo are intentionally **modest** and **replication-oriented**—the tables are for **screening**, not for publishing conclusions by themselves.
