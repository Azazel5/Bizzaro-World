# Bizarro World: Fact-Retrieval Circuits in Gemma 2B

## Abstract

This repository supports **mechanistic interpretability** experiments on **`google/gemma-2b`** using **TransformerLens**. The scientific objective is to **map and localize fact-retrieval circuitry**—components of the network whose activations causally support veridical completions on simple, high-coverage factual prompts—using **activation patching** (causal tracing) under **controlled distributional shifts**. All interventions are designed around **structurally aligned prompt pairs** so that differences in behavior can be attributed to **targeted internal states** rather than confounding token-position or syntactic drift. The engineering target is reproducible execution on **university HPC** (Slurm-managed GPUs, tight quotas, gated Hugging Face weights), with numerically stable evaluation metrics suitable for **fp16** forward passes at large vocabulary size.

---

## The “Bizarro” Methodology (Core Pivot)

### What we no longer assume

An earlier framing suggested that a “Bizarro” experimental condition should be implemented **in the surface text**: explicitly counterfactual or whimsical prompts (e.g., narrative rules that force geometric contradictions). That design is useful for **behavioral** demonstrations of in-context override, but it is a **weak** substrate for **circuit identification**.

### What we now treat as normative

In mechanistic interpretability, the operative contrast is best implemented **in latent space**, not by injecting gratuitous stylistic weirdness into the prompt.

1. **Clean run.** The model receives a **factually normal**, **minimal** stem that elicits a well-defined completion distribution (e.g., *The capital of France is* → **Paris**).

2. **Corrupt run (aligned).** The model receives a **parallel** stem that preserves **grammar, function words, and token-length alignment** while swapping only the **subject entity** (e.g., *The capital of Spain is* → **Madrid**).

3. **“Bizarro” as an intervention, not a genre.** The “Bizarro” condition is realized by **freezing the clean forward pass as a reference** and **surgically patching activations** from the corrupt run into the clean run’s **residual stream** (or selected submodules), then measuring how logits shift. That is the sense in which we **force** the model toward an alternate factual completion **mechanistically** while the **prompt text** remains ordinary.

### Why structural alignment is mandatory

Activation patching and causal tracing require **positional correspondence** between runs. If clean and corrupt prompts differ in **token length** or **syntactic skeleton**, then differences in hidden states mix **(i)** semantic fact content, **(ii)** low-level positional encodings, and **(iii)** trivial tokenizer artifacts. Those confounds make it difficult to interpret a patch as targeting a **fact circuit** rather than a **length / syntax** compensator.

Accordingly, this project enforces:

- **Token-aligned prefixes** between clean and corrupt prompts (verified under the Gemma tokenizer at evaluation time).
- **Single-token continuation targets** (with leading-space continuations where appropriate for SentencePiece-style models), so next-token probabilities and entropies are read from a **well-defined scalar** channel.

### Why explicit “context override” prompts are insufficient for circuit discovery

Narrative overrides can dominate the **late distribution** through shallow **pattern-completion** channels without isolating **which internal edges** carry factual content. Aligned factual pairs keep the **task** constant (“complete a capital / symbol / unit …”) while changing only the **binding** (France vs Spain). That raises the experimental **signal-to-noise ratio** for patching: interventions are interpreted relative to a **matched syntactic scaffold**, which is the standard of care in rigorous activation patching work.

---

## Experimental Pipeline

### Phase 1 — The Fact Battery

We curate **`fact_battery.json`**: a list of dictionaries, each specifying

- `category` (e.g., geography, chemistry, units),
- `clean_prompt` / `corrupt_prompt` (**token-length–matched** under Gemma),
- `clean_target` / `corrupt_target` (**single-token** continuations).

The battery spans **twenty** lightweight knowledge domains with **three** aligned pairs per domain (**60** pairs total). The file is **data**, not code: edit the JSON to extend or ablate conditions without touching the evaluation logic.

### Phase 2 — Baseline Evaluation (Behavioral Layer)

The driver script loads Gemma via TransformerLens and, for every battery entry:

- computes **\(P(\text{clean\_target} \mid \text{clean\_prompt})\)** and **\(P(\text{corrupt\_target} \mid \text{corrupt\_prompt})\)** at the final prompt position;
- reports the **probability delta** \(\Delta P = \lvert P_{\text{clean}} - P_{\text{corrupt}}\rvert\) as a **ranking key** for which contrasts exhibit the strongest *behavioral* divergence before any intervention;
- computes **Shannon entropy** of the full next-token distribution as a coarse **epistemic friction** signal.

**Numerical stability.** At fp16 precision and large vocabulary size, the naive pipeline `softmax → log` produces **\(-\infty\)** probabilities and **NaN** entropies. The implementation uses **`torch.log_softmax` in fp32** (log-sum-exp) and forms **\(-\sum p \log p\)** from stable log-probabilities, which removes the dominant source of entropy collapse artifacts in baseline tables.

### Phase 3 — Activation Patching (Causal Tracing)

The behavioral phases select **high-\(\Delta P\)** pairs as **priority interventions**. Phase 3 (implemented in forthcoming notebooks or modules) will:

- cache activations on clean vs corrupt forwards;
- patch **residual stream** (or attention/MLP channels) **position-aligned** components from corrupt into clean;
- attribute **logit changes** on the clean target token to **specific layers and heads**.

TransformerLens provides the **hook points** required for this program; the fact battery supplies **audited, aligned** counterfactuals.

---

## ML Systems Engineering (Infrastructure)

Factored models and hook-heavy runs are **memory-sensitive** even at **2B** parameters: weight streaming, optimizer states (when training is added), activation caches, and Hugging Face snapshot directories can exceed **consumer laptop** comfort. This project is therefore **engineered for university HPC**:

- **Slurm** allocations (`srun` / `sbatch`) request explicit **GPU**, **CPU RAM**, and **walltime**; preemptible queues are acceptable for development, with **higher memory** requests for shard loading peaks.
- **Disk quotas** (order **tens of GB** on some clusters) require **`HF_HOME`** (or equivalent cache roots) on **scratch** or project storage, not on small home volumes.
- **Containerized PyTorch modules** (e.g., NGC/Singularity stacks) may **strip environment variables**; Hugging Face authentication may require **`SINGULARITYENV_HF_TOKEN`**, **`huggingface-cli login`**, or site-specific forwarding—tokens should never be committed.
- **VRAM** is managed with **fp16** inference where supported; entropy and probability reads use **fp32 softmax math** on logits to avoid silent **NaNs** in metrics.

The canonical entrypoint for Phase 1–2 evaluation is:

```bash
python behavioral_friction_gemma2b.py
```

Ensure **`fact_battery.json`** sits beside that script (or call `load_fact_battery(path=...)` from another module).

---

## Repository Layout

| Path | Role |
|------|------|
| `fact_battery.json` | Curated aligned prompt pairs (Phase 1 data). |
| `behavioral_friction_gemma2b.py` | Model load, validation, baseline metrics, ranked \(\Delta P\) table. |
| `behavioral_friction_gemma2b_colab.ipynb` | Optional Colab-oriented workflow (legacy / exploratory). |

---

## References (directional)

Mechanistic interpretability and activation patching follow the broader literature on **causal scrubbing**, **path patching**, and **attribution graphs**; see work by Anthropic and NEEL for methodological precedent. Fine-grained factual circuits in small LMs are an active research area; this repository keeps claims **empirical** and **replication-first**.
