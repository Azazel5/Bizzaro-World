# Bizzaro-World

Utilizing principles of mechanistic interpretability to determine if mathematically, conceptually, and philosophically if transformers actually "know" things. Which we know they don't if we look at the architecture. However, a mathematical proof needs something actually grounded, so let's explore/build/write that!

# Context

Mechanistic Interpretability
: By studying the connections between neurons, we can find meaningful algorithms in the weights of neural networks.

Large language models are, well, large. It is impossible, as we've learnt in the CS 150 Special Topics class, to make sense of anything manually. And here's where the field of mechanistic interpreability arises. Can we study anything in the attention heads? Can we find circuits that are responsible for basic facts of the world?

Haugeland’s critique, particularly in **Artificial Intelligence: The Very Idea (1985)**, suggests that true intelligence requires being invested in one's own existence, or "giving a damn."

1. **Lack of Care**: Computers and LLMs (Large Language Models) manipulate symbols efficiently but lack any stake in the outcomes, consequences, or real-world risks, making their "understanding" merely technical rather than existential.
2. **Embodiment**: Genuine, human-like intelligence requires being embedded in a social and practical world where one is "concerned" about what happens next, a concept that AI lacks.

---

# Setup

This is the initial version of this project which I will attempt to run on my local computer, with only 10GBs of free space. For this, I have picked the Gemma-2b model, which should fit within those constraints, and this is a good model for the proof of concept and initial experimentations.

I'll be using the quite nice TransformerLens library written by an ex-Anthropic interpretability team person. Big shoutout for that! 

I will design "Bizzaro World" where gravity goes up, the shape of a basketball is cubic, or other things that will completely break the rules of what we know. Physics, basic facts, etc. I'll create two sets of prompts where one of meant to confuse the LLM. Doing so, we'll be able to measure entropy, confidence, etc using TransformerLens. We'll try to find both the logit entropy at the final output layer is a behavioral observation, but also a mechanistic one to proves that the model is great at In-Context Learning (ICL), but it doesn't map the internal conflict between "Truth" and "Context."

---

We'll draw heavy inspiration from methods described in the paper [Fine-Tuning Enhances Existing Mechanisms:
a case study on entity tracking](https://arxiv.org/pdf/2402.14811)

To prove it mathematically using the methodologies from the paper, we'll need to find the specific subnetworks (circuits) fighting each other. We need to pit the Factual Recall Circuit against the In-Context Learning / Induction Heads. 

**Hypothesis**: eventually the entropy that the relevant transformer heads we'll find will converge in both set of prompts, which is strange because entropy, the uncertainty, should go up and confidence should go down on the model in the BizzaroWorld prompts because it goes completely against the rules of the normal world. However, I hypothesize that it will not, showcasing that Transformers don't actually KNOW anything. In the architecture there's nothing there that shows "skin in the game", as Haugeland mentioned, as it is only a pattern recognizer. 

---

# Experimental Design: Activation Patching

To prove the thesis that Transformers lack semantic commitment (judgment) and only do syntax-matching (reckoning), we'll structure the experiment using Activation Patching.


1. Step 1: Locate the Facts. Feed the model a standard factual prompt: "The shape of a standard basketball is a". Use causal tracing to find the late-layer MLP nodes or attention heads that inject the concept of "sphere" or "round".

2. Step 2: Create the Bizarro Conflict. Feed the model the corrupted prompt: "In Bizarro World, everything round becomes a cube. The shape of a standard basketball is a".

3. Step 3: Patch and Measure. The model will likely output "cube". Now, take the activation from the "Truth/Factual" head we found in Step 1, and forcefully inject (patch) it into the Bizarro World forward pass.

4. Step 4: The Epistemic Measurement. Does patching the "Truth" override the Bizarro context? Or do the model's late-layer attention heads simply suppress the factual recall to minimize the cross-entropy loss of the current prompt?

If we can map the exact layer where the model's internal representation of "basketball = sphere" is mathematically zeroed out and overwritten by "basketball = cube" just because the prompt said so, we have found mechanistic proof of Haugeland's theory. We have physically mapped the lack of "epistemic friction." The model doesn't care about the truth of the basketball; it only cares about the gradient of the Bizarro rule.
