## `fact_battery/`

This folder holds **model-specific** fact batteries.

Why:
- tokenization constraints (prompt-length alignment, single-token targets) differ by model/tokenizer
- patching assumes position alignment, so we validate per tokenizer and drop incompatible rows

Files:
- `gemma-2b.json`: the original Gemma-aligned battery
- `llama3-70b.json`: battery filtered to satisfy LLaMA-3 tokenizer constraints

To (re)generate the LLaMA-3 battery:

```bash
python shared/build_llama3_fact_battery.py
```
