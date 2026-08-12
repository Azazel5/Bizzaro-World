#!/usr/bin/env python3
"""
AR faithfulness scoring: feed each AV-generated text description back through
the corresponding AR (activation reconstructor) checkpoint to get a
reconstructed activation vector, and score it against the real ground-truth
activation extracted in Phase A of the AV pipeline.

Read-only consumer of results/activations_{model_key}.pt and
results/av_descriptions_{model_key}.json -- does not touch nla_av_extraction.py
or its outputs.

--- The scale question: RESOLVED, not empirically diagnosed ---
The task this was written against asked whether AR's raw output vector comes
out in "raw" activation scale or "injection_scaled" space, to decide whether
to rescale the ground-truth vector before comparing. That framing doesn't
match how this actually needs to work, and guessing between those two options
empirically would be fragile. What actually resolves it, verified from the
real kitft/nla-inference source (fetched and read earlier this session,
before it was removed from this repo when the AV pipeline was simplified off
SGLang -- see nla_av_extraction.py's docstring for that story):

  - cosine_similarity needs no scale resolution AT ALL. Rescaling either
    vector by a positive scalar cannot change the angle between them, so it's
    computed directly on the raw vectors, always valid, regardless of what
    raw scale AR happens to output at.
  - MSE-family metrics DO need a shared reference scale to mean anything, and
    kitft's own reference implementation (NLACritic.score()) sidesteps the
    raw-vs-injection-scale question entirely: it independently L2-renormalizes
    BOTH the reconstruction and the ground truth to `mse_scale` (== sqrt(d_model),
    read straight from this checkpoint's own nla_meta.yaml -- 61.9677 for 12B,
    73.3212 for 27B, both confirmed against d_model in this session's earlier
    diagnostic) before computing MSE. It doesn't matter what raw scale either
    vector starts at; both land at the same norm before comparison. Under this
    construction MSE = 2*(1-cosine) exactly, so normalized_mse is reported as
    a consistency check on cosine, not an independent signal -- and it's
    directly comparable to kitft's own published FVE baseline constants (seen
    in their examples/ transcripts: 0.0302 for 12B, 0.0579 for 27B; FVE =
    1 - normalized_mse/const, not computed here but derivable from this output).

This is why `output_scale_diagnosis` in the saved JSON is a descriptive string
naming the resolution actually used, not a forced "raw"/"injection_scaled"/
"unresolved" enum -- picking one of those three would misrepresent what's
happening. Raw AR-output norms are still printed as an informational
diagnostic (see the "sampling first 3 AR reconstructions" block below), just
not used to gate the scoring methodology.

--- AR architecture, verified from the real source ---
AR's job is text -> vector, the reverse of AV's vector -> text, and its
loading/readout mechanism is NOT model.generate():
  - The checkpoint's own config.json already has num_hidden_layers truncated
    to K+1 (K = extraction layer -- confirmed empirically this session: 12B
    AR reports num_hidden_layers=33 for K=32). AutoModelForCausalLM.from_pretrained
    naturally loads only that many layers -- no manual truncation needed here.
  - lm_head and the final LayerNorm/RMSNorm are replaced with Identity: the
    value head reads the RAW residual-stream output of block K, not a
    normed or logit-projected version.
  - A separately-shipped value_head.safetensors (Linear(d_model, d_model),
    no bias) sits on top of the last token's hidden state -- NOT part of the
    main sharded model weights, fetched and loaded independently.
  - Prompts are the AR's own raw-string template (verified against the live
    sidecar, NOT assumed -- the guessed template in the original task spec,
    f"<text>{{d}}</text> <summary>", is missing a real prefix: the actual
    template is 'Summary of the following text: <text>{{explanation}}</text> <summary>'),
    tokenized with add_special_tokens=True (BOS-prefixed), NOT chat-templated
    the way AV's injection prompt was.

Checkpoints (sizes confirmed via HfApi.model_info(files_metadata=True) this
session, no download needed to check):
    12B: kitft/nla-gemma3-12b-L32-ar  (16.87 GB, 33 layers)
    27B: kitft/nla-gemma3-27b-L41-ar  (37.60 GB, 42 layers) -- size- and
         disk-margin-gated below (check_repo_size_before_download), given the
         AV checkpoint's disk blowup earlier this session. Neither AR is as
         large as the ~54GB full base/AV models, but 27B's AR is still
         substantial enough to check before committing.

Usage:
    python nla_ar_faithfulness.py --model gemma_12b_L32
    python nla_ar_faithfulness.py --model gemma_27b_L41   # gated by disk check
    python nla_ar_faithfulness.py --model gemma_12b_L32 --max_download_gb 20

Only AR needs to be loaded here -- ground truth and AV descriptions are
already on disk from prior runs.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch as t
import torch.nn.functional as F
import yaml
from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "gemma_12b_L32": {
        "ar_repo": "kitft/nla-gemma3-12b-L32-ar",
        "expected_d_model": 3840,
        "target_layer": 32,
    },
    "gemma_27b_L41": {
        "ar_repo": "kitft/nla-gemma3-27b-L41-ar",
        "expected_d_model": 5376,
        "target_layer": 41,
    },
}

_FINAL_LN_ATTRS = ("norm", "final_layernorm", "ln_f")
DEFAULT_MAX_DOWNLOAD_GB = 40.0
DEFAULT_MIN_FREE_DISK_GB = 20.0  # margin left over after the download


# ---------------------------------------------------------------------------
# Disk safety (same helpers as nla_av_extraction.py -- duplicated rather than
# imported, matching this project's convention of independently-runnable
# pipeline scripts, e.g. sae_differential_features.py / _attn_out.py)
# ---------------------------------------------------------------------------

def print_disk_usage(label: str) -> None:
    try:
        cache_info = scan_cache_dir()
    except Exception as e:
        print(f"[disk:{label}] could not scan HF cache: {type(e).__name__}: {e}", flush=True)
        return
    total = sum(repo.size_on_disk for repo in cache_info.repos)
    print(f"[disk:{label}] HF cache total: {total / 1e9:.1f} GB across {len(cache_info.repos)} repos:", flush=True)
    for repo in sorted(cache_info.repos, key=lambda r: -r.size_on_disk):
        print(f"    {repo.repo_id}: {repo.size_on_disk / 1e9:.2f} GB", flush=True)


def cleanup_hf_cache(repo_id: str) -> None:
    try:
        cache_info = scan_cache_dir()
    except Exception as e:
        print(f"[cleanup] could not scan HF cache: {type(e).__name__}: {e}", flush=True)
        return
    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            size_gb = repo.size_on_disk / 1e9
            shutil.rmtree(repo.repo_path, ignore_errors=True)
            print(f"[cleanup] removed {repo.repo_id} cache ({size_gb:.1f} GB freed)", flush=True)
            return
    print(f"[cleanup] {repo_id} not in HF cache -- nothing to remove", flush=True)


def check_repo_size_before_download(repo_id: str, max_gb: float, min_free_gb: float) -> float:
    """Query file sizes via the Hub API -- NOTHING is downloaded to check this.
    Refuses (raises) rather than silently attempting a download that might not
    fit, given the AV checkpoint's disk blowup earlier this session."""
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    total_gb = sum((f.size or 0) for f in info.siblings) / 1e9
    print(f"[disk] {repo_id} reports {total_gb:.2f} GB across {len(info.siblings)} files "
          f"(queried via Hub API metadata, nothing downloaded yet)", flush=True)

    free_gb = shutil.disk_usage(SCRIPT_DIR).free / 1e9
    print(f"[disk] {free_gb:.1f} GB currently free locally", flush=True)

    if total_gb > max_gb:
        raise RuntimeError(
            f"{repo_id} is {total_gb:.1f} GB, over the {max_gb:.1f} GB safety cap "
            f"(--max_download_gb to override). Refusing to download without an "
            f"explicit override, given the AV checkpoint's disk blowup earlier this session."
        )
    margin = free_gb - total_gb
    if margin < min_free_gb:
        raise RuntimeError(
            f"{free_gb:.1f} GB free, {total_gb:.1f} GB needed -- only {margin:.1f} GB "
            f"margin left, below the {min_free_gb:.1f} GB required. Free disk space first "
            f"(nla_av_extraction.py's cleanup_hf_cache() pattern, or check for leftover "
            f"caches with `df -h` / print_disk_usage() above) before retrying."
        )
    print(f"[disk] size check OK: {total_gb:.1f} GB download, {margin:.1f} GB margin after", flush=True)
    return total_gb


# ---------------------------------------------------------------------------
# AR config + reconstructor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ARConfig:
    d_model: int
    template: str
    mse_scale: float


def load_ar_config(ar_repo: str) -> ARConfig:
    """Fetch nla_meta.yaml and pull the template/scale verbatim -- do not
    assume the template string, verify it (see module docstring: the naive
    guess is missing a real prefix)."""
    meta_path = hf_hub_download(repo_id=ar_repo, filename="nla_meta.yaml")
    meta = yaml.safe_load(Path(meta_path).read_text())

    assert meta.get("role") in ("critic", "ar"), (
        f"{ar_repo} sidecar role={meta.get('role')!r}, expected 'critic' or 'ar' "
        f"-- pointed at the wrong checkpoint?"
    )
    mse_scale = meta.get("extraction", {}).get("mse_scale")
    assert mse_scale is not None, f"{ar_repo}'s sidecar has no extraction.mse_scale"

    template = meta["prompt_templates"].get("ar") or meta["prompt_templates"].get("critic")
    assert template is not None, f"{ar_repo}'s sidecar has no 'ar' or 'critic' prompt template"

    print(f"[ar-config] template (verbatim from sidecar): {template!r}", flush=True)
    print(f"[ar-config] mse_scale={float(mse_scale):.4f} (sqrt(d_model)={meta['d_model'] ** 0.5:.4f})", flush=True)

    return ARConfig(d_model=meta["d_model"], template=template, mse_scale=float(mse_scale))


class ARReconstructor:
    """Text -> reconstructed activation vector."""

    def __init__(self, ar_repo: str, cfg: ARConfig, device: str, dtype: t.dtype = t.bfloat16):
        self.cfg = cfg
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(ar_repo, trust_remote_code=True)

        # BOS invariant (matches real NLACritic source): critic prompts are a
        # raw string tokenized with add_special_tokens=True, not chat-templated.
        probe = self.tokenizer("x", add_special_tokens=True)["input_ids"]
        bos = self.tokenizer.bos_token_id
        assert bos is None or probe[0] == bos, (
            f"tokenizer bos_token_id={bos} but add_special_tokens=True gave first "
            f"token {probe[0]} -- BOS-prefix invariant broken, reconstruct() assumes it holds."
        )

        print(f"[ar] loading backbone {ar_repo} ({dtype}, {device})...", flush=True)
        backbone = AutoModelForCausalLM.from_pretrained(ar_repo, dtype=dtype, trust_remote_code=True)
        print(f"[ar] backbone has {backbone.config.num_hidden_layers} layers "
              f"(already truncated to K+1 in this checkpoint's own config.json)", flush=True)

        backbone.lm_head = t.nn.Identity()
        inner = backbone.model
        for attr in _FINAL_LN_ATTRS:
            if hasattr(inner, attr):
                setattr(inner, attr, t.nn.Identity())
                print(f"[ar] stripped final-norm attribute {attr!r} (value head reads raw resid-stream output)", flush=True)
                break
        else:
            raise AssertionError(f"no final-norm attribute on {type(inner).__name__} -- tried {_FINAL_LN_ATTRS!r}")

        d = backbone.config.hidden_size
        assert d == cfg.d_model, f"backbone hidden_size={d} != sidecar d_model={cfg.d_model}"

        value_head_path = hf_hub_download(repo_id=ar_repo, filename="value_head.safetensors")
        value_head = t.nn.Linear(d, d, bias=False, dtype=dtype)
        value_head.load_state_dict(load_file(value_head_path))
        print(f"[ar] loaded value_head.safetensors ({d}x{d} linear, no bias)", flush=True)

        self.backbone = backbone.to(device).eval()
        self.value_head = value_head.to(device).eval()
        print("[ar] model ready", flush=True)

    @t.inference_mode()
    def reconstruct(self, description: str) -> t.Tensor:
        prompt = self.cfg.template.format(explanation=description)
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(self.device)
        h = self.backbone.model(ids, use_cache=False).last_hidden_state[0, -1]
        return self.value_head(h).float().cpu()


# ---------------------------------------------------------------------------
# Faithfulness scoring
# ---------------------------------------------------------------------------

def faithfulness_score(v_true: t.Tensor, v_recon: t.Tensor, mse_scale: float) -> dict[str, float]:
    """cosine_similarity: computed on raw vectors, scale-invariant, always
    valid regardless of AR's raw output scale. mse: raw, NOT meaningful alone
    (different raw scales) -- reported for transparency only. normalized_mse:
    both vectors independently L2-renormalized to mse_scale before MSE, per
    kitft's own NLACritic.score() methodology -- this is the meaningful
    secondary metric, and equals 2*(1-cosine_similarity) by construction."""
    v_true = v_true.float()
    v_recon = v_recon.float()

    cosine_sim = F.cosine_similarity(v_true.unsqueeze(0), v_recon.unsqueeze(0)).item()
    raw_mse = F.mse_loss(v_true, v_recon).item()

    v_true_n = v_true / v_true.norm().clamp_min(1e-12) * mse_scale
    v_recon_n = v_recon / v_recon.norm().clamp_min(1e-12) * mse_scale
    normalized_mse = F.mse_loss(v_true_n, v_recon_n).item()

    return {"cosine_similarity": cosine_sim, "mse": raw_mse, "normalized_mse": normalized_mse}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_model(
    model_key: str,
    results_dir: Path,
    device: str,
    max_download_gb: float,
    min_free_disk_gb: float,
) -> None:
    config = MODEL_CONFIGS[model_key]
    activations_path = results_dir / f"activations_{model_key}.pt"
    av_json_path = results_dir / f"av_descriptions_{model_key}.json"

    print(f"\n{'#' * 60}")
    print(f"# NLA AR faithfulness scoring: {model_key}")
    print(f"{'#' * 60}\n", flush=True)
    print_disk_usage("start")

    if not activations_path.exists():
        raise FileNotFoundError(
            f"missing {activations_path} -- run nla_av_extraction.py (with or without "
            f"--extract-only) first to produce it."
        )
    if not av_json_path.exists():
        raise FileNotFoundError(
            f"missing {av_json_path} -- run nla_av_extraction.py first to produce it."
        )

    saved = t.load(activations_path)
    clean_acts, corrupt_acts = saved["clean_acts"], saved["corrupt_acts"]
    av_data = json.loads(av_json_path.read_text())
    records_in = av_data["records"]

    print(f"[load] activations: clean={tuple(clean_acts.shape)} corrupt={tuple(corrupt_acts.shape)}", flush=True)
    print(f"[load] av_descriptions: {len(records_in)} records", flush=True)

    # Alignment check -- fail loudly rather than silently mismatch. The AV
    # script's decode loop can, in principle, skip a failed pair (try/except:
    # continue), which would both shrink records_in below len(clean_acts) AND
    # break idx==position alignment for everything after the skip.
    assert len(records_in) == clean_acts.shape[0] == corrupt_acts.shape[0], (
        f"record count mismatch: av_descriptions has {len(records_in)} records, "
        f"activations has {clean_acts.shape[0]} clean / {corrupt_acts.shape[0]} corrupt. "
        f"Stopping -- proceeding would align the wrong description to the wrong vector."
    )
    for i, rec in enumerate(records_in):
        assert rec["idx"] == i, (
            f"records[{i}]['idx']={rec['idx']} != {i} -- av_descriptions has a gap "
            f"(a skipped pair), so position-based alignment with activations_*.pt "
            f"(indexed by original battery position) is no longer valid. Stopping."
        )
    print(f"[load] alignment confirmed: {len(records_in)} records, idx == position for all", flush=True)

    clean_norms = clean_acts.norm(dim=-1)
    corrupt_norms = corrupt_acts.norm(dim=-1)
    print(f"[diag] ground-truth clean_acts norms: min={clean_norms.min():.1f} "
          f"max={clean_norms.max():.1f} mean={clean_norms.mean():.1f}", flush=True)
    print(f"[diag] ground-truth corrupt_acts norms: min={corrupt_norms.min():.1f} "
          f"max={corrupt_norms.max():.1f} mean={corrupt_norms.mean():.1f}", flush=True)

    # --- size check, then load AR ---
    check_repo_size_before_download(config["ar_repo"], max_download_gb, min_free_disk_gb)
    ar_cfg = load_ar_config(config["ar_repo"])
    assert ar_cfg.d_model == config["expected_d_model"], (
        f"d_model={ar_cfg.d_model} != expected {config['expected_d_model']} -- checkpoint drift, stopping."
    )

    scale_diagnosis = (
        f"mutual_renormalization_to_mse_scale={ar_cfg.mse_scale:.4f} (sqrt(d_model), "
        f"from this checkpoint's own nla_meta.yaml) -- matches kitft's own NLACritic.score() "
        f"methodology. cosine_similarity is scale-invariant and needs no such resolution."
    )
    print(f"[diag] scale resolution: {scale_diagnosis}", flush=True)

    ar = ARReconstructor(config["ar_repo"], ar_cfg, device)
    print_disk_usage("after AR load")

    print("\n[diag] sampling first 3 AR reconstructions' raw output norms "
          "(informational only -- faithfulness_score() does not depend on this):", flush=True)
    for i in range(min(3, len(records_in))):
        v = ar.reconstruct(records_in[i]["clean_description"])
        print(f"    pair {i}: AR raw output norm={v.norm().item():.1f}  "
              f"(vs ground-truth clean norm={clean_acts[i].norm().item():.1f}, "
              f"vs mse_scale={ar_cfg.mse_scale:.1f})", flush=True)

    # --- reconstruct + score every pair ---
    out_records: list[dict[str, Any]] = []
    n = len(records_in)
    for i, rec in enumerate(records_in):
        print(f"[{i + 1}/{n}] reconstructing pair {i} ({rec.get('category')})...", flush=True)
        clean_recon = ar.reconstruct(rec["clean_description"])
        corrupt_recon = ar.reconstruct(rec["corrupt_description"])

        clean_scores = faithfulness_score(clean_acts[i], clean_recon, ar_cfg.mse_scale)
        corrupt_scores = faithfulness_score(corrupt_acts[i], corrupt_recon, ar_cfg.mse_scale)

        out_records.append({
            "idx": i,
            "category": rec.get("category"),
            "clean_cosine_similarity": clean_scores["cosine_similarity"],
            "clean_mse": clean_scores["mse"],
            "clean_normalized_mse": clean_scores["normalized_mse"],
            "corrupt_cosine_similarity": corrupt_scores["cosine_similarity"],
            "corrupt_mse": corrupt_scores["mse"],
            "corrupt_normalized_mse": corrupt_scores["normalized_mse"],
        })

    clean_cos = [r["clean_cosine_similarity"] for r in out_records]
    corrupt_cos = [r["corrupt_cosine_similarity"] for r in out_records]
    summary_stats = {
        "mean_clean_cosine": statistics.mean(clean_cos),
        "mean_corrupt_cosine": statistics.mean(corrupt_cos),
        "std_clean_cosine": statistics.pstdev(clean_cos) if len(clean_cos) > 1 else 0.0,
        "std_corrupt_cosine": statistics.pstdev(corrupt_cos) if len(corrupt_cos) > 1 else 0.0,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"ar_faithfulness_{model_key}.json"
    out_path.write_text(json.dumps({
        "model_key": model_key,
        "ar_repo": config["ar_repo"],
        "output_scale_diagnosis": scale_diagnosis,
        "n_pairs": n,
        "records": out_records,
        "summary_stats": summary_stats,
    }, indent=2) + "\n")
    print(f"\n[save] wrote {out_path}", flush=True)

    print("\n" + "=" * 78)
    print(f"{'idx':<4} {'category':<24} {'clean_cos':>10} {'clean_nmse':>11} {'corr_cos':>9} {'corr_nmse':>10}")
    print("-" * 78)
    for r in out_records:
        print(f"{r['idx']:<4} {str(r['category'])[:24]:<24} "
              f"{r['clean_cosine_similarity']:>10.4f} {r['clean_normalized_mse']:>11.4f} "
              f"{r['corrupt_cosine_similarity']:>9.4f} {r['corrupt_normalized_mse']:>10.4f}")
    print("-" * 78)
    print(f"mean clean cosine   = {summary_stats['mean_clean_cosine']:.4f}  (std {summary_stats['std_clean_cosine']:.4f})")
    print(f"mean corrupt cosine = {summary_stats['mean_corrupt_cosine']:.4f}  (std {summary_stats['std_corrupt_cosine']:.4f})")
    print("=" * 78, flush=True)

    del ar.backbone, ar.value_head, ar
    gc.collect()
    if device == "cuda":
        t.cuda.empty_cache()
    cleanup_hf_cache(config["ar_repo"])
    print_disk_usage("end of run")


def _resolve_device() -> str:
    if t.cuda.is_available():
        return "cuda"
    print("[warn] CUDA not available -- falling back to CPU. AR is much smaller than the "
          "base/AV models (truncated to K+1 layers), CPU may be practical here.", flush=True)
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="NLA AR faithfulness scoring.")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--results_dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--max_download_gb", type=float, default=DEFAULT_MAX_DOWNLOAD_GB,
                        help=f"Refuse to download an AR checkpoint larger than this "
                             f"(default {DEFAULT_MAX_DOWNLOAD_GB} GB).")
    parser.add_argument("--min_free_disk_gb", type=float, default=DEFAULT_MIN_FREE_DISK_GB,
                        help=f"Refuse to download unless at least this much disk margin "
                             f"remains afterward (default {DEFAULT_MIN_FREE_DISK_GB} GB).")
    args = parser.parse_args()

    device = _resolve_device()
    run_model(
        model_key=args.model,
        results_dir=args.results_dir,
        device=device,
        max_download_gb=args.max_download_gb,
        min_free_disk_gb=args.min_free_disk_gb,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
