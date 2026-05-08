#!/bin/bash
#SBATCH --job-name=gemma12b_smoke_exp1a
#SBATCH -p preempt
#SBATCH -n 2
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Gemma-12B smoke test: run only Exp1 mode A and verify output has no NaN/Inf.
#
# Configurable via --export:
#   TRIAGE_CSV  default: gemma-12b-it/triage/fact_battery_triage.csv
#   MODEL_ID    default: google/gemma-3-12b-it
#   OUTDIR      default: gemma-12b-it/phase1/A
#
# Auth:
#   export HF_TOKEN=... before sbatch (or pass HUGGINGFACE_HUB_TOKEN)

TRIAGE_CSV="${TRIAGE_CSV:-gemma-12b-it/triage/fact_battery_triage.csv}"
MODEL_ID="${MODEL_ID:-google/gemma-3-12b-it}"
OUTDIR="${OUTDIR:-gemma-12b-it/phase1/A}"

if command -v module >/dev/null 2>&1; then
  module load pytorch/2.8.0-cuda12.9-cudnn9
  if ! command -v python3 >/dev/null 2>&1; then
    module load python || true
  fi
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found on PATH." >&2
  exit 127
fi

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HF_HOME="${HF_HOME:-${SLURM_TMPDIR:-/tmp}/${USER}/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TMPDIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}/${USER}/tmp}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

mkdir -p "${TRANSFORMERS_CACHE}" "${TMPDIR}" "${OUTDIR}"

echo "==> Smoke test: exp1 mode A"
SCRIPT="scripts/experiments/exp1.py" MODE="A" \
TRIAGE="${TRIAGE_CSV}" MODEL="${MODEL_ID}" OUTDIR="${OUTDIR}" \
python3 - <<'PY'
import os
import sys
import importlib.util

script = os.environ["SCRIPT"]
mode = os.environ["MODE"]
outdir = os.environ["OUTDIR"]
triage = os.environ["TRIAGE"]
model = os.environ["MODEL"]

spec = importlib.util.spec_from_file_location("expmod", script)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

if hasattr(mod, "MODEL_NAME"):
    mod.MODEL_NAME = model

sys.argv = [script, "--mode", mode, "--triage-csv", triage, "--outdir", outdir]
mod.main()
PY

echo "==> Validating output JSON for NaN/Inf"
OUT_JSON="${OUTDIR}/experiment_A.json"
OUT_JSON="${OUT_JSON}" python3 - <<'PY'
import json
import math
import os
import sys

path = os.environ["OUT_JSON"]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)

def count_bad(x):
    if isinstance(x, float):
        return int(math.isnan(x) or math.isinf(x))
    if isinstance(x, list):
        return sum(count_bad(v) for v in x)
    if isinstance(x, dict):
        return sum(count_bad(v) for v in x.values())
    return 0

bad = count_bad(payload)
print(f"bad_values={bad}")
if bad:
    print(f"ERROR: non-finite values detected in {path}", file=sys.stderr)
    sys.exit(3)
print(f"Smoke test passed: {path}")
PY
