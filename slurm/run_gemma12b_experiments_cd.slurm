#!/bin/bash
#SBATCH --job-name=gemma12b_exp_cd
#SBATCH -p preempt
#SBATCH -n 2
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# This script can run the full Gemma-12B experiment suite.
# Experiment selectors (space-separated via EXPERIMENTS):
#   1 / EXP1        => scripts/experiments/exp1.py
#   2A / EXP2A      => scripts/experiments/exp2a.py
#   2B / EXP2B      => scripts/experiments/exp2b.py
#   3 / C / EXP3    => scripts/experiments/exp3.py
#   4 / D / EXP4    => scripts/experiments/exp4.py
#   ALL             => 1 2A 2B 3 4
#
# Configurable env vars via --export:
#   TRIAGE_CSV   default: gemma-12b-it/triage/fact_battery_triage.csv
#   MODEL_ID     default: google/gemma-3-12b-it
#   BATTERY_JSON default: fact_battery/gemma-3-12b-it.json
#   BASE_OUTDIR  default: gemma-12b-it
#   MODES        default: "A B C"
#   EXPERIMENTS  default: "ALL"
#
# Authentication:
#   export HF_TOKEN=... before sbatch
#   (or pass HUGGINGFACE_HUB_TOKEN via --export)

TRIAGE_CSV="${TRIAGE_CSV:-gemma-12b-it/triage/fact_battery_triage.csv}"
MODEL_ID="${MODEL_ID:-google/gemma-3-12b-it}"
BATTERY_JSON="${BATTERY_JSON:-fact_battery/gemma-3-12b-it.json}"
BASE_OUTDIR="${BASE_OUTDIR:-gemma-12b-it}"
MODES="${MODES:-A B C}"
EXPERIMENTS="${EXPERIMENTS:-ALL}"

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

mkdir -p "${TRANSFORMERS_CACHE}" "${TMPDIR}"

run_one() {
  local script="$1"
  local mode="$2"
  local outdir="$3"
  mkdir -p "${outdir}"
  echo "==> Running ${script} mode ${mode} -> ${outdir}"
  SCRIPT="${script}" MODE="${mode}" OUTDIR="${outdir}" \
  TRIAGE="${TRIAGE_CSV}" MODEL="${MODEL_ID}" BATTERY="${BATTERY_JSON}" \
  python3 - <<'PY'
import os
import sys
import importlib.util
from pathlib import Path

script = os.environ["SCRIPT"]
mode = os.environ["MODE"]
outdir = os.environ["OUTDIR"]
triage = os.environ["TRIAGE"]
model = os.environ["MODEL"]
battery = os.environ["BATTERY"]

spec = importlib.util.spec_from_file_location("expmod", script)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

if hasattr(mod, "MODEL_NAME"):
    mod.MODEL_NAME = model
if hasattr(mod, "BATTERY_PATH"):
    mod.BATTERY_PATH = Path(battery)

sys.argv = [script, "--mode", mode, "--triage-csv", triage, "--outdir", outdir]
mod.main()
PY
}

for exp in ${EXPERIMENTS}; do
  case "${exp^^}" in
    ALL)
      for m in ${MODES}; do
        run_one "scripts/experiments/exp1.py" "${m}" "${BASE_OUTDIR}/phase1/${m}"
      done
      for m in ${MODES}; do
        run_one "scripts/experiments/exp2a.py" "${m}" "${BASE_OUTDIR}/phase2/${m}/exp2a"
      done
      for m in ${MODES}; do
        run_one "scripts/experiments/exp2b.py" "${m}" "${BASE_OUTDIR}/phase2/${m}/exp2b"
      done
      for m in ${MODES}; do
        run_one "scripts/experiments/exp3.py" "${m}" "${BASE_OUTDIR}/phase3/${m}"
      done
      for m in ${MODES}; do
        run_one "scripts/experiments/exp4.py" "${m}" "${BASE_OUTDIR}/phase4/${m}"
      done
      ;;
    1|EXP1)
      for m in ${MODES}; do
        run_one "scripts/experiments/exp1.py" "${m}" "${BASE_OUTDIR}/phase1/${m}"
      done
      ;;
    2A|EXP2A)
      for m in ${MODES}; do
        run_one "scripts/experiments/exp2a.py" "${m}" "${BASE_OUTDIR}/phase2/${m}/exp2a"
      done
      ;;
    2B|EXP2B)
      for m in ${MODES}; do
        run_one "scripts/experiments/exp2b.py" "${m}" "${BASE_OUTDIR}/phase2/${m}/exp2b"
      done
      ;;
    C)
      for m in ${MODES}; do
        run_one "scripts/experiments/exp3.py" "${m}" "${BASE_OUTDIR}/phase3/${m}"
      done
      ;;
    D)
      for m in ${MODES}; do
        run_one "scripts/experiments/exp4.py" "${m}" "${BASE_OUTDIR}/phase4/${m}"
      done
      ;;
    3|EXP3)
      for m in ${MODES}; do
        run_one "scripts/experiments/exp3.py" "${m}" "${BASE_OUTDIR}/phase3/${m}"
      done
      ;;
    4|EXP4)
      for m in ${MODES}; do
        run_one "scripts/experiments/exp4.py" "${m}" "${BASE_OUTDIR}/phase4/${m}"
      done
      ;;
    *)
      echo "ERROR: unknown experiment selector '${exp}'. Use ALL, 1, 2A, 2B, 3/C, or 4/D." >&2
      exit 2
      ;;
  esac
done

echo "All requested experiment runs completed."
