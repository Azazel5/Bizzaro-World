#!/bin/bash
#SBATCH --job-name=bizzaro_exp
#SBATCH -p preempt
#SBATCH -n 2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#
# Generic experiment runner. Output/error paths must be passed at submission time
# (Slurm does not expand shell vars in #SBATCH directives).
#
# Required/optional environment variables (via --export):
#   MODE   : A/B/C (default A)
#   OUTDIR : output directory (default ./outputs)
#   SCRIPT : python entrypoint path (default scripts/experiments/exp1.py)
#
# Example:
#   OUTDIR="experiments/experiment3_A"
#   mkdir -p "$OUTDIR"
#   sbatch \
#     --output="${OUTDIR}/exp_%j.out" \
#     --error="${OUTDIR}/exp_%j.err" \
#     --export=ALL,MODE=A,OUTDIR="${OUTDIR}",SCRIPT="scripts/experiments/exp3.py" \
#     slurm/run_experiment.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

MODE="${MODE:-A}"
OUTDIR="${OUTDIR:-./outputs}"
SCRIPT="${SCRIPT:-scripts/experiments/exp1.py}"

if command -v module >/dev/null 2>&1; then
  module load pytorch/2.8.0-cuda12.9-cudnn9
fi

mkdir -p "$OUTDIR"
python3 "$SCRIPT" --mode "$MODE" --outdir "$OUTDIR"

