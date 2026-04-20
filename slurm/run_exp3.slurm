#!/bin/bash
#SBATCH --job-name=exp3
#SBATCH -p preempt
#SBATCH -n 2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#
# Output paths should be passed at submission time (Slurm does not expand shell vars in #SBATCH).
# Example submit:
#   OUTDIR="experiments/experiment3_A"
#   mkdir -p "$OUTDIR"
#   sbatch \
#     --output="${OUTDIR}/exp3_%j.out" \
#     --error="${OUTDIR}/exp3_%j.err" \
#     --export=ALL,MODE=A,OUTDIR="${OUTDIR}" \
#     slurm/run_exp3.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

MODE="${MODE:-A}"
OUTDIR="${OUTDIR:-./outputs}"

module load pytorch/2.8.0-cuda12.9-cudnn9

mkdir -p "$OUTDIR"
python3 scripts/exp3.py --mode "$MODE" --outdir "$OUTDIR"

