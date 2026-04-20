#!/bin/bash
#SBATCH -J exp1
#SBATCH -p preempt
#SBATCH -n 2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o exp1_%j.out
#SBATCH -e exp1_%j.err
#
# Submit from repo root. Optional: set MODE (default A).
#   sbatch slurm/run_exp1.slurm
#   sbatch --export=ALL,MODE=B slurm/run_exp1.slurm
#
# Writes to OUTDIR (default ./outputs):
#   OUTDIR/experiment_{MODE}.json

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

MODE="${MODE:-A}"
OUTDIR="${OUTDIR:-./outputs}"

module load pytorch/2.8.0-cuda12.9-cudnn9

mkdir -p "$OUTDIR"
python3 scripts/exp1.py --mode "$MODE" --outdir "$OUTDIR"
