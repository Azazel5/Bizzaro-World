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
# Writes experiment_{MODE}.json (e.g. experiment_A.json).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

MODE="${MODE:-A}"
python3 scripts/exp1.py --mode "$MODE"
