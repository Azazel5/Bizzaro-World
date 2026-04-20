#!/bin/bash
#SBATCH -J exp2a
#SBATCH -p preempt
#SBATCH -n 2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o exp2a_%j.out
#SBATCH -e exp2a_%j.err
#
# Submit from repo root. Optional: set MODE (default A).
#   sbatch slurm/run_exp2a.slurm
#   sbatch --export=ALL,MODE=B slurm/run_exp2a.slurm
#   sbatch --export=ALL,MODE=C slurm/run_exp2a.slurm
#
# Writes:
#   experiment2a_{MODE}.json
#   experiment2a_{MODE}.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

MODE="${MODE:-A}"

module load pytorch/2.8.0-cuda12.9-cudnn9

python3 scripts/exp2a.py --mode "$MODE"

