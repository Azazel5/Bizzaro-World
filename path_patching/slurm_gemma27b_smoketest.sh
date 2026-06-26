#!/bin/bash
#SBATCH --job-name=bizzaro-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoketest_%j.out
#SBATCH --error=logs/smoketest_%j.err

set -eo pipefail

module load cuda/12.9.0
module load anaconda/2025.06.0
conda activate bizzaro

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before submitting}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "============================================================"
echo "[$(date)] Smoke test — 1 prompt, receiver only"

python run_experiments.py --model gemma_27b \
  --receiver-heads 54:23 55:24 53:11 58:29 61:21 54:22 58:28 61:10 58:31 47:23 \
  --receiver-input q \
  --skip-resid \
  --export-json \
  --max-prompts 1

echo "[$(date)] Smoke test done — check results/gemma_27b/"
