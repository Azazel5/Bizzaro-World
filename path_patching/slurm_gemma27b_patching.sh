#!/bin/bash
#SBATCH --job-name=bizzaro-27b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/patching_27b_%j.out
#SBATCH --error=logs/patching_27b_%j.err

set -euo pipefail

# ── environment ───────────────────────────────────────────────────────────────
module load pytorch/2.11.0-cuda12.6-cudnn9-devel           # run `module avail cuda` to find exact name on your cluster

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bizzaro

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before submitting}"

# ── repo ──────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"    # run from path_patching/
mkdir -p logs

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "============================================================"

# ── Experiment 2: receiver heads ─────────────────────────────────────────────
echo "[$(date)] Starting receiver-head experiment"
python run_experiments.py --model gemma_27b \
  --receiver-heads 54:23 55:24 53:11 58:29 61:21 54:22 58:28 61:10 58:31 47:23 \
  --receiver-input q \
  --skip-resid \
  --export-json

echo "[$(date)] Receiver experiment done"
echo "============================================================"

# ── Experiment 3: sender heads ────────────────────────────────────────────────
echo "[$(date)] Starting sender experiment"
python run_experiments.py --model gemma_27b \
  --sender-heads 54:23 55:24 53:11 58:29 61:21 54:22 58:28 61:10 58:31 47:23 \
  --skip-resid \
  --export-json

echo "[$(date)] Sender experiment done"
echo "============================================================"
echo "All done. Results in results/gemma_27b/"
