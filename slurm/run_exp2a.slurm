#!/bin/bash
# Backwards-compatible wrapper. Prefer: slurm/run_exp2.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

MODE="${MODE:-A}"
OUTDIR="${OUTDIR:-./outputs}"

module load pytorch/2.8.0-cuda12.9-cudnn9

mkdir -p "$OUTDIR"
python3 scripts/exp2.py --mode "$MODE" --outdir "$OUTDIR"
