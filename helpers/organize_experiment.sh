#!/usr/bin/env bash
set -euo pipefail

# Organize experiment artifacts into a dedicated directory.
#
# Usage (from repo root):
#   bash helpers/organize_experiment.sh 2a
#   bash helpers/organize_experiment.sh 2b
#   bash helpers/organize_experiment.sh 2c
#   bash helpers/organize_experiment.sh 1
#
# What it moves (no overwrite, verbose):
# - Experiment JSON/LOG named like:
#     experiment2a_A.json, experiment2a_A.log (and B/C), etc.
# - Slurm stdout/err named like:
#     exp2a_<jobid>.out, exp2a_<jobid>.err, etc.
#
# Notes:
# - This script intentionally does NOT recurse into existing Experiment* directories.
# - It is safe to run multiple times; already-moved files won't be re-moved.

exp="${1:-}"
if [[ -z "$exp" ]]; then
  echo "Usage: bash helpers/organize_experiment.sh <experiment_tag>"
  echo "Example: bash helpers/organize_experiment.sh 2a"
  exit 2
fi

exp_lc="$(echo "$exp" | tr '[:upper:]' '[:lower:]')"

# Directory name: 2a -> Experiment2A, 2b -> Experiment2B, 1 -> Experiment1, etc.
exp_uc="$(echo "$exp_lc" | tr '[:lower:]' '[:upper:]')"
exp_dir="Experiment${exp_uc}"

# Filename stem used by Slurm logs: 2a -> exp2a, 1 -> exp1
slurm_stem="exp${exp_lc}"

# Filename stem used by JSON/LOG outputs:
# - exp1.py writes experiment_A.json (no "1" in the name)
# - exp2a.py writes experiment2a_A.json (includes the tag)
json_stem=""
if [[ "$exp_lc" == "1" ]]; then
  json_stem="experiment"
else
  json_stem="experiment${exp_lc}"
fi

mkdir -p "$exp_dir"
shopt -s nullglob

files=(
  "${json_stem}"_A.json
  "${json_stem}"_B.json
  "${json_stem}"_C.json
  "${json_stem}"_A.log
  "${json_stem}"_B.log
  "${json_stem}"_C.log
  "${slurm_stem}"_*.out
  "${slurm_stem}"_*.err
)

to_move=()
for f in "${files[@]}"; do
  if [[ -e "$f" ]]; then
    to_move+=("$f")
  fi
done

if ((${#to_move[@]} == 0)); then
  echo "No files found for experiment tag '$exp' (expected stems: '${json_stem}*' and '${slurm_stem}_*')."
  exit 0
fi

echo "Moving ${#to_move[@]} files to $exp_dir/"
mv -vn "${to_move[@]}" "$exp_dir/"
echo "Done."

