#!/usr/bin/env bash
set -euo pipefail

# Backwards-compatible wrapper. Prefer:
#   bash helpers/organize_experiment.sh 2a

bash "$(dirname "$0")/organize_experiment.sh" 2a

