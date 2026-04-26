#!/usr/bin/env bash
# Wrapper that always uses the project venv.
# Usage: ./train.sh --eval-all
#        ./train.sh --train --task alert_triage --seeds 20 --epochs 3 --num-generations 4
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python3"
if [[ ! -f "$PYTHON" ]]; then
  echo "ERROR: venv not found. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -e .[train]"
  exit 1
fi
exec "$PYTHON" -m training.train_unsloth_grpo "$@"
