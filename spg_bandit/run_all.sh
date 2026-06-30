#!/bin/bash
# Run main experiment then eval comparison
# Usage: bash spg_bandit/run_all.sh [--config <name>] [--no-wandb]

set -e

CONFIG="default"
WANDB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --no-wandb) WANDB="--no-wandb"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "Phase 1: Main experiment ($CONFIG)"
echo "========================================"
python3 spg_bandit/main.py --config "$CONFIG" $WANDB

echo ""
echo "========================================"
echo "Phase 2: Eval comparison"
echo "========================================"
python3 spg_bandit/eval.py --config "$CONFIG" $WANDB

echo ""
echo "Done."
