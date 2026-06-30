#!/bin/bash
# Run main experiment then eval with evolved skills
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
python3 spg_bandit/main.py --config "$CONFIG" $WANDB 2>&1 | tee /tmp/main_output.log

RUN_ID=$(grep "Run ID:" /tmp/main_output.log | head -1 | sed 's/.*Run ID: //' | tr -d '[:space:]')
SKILLS_BASE="spg_bandit/modules/skill_evolving/simple_agent/skills/$RUN_ID"

echo ""
echo "Main run ID: $RUN_ID"

echo ""
echo "========================================"
echo "Phase 2: Eval with each skill set"
echo "========================================"

for sel_dir in "$SKILLS_BASE"/*/; do
    sel_name=$(basename "$sel_dir")
    echo ""
    echo "--- Eval with ${sel_name} skills ---"
    python3 spg_bandit/eval.py --config "$CONFIG" \
        --label "$sel_name" \
        --skills "$sel_dir"
done

echo ""
echo "All done. Skills: $SKILLS_BASE"
