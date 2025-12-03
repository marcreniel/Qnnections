#!/usr/bin/env bash
# Run the three-stage reward curriculum sequentially.
# Usage: scripts/run_curriculum.sh [additional PPO/training args]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/raw/connections.json}"
EVAL_PATH="${EVAL_PATH:-$ROOT_DIR/data/raw/connections_test.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/curriculum}"
MAX_STAGE="${MAX_STAGE:-3}"

mkdir -p "$OUTPUT_ROOT"

EXTRA_ARGS=("$@")

run_stage() {
    local stage="$1"
    local init_model="$2"
    shift 2 || true
    local stage_dir="$OUTPUT_ROOT/stage${stage}"

    echo "=== Running Stage ${stage} (init=$init_model -> out=$stage_dir) ==="
    local -a cmd=(
        "$PYTHON_BIN" "$ROOT_DIR/train_llm_ppo.py"
        --model_name_or_path "$init_model"
        --data_path "$DATA_PATH"
        --eval_path "$EVAL_PATH"
        --reward_stage "$stage"
        --output_dir "$stage_dir"
    )
    if [ "$#" -gt 0 ]; then
        cmd+=("$@")
    fi
    "${cmd[@]}"

    echo "=== Stage ${stage} finished; artifacts in $stage_dir ==="
    echo
}

current_model="$BASE_MODEL"
for stage in 1 2 3; do
    if [ "$stage" -gt "$MAX_STAGE" ]; then
        break
    fi
    run_stage "$stage" "$current_model" "${EXTRA_ARGS[@]}"
    current_model="$OUTPUT_ROOT/stage${stage}"
done
