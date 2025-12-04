#!/usr/bin/env bash
# Run a two-stage curriculum (stage 2 -> stage 3) with chained checkpoints.
# Usage: scripts/run_curriculum.sh [additional PPO/training args]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/raw/connections.json}"
EVAL_PATH="${EVAL_PATH:-$ROOT_DIR/data/raw/connections_test.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/curriculum}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-1}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-2}"
STAGE3_LR="${STAGE3_LR:-3e-6}"
STAGE3_ENTROPY="${STAGE3_ENTROPY:-0.01}"
STAGE3_REPLAY_PATH="${STAGE3_REPLAY_PATH:-$DATA_PATH}"
STAGE3_REPLAY_FRACTION="${STAGE3_REPLAY_FRACTION:-0.2}"
STAGE2_EVAL_SAMPLES="${STAGE2_EVAL_SAMPLES:-10}"
STAGE3_EVAL_SAMPLES="${STAGE3_EVAL_SAMPLES:-100}"

mkdir -p "$OUTPUT_ROOT"

EXTRA_ARGS=("$@")

PROTECTED_FLAGS=(--model_name_or_path --data_path --eval_path --reward_stage --num_train_epochs --output_dir --eval_samples)

FILTERED_EXTRA_ARGS=()
filter_stage_args() {
    FILTERED_EXTRA_ARGS=()
    local -a kept=()
    if [ "$#" -eq 0 ]; then
        FILTERED_EXTRA_ARGS=()
        return
    fi
    while (($#)); do
        local arg="$1"
        shift
        local skip=0
        for flag in "${PROTECTED_FLAGS[@]}"; do
            if [[ "$arg" == "$flag" ]]; then
                skip=1
                if (($#)); then
                    local candidate="$1"
                    if [[ "$candidate" != --* ]]; then
                        shift
                    fi
                fi
                break
            fi
        done
        if ((skip == 0)); then
            kept+=("$arg")
        fi
    done
    FILTERED_EXTRA_ARGS=("${kept[@]}")
}

filter_stage_args "${EXTRA_ARGS[@]}"

run_stage() {
    local stage="$1"
    local init_model="$2"
    local epochs="$3"
    shift 3 || true
    local stage_dir="$OUTPUT_ROOT/stage${stage}"

    echo "=== Running Stage ${stage} (init=$init_model -> out=$stage_dir, epochs=$epochs) ==="
    local stage_eval_samples="$STAGE2_EVAL_SAMPLES"
    if [ "$stage" -eq 3 ]; then
        stage_eval_samples="$STAGE3_EVAL_SAMPLES"
    fi
    local -a cmd=(
        "$PYTHON_BIN" "$ROOT_DIR/train_llm_ppo.py"
        --model_name_or_path "$init_model"
        --data_path "$DATA_PATH"
        --eval_path "$EVAL_PATH"
        --reward_stage "$stage"
        --num_train_epochs "$epochs"
        --output_dir "$stage_dir"
        --eval_samples "$stage_eval_samples"
    )
    if [ "$stage" -eq 3 ]; then
        cmd+=(--learning_rate "$STAGE3_LR")
        cmd+=(--entropy_coef "$STAGE3_ENTROPY")
        cmd+=(--stage2_replay_path "$STAGE3_REPLAY_PATH")
        cmd+=(--stage2_replay_fraction "$STAGE3_REPLAY_FRACTION")
    fi

    if [ ${#FILTERED_EXTRA_ARGS[@]} -gt 0 ]; then
        cmd+=("${FILTERED_EXTRA_ARGS[@]}")
    fi
    "${cmd[@]}"

    echo "=== Stage ${stage} finished; artifacts in $stage_dir ==="
    echo
}

current_model="$BASE_MODEL"
declare -A stage_epochs=(
    [2]="$STAGE2_EPOCHS"
    [3]="$STAGE3_EPOCHS"
)

for stage in 2 3; do
    epochs="${stage_epochs[$stage]}"
    if [ -z "$epochs" ]; then
        echo "Missing epoch count for stage $stage" >&2
        exit 1
    fi
    run_stage "$stage" "$current_model" "$epochs" "${EXTRA_ARGS[@]}"
    current_model="$OUTPUT_ROOT/stage${stage}"
done
