#!/usr/bin/env bash
set -euo pipefail

# Progressive Parameter Inversion — Full Pipeline Launcher (8x A100)

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_DIR/configs/inversion_config.yaml"

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-9B}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3.5-0.8B}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/results}"
QUERY_BUDGET="${QUERY_BUDGET:-500000}"
NUM_GPUS="${NUM_GPUS:-8}"

mkdir -p "$RESULT_DIR"/{distillation,inversion,evaluation,defense}
mkdir -p "$PROJECT_DIR/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Progressive Parameter Inversion Pipeline ==="
log "Teacher: $TEACHER_MODEL"
log "Student: $STUDENT_MODEL"
log "Query budget: $QUERY_BUDGET"
log "GPUs: $NUM_GPUS"

# --- Stage 1: Behavioral Distillation ---
log "--- Stage 1: Behavioral Distillation ---"
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29500 \
    "$SCRIPT_DIR/distill_student.py" \
    --teacher_model "$TEACHER_MODEL" \
    --student_model "$STUDENT_MODEL" \
    --num_queries 100000 \
    --batch_size 32 \
    --temperature 2.0 \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --max_seq_len 512 \
    --output_dir "$RESULT_DIR/distillation" \
    --config "$CONFIG" \
    2>&1 | tee "$PROJECT_DIR/logs/stage1_distill.log"

log "Stage 1 complete."

# --- Stage 2: Progressive Parameter Inversion ---
log "--- Stage 2: Progressive Parameter Inversion ---"
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29501 \
    "$SCRIPT_DIR/invert_parameters.py" \
    --teacher_model "$TEACHER_MODEL" \
    --student_checkpoint "$RESULT_DIR/distillation/best_student" \
    --query_budget "$QUERY_BUDGET" \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --max_steps_per_layer 10000 \
    --active_query_strategy gradient_magnitude \
    --output_dir "$RESULT_DIR/inversion" \
    --config "$CONFIG" \
    2>&1 | tee "$PROJECT_DIR/logs/stage2_invert.log"

log "Stage 2 complete."

# --- Stage 3: Evaluation ---
log "--- Stage 3: Extraction Quality Evaluation ---"
python "$SCRIPT_DIR/eval_extraction.py" \
    --teacher_model "$TEACHER_MODEL" \
    --recovered_model "$RESULT_DIR/inversion/recovered_model" \
    --num_eval_samples 5000 \
    --output_dir "$RESULT_DIR/evaluation" \
    --config "$CONFIG" \
    2>&1 | tee "$PROJECT_DIR/logs/stage3_eval.log"

log "Stage 3 complete."

# --- Stage 4: Defense Evaluation ---
log "--- Stage 4: Defense Evaluation ---"
for defense in logit_rounding gaussian_noise temperature_perturbation topk_masking watermarking; do
    log "Testing defense: $defense"
    python "$SCRIPT_DIR/defense_evaluation.py" \
        --teacher_model "$TEACHER_MODEL" \
        --student_model "$STUDENT_MODEL" \
        --defense_type "$defense" \
        --query_budget 100000 \
        --output_dir "$RESULT_DIR/defense/$defense" \
        --config "$CONFIG" \
        2>&1 | tee "$PROJECT_DIR/logs/stage4_defense_${defense}.log"
done

log "Stage 4 complete."
log "=== Full pipeline finished ==="
log "Results in: $RESULT_DIR"
