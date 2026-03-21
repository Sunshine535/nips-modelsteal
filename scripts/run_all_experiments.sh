#!/usr/bin/env bash
# Master experiment script for Progressive Parameter Inversion.
#
# Pipeline:
#   1. KD baseline (teacher → student via logit distillation)
#   2. Progressive inversion with 3 query strategies
#   3. Recovery quality evaluation (weight + output + downstream)
#   4. Defense evaluation (rounding, noise, temperature, watermark)
#   5. Scaling analysis (vary query budget)
#   6. Generate figures
#
# Usage:
#   bash scripts/run_all_experiments.sh
#   bash scripts/run_all_experiments.sh --skip_kd        # resume from inversion
#   bash scripts/run_all_experiments.sh --gpus 4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_utils.sh"
auto_setup

# --- Activate project venv (created by setup.sh) ---
PROJ_DIR_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJ_DIR_ROOT/.venv/bin/activate" ]; then
    source "$PROJ_DIR_ROOT/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="configs/inversion_config.yaml"
RESULTS_DIR="results"
LOG_DIR="logs"
SEED=42
SKIP_KD=false
SKIP_INVERSION=false
SKIP_DEFENSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_kd)          SKIP_KD=true; shift ;;
        --skip_inversion)   SKIP_INVERSION=true; shift ;;
        --skip_defense)     SKIP_DEFENSE=true; shift ;;
        --seed)             SEED="$2"; shift 2 ;;
        *)                  echo "WARNING: Unknown arg: $1 (ignored)"; shift ;;
    esac
done

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
run_timed() {
    local name="$1"; shift
    log "START: $name"
    local start=$SECONDS
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    local elapsed=$(( SECONDS - start ))
    log "DONE:  $name (${elapsed}s)"
}

# ── Step 1: Knowledge Distillation Baseline ──────────────────────────
if [ "$SKIP_KD" = false ]; then
    run_timed "01_kd_baseline" \
        python scripts/run_kd_baseline.py \
            --config "$CONFIG" \
            --query_budget 500000 \
            --batch_size 32 \
            --temperature 2.0 \
            --alpha 0.7 \
            --num_epochs 3 \
            --output_dir "$RESULTS_DIR/kd_baseline" \
            --seed "$SEED"
fi

# ── Step 2: Progressive Inversion (3 strategies) ────────────────────
if [ "$SKIP_INVERSION" = false ]; then
    run_timed "02_progressive_inversion" \
        python scripts/run_progressive_inversion.py \
            --config "$CONFIG" \
            --query_budget 500000 \
            --batch_size 64 \
            --learning_rate 1e-4 \
            --max_steps_per_layer 10000 \
            --query_pool_size 10000 \
            --strategies random gradient_magnitude fisher_information \
            --output_dir "$RESULTS_DIR/progressive_inversion" \
            --seed "$SEED"
fi

# ── Step 3: Recovery Quality Evaluation ──────────────────────────────
run_timed "03_eval_recovery" \
    python scripts/eval_recovery_quality.py \
        --config "$CONFIG" \
        --kd_model "$RESULTS_DIR/kd_baseline/best_student" \
        --inversion_model "$RESULTS_DIR/progressive_inversion/strategy_gradient_magnitude/recovered_model" \
        --output_dir "$RESULTS_DIR/recovery_evaluation" \
        --num_output_samples 2000 \
        --downstream_samples 200

# ── Step 4: Defense Evaluation ───────────────────────────────────────
if [ "$SKIP_DEFENSE" = false ]; then
    run_timed "04_defense_eval" \
        python scripts/run_defense_eval.py \
            --config "$CONFIG" \
            --query_budget 100000 \
            --output_dir "$RESULTS_DIR/defense_eval" \
            --defenses logit_rounding gaussian_noise temperature_perturbation watermarking \
            --seed "$SEED"
fi

# ── Step 5: Scaling Analysis (vary query budget) ─────────────────────
log "START: 05_scaling_analysis"
for BUDGET in 50000 100000 250000 500000; do
    log "  Scaling: budget=$BUDGET"
    python scripts/run_progressive_inversion.py \
        --config "$CONFIG" \
        --query_budget "$BUDGET" \
        --strategies gradient_magnitude \
        --output_dir "$RESULTS_DIR/scaling/budget_${BUDGET}" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_DIR/05_scaling_${BUDGET}.log"
done
log "DONE:  05_scaling_analysis"

# ── Step 6: Eval existing distillation baseline (reuse existing) ─────
if [ -d "$RESULTS_DIR/distillation" ]; then
    run_timed "06_eval_distillation" \
        python scripts/eval_recovery_quality.py \
            --config "$CONFIG" \
            --kd_model "$RESULTS_DIR/distillation/best_student" \
            --inversion_model "$RESULTS_DIR/progressive_inversion/strategy_gradient_magnitude/recovered_model" \
            --output_dir "$RESULTS_DIR/eval_distillation_vs_inversion"
fi

# ── Summary ──────────────────────────────────────────────────────────
log "============================================="
log "All experiments complete."
log "Results directory: $RESULTS_DIR"
log "Logs directory:    $LOG_DIR"
log ""
log "Key outputs:"
log "  KD baseline:       $RESULTS_DIR/kd_baseline/"
log "  Progressive inv:   $RESULTS_DIR/progressive_inversion/"
log "  Recovery eval:     $RESULTS_DIR/recovery_evaluation/"
log "  Defense eval:      $RESULTS_DIR/defense_eval/"
log "  Scaling analysis:  $RESULTS_DIR/scaling/"
log ""
log "Key JSON files:"
log "  $RESULTS_DIR/progressive_inversion/strategy_comparison.json"
log "  $RESULTS_DIR/recovery_evaluation/recovery_evaluation.json"
log "  $RESULTS_DIR/defense_eval/defense_impact.json"
log "============================================="
