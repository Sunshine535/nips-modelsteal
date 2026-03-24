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

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

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
        --gpus)             export NUM_GPUS="$2"; shift 2 ;;
        --seed)             SEED="$2"; shift 2 ;;
        *)                  echo "WARNING: Unknown arg: $1 (ignored)"; shift ;;
    esac
done

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

TORCHRUN="$(get_torchrun_cmd "$NUM_GPUS")"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
run_timed() {
    local name="$1"; shift
    log "START: $name"
    local start=$SECONDS
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    local elapsed=$(( SECONDS - start ))
    log "DONE:  $name (${elapsed}s)"
}

log "Using $NUM_GPUS GPU(s). TORCHRUN: $TORCHRUN"

# ═══════════════════════════════════════════════════════════════════════
#  Phase A: KD Baseline — torchrun DDP across ALL $NUM_GPUS GPUs
# ═══════════════════════════════════════════════════════════════════════
if [ "$SKIP_KD" = false ]; then
    run_timed "01_kd_baseline" \
        $TORCHRUN scripts/run_kd_baseline.py \
            --config "$CONFIG" \
            --query_budget 500000 \
            --batch_size 4 \
            --gradient_accumulation_steps 8 \
            --temperature 2.0 \
            --alpha 0.7 \
            --num_epochs 3 \
            --output_dir "$RESULTS_DIR/kd_baseline" \
            --seed "$SEED"
fi

# ═══════════════════════════════════════════════════════════════════════
#  Phase B: Inversion + Scaling + Defense — 8 parallel single-GPU tasks
#   GPU 0-2: 3 inversion strategies
#   GPU 3-6: 4 scaling budgets
#   GPU 7:   defense evaluation
# ═══════════════════════════════════════════════════════════════════════
log "START: Phase B (inversion + scaling + defense, 8 parallel tasks on $NUM_GPUS GPUs)"
GPU_IDX=0
PIDS=()
LABELS=()

# --- 3 inversion strategies ---
if [ "$SKIP_INVERSION" = false ]; then
    STRATEGIES=(random gradient_magnitude fisher_information)
    for s in "${STRATEGIES[@]}"; do
        log "  GPU $GPU_IDX ← strategy=$s"
        CUDA_VISIBLE_DEVICES=$GPU_IDX python scripts/run_progressive_inversion.py \
            --config "$CONFIG" \
            --query_budget 500000 \
            --batch_size 64 \
            --learning_rate 1e-4 \
            --max_steps_per_layer 10000 \
            --query_pool_size 10000 \
            --strategies "$s" \
            --output_dir "$RESULTS_DIR/progressive_inversion" \
            --seed "$SEED" \
            > "$LOG_DIR/02_strategy_${s}.log" 2>&1 &
        PIDS+=($!); LABELS+=("strategy_$s")
        GPU_IDX=$((GPU_IDX + 1))
    done
fi

# --- 4 scaling budgets ---
BUDGETS=(50000 100000 250000 500000)
for b in "${BUDGETS[@]}"; do
    log "  GPU $GPU_IDX ← scaling budget=$b"
    CUDA_VISIBLE_DEVICES=$GPU_IDX python scripts/run_progressive_inversion.py \
        --config "$CONFIG" \
        --query_budget "$b" \
        --strategies gradient_magnitude \
        --output_dir "$RESULTS_DIR/scaling/budget_${b}" \
        --seed "$SEED" \
        > "$LOG_DIR/05_scaling_${b}.log" 2>&1 &
    PIDS+=($!); LABELS+=("budget_$b")
    GPU_IDX=$((GPU_IDX + 1))
done

# --- 1 defense evaluation ---
if [ "$SKIP_DEFENSE" = false ]; then
    log "  GPU $GPU_IDX ← defense eval (4 defenses)"
    CUDA_VISIBLE_DEVICES=$GPU_IDX python scripts/run_defense_eval.py \
        --config "$CONFIG" \
        --query_budget 100000 \
        --output_dir "$RESULTS_DIR/defense_eval" \
        --defenses logit_rounding gaussian_noise temperature_perturbation watermarking \
        --seed "$SEED" \
        > "$LOG_DIR/04_defense_eval.log" 2>&1 &
    PIDS+=($!); LABELS+=("defense_eval")
    GPU_IDX=$((GPU_IDX + 1))
fi

# Wait for all Phase B tasks
FAIL=0
for j in "${!PIDS[@]}"; do
    wait "${PIDS[$j]}" || { log "ERROR: ${LABELS[$j]} failed (pid=${PIDS[$j]})"; FAIL=1; }
done
if [ $FAIL -ne 0 ]; then exit 1; fi

# Merge per-strategy summaries into combined comparison JSON
if [ "$SKIP_INVERSION" = false ]; then
    python3 -c "
import json; from pathlib import Path
strats = ['random', 'gradient_magnitude', 'fisher_information']
d = Path('$RESULTS_DIR/progressive_inversion')
comp = {'strategies': {}, 'query_budget': 500000}
for s in strats:
    p = d / f'strategy_{s}' / 'inversion_summary.json'
    if p.exists():
        r = json.loads(p.read_text())
        comp['strategies'][s] = dict(
            total_queries=r['total_queries'],
            final_top1_match=r['final_downstream']['top1_match_rate'],
            final_kl_divergence=r['final_downstream']['mean_kl_divergence'],
            num_phases=len(r['phases']))
(d / 'strategy_comparison.json').write_text(json.dumps(comp, indent=2))
print('  Strategy comparison merged.')
"
fi
log "DONE:  Phase B"

# ═══════════════════════════════════════════════════════════════════════
#  Phase C: Recovery Evaluation — torchrun DDP across ALL $NUM_GPUS GPUs
# ═══════════════════════════════════════════════════════════════════════
run_timed "03_eval_recovery" \
    $TORCHRUN scripts/eval_recovery_quality.py \
        --config "$CONFIG" \
        --kd_model "$RESULTS_DIR/kd_baseline/best_student" \
        --inversion_model "$RESULTS_DIR/progressive_inversion/strategy_gradient_magnitude/recovered_model" \
        --output_dir "$RESULTS_DIR/recovery_evaluation" \
        --num_output_samples 2000 \
        --downstream_samples 200

if [ -d "$RESULTS_DIR/distillation" ]; then
    run_timed "06_eval_distillation" \
        $TORCHRUN scripts/eval_recovery_quality.py \
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

# --- Pipeline completion marker ---
DONE_FILE="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/results/.pipeline_done"
mkdir -p "$(dirname "$DONE_FILE")"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "$(basename "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo ""
echo "[PIPELINE_COMPLETE] All experiments finished successfully."
echo "  Marker: $DONE_FILE"
echo "  Run 'bash collect_results.sh' to package results."
