#!/usr/bin/env bash
# S-PSI: Master experiment script.
#
# Pipeline:
#   Phase A: KD Baseline (DDP)
#   Phase B: S-PSI inversion — oracle + pure_logits (parallel per-init)
#   Phase C: Ablation — β=0 (no sensitivity matching)
#   Phase D: Depth boundary analysis — vary num_suffix_blocks
#   Phase E: Recovery evaluation
#
# Usage:
#   bash scripts/run_all_experiments.sh
#   bash scripts/run_all_experiments.sh --skip_kd
#   bash scripts/run_all_experiments.sh --gpus 4

set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_utils.sh"
auto_setup

PROJ_DIR_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJ_DIR_ROOT/.venv/bin/activate" ]; then
    source "$PROJ_DIR_ROOT/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"

PHASE_MARKER_DIR="$PROJ_DIR_ROOT/results/.phase_markers"
mkdir -p "$PHASE_MARKER_DIR"
FORCE_RERUN="${FORCE_RERUN:-0}"

phase_done() { touch "$PHASE_MARKER_DIR/phase_${1}.done"; echo "[PHASE $1] Completed at $(date)"; }
is_phase_done() {
    [[ "$FORCE_RERUN" == "1" ]] && return 1
    [[ -f "$PHASE_MARKER_DIR/phase_${1}.done" ]] && echo "[PHASE $1] Already completed. Skipping. (FORCE_RERUN=1 to override)" && return 0
    return 1
}

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="configs/inversion_config.yaml"
RESULTS_DIR="results"
LOG_DIR="logs"
SEED=42
SKIP_KD=false
SKIP_ABLATION=false
NUM_INITS=5
NUM_SUFFIX=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_kd)       SKIP_KD=true; shift ;;
        --skip_ablation) SKIP_ABLATION=true; shift ;;
        --gpus)          export NUM_GPUS="$2"; shift 2 ;;
        --seed)          SEED="$2"; shift 2 ;;
        --num_inits)     NUM_INITS="$2"; shift 2 ;;
        --num_suffix)    NUM_SUFFIX="$2"; shift 2 ;;
        *)               echo "WARNING: Unknown arg: $1 (ignored)"; shift ;;
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

log "=== S-PSI Experiment Pipeline ==="
log "Using $NUM_GPUS GPU(s). TORCHRUN: $TORCHRUN"
log "Inits: $NUM_INITS | Suffix blocks: $NUM_SUFFIX | Seed: $SEED"

# Kill stale GPU processes
log "Checking for stale GPU processes..."
for _attempt in 1 2; do
    STALE_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u || true)
    if [ -z "$STALE_PIDS" ]; then break; fi
    log "WARNING: Found GPU processes (attempt $_attempt): $STALE_PIDS"
    for spid in $STALE_PIDS; do
        CMDLINE=$(ps -p "$spid" -o comm= 2>/dev/null || true)
        if [[ "$CMDLINE" == *python* ]] || [[ "$CMDLINE" == *torchrun* ]]; then
            log "  Killing stale process $spid ($CMDLINE)"
            kill -9 "$spid" 2>/dev/null || true
        fi
    done
    sleep 5
done

# ═══════════════════════════════════════════════════════════════════════
#  Phase A: KD Baseline
# ═══════════════════════════════════════════════════════════════════════
if ! is_phase_done A; then
if [ "$SKIP_KD" = false ]; then
    run_timed "01_kd_baseline" \
        $TORCHRUN scripts/run_kd_baseline.py \
            --config "$CONFIG" \
            --query_budget 500000 \
            --batch_size 2 \
            --gradient_accumulation_steps 16 \
            --num_epochs 3 \
            --output_dir "$RESULTS_DIR/kd_baseline" \
            --seed "$SEED"
fi
phase_done A
fi

# ═══════════════════════════════════════════════════════════════════════
#  Phase B: S-PSI Oracle-Prefix
# ═══════════════════════════════════════════════════════════════════════
if ! is_phase_done B; then
    run_timed "02_spsi_oracle" \
        python scripts/run_spsi.py \
            --config "$CONFIG" \
            --regime oracle \
            --num_inits "$NUM_INITS" \
            --num_suffix_blocks "$NUM_SUFFIX" \
            --output_dir "$RESULTS_DIR/spsi_oracle" \
            --seed "$SEED"
    phase_done B
fi

# ═══════════════════════════════════════════════════════════════════════
#  Phase C: S-PSI Pure-Logits
# ═══════════════════════════════════════════════════════════════════════
if ! is_phase_done C; then
    run_timed "03_spsi_pure_logits" \
        python scripts/run_spsi.py \
            --config "$CONFIG" \
            --regime pure_logits \
            --num_inits "$NUM_INITS" \
            --num_suffix_blocks "$NUM_SUFFIX" \
            --output_dir "$RESULTS_DIR/spsi_pure_logits" \
            --seed "$SEED"
    phase_done C
fi

# ═══════════════════════════════════════════════════════════════════════
#  Phase D: Ablation — β=0 (no sensitivity matching)
# ═══════════════════════════════════════════════════════════════════════
if ! is_phase_done D; then
if [ "$SKIP_ABLATION" = false ]; then
    run_timed "04_ablation_beta0" \
        python scripts/run_spsi.py \
            --config "$CONFIG" \
            --regime both \
            --beta 0.0 \
            --num_inits "$NUM_INITS" \
            --num_suffix_blocks "$NUM_SUFFIX" \
            --output_dir "$RESULTS_DIR/spsi_ablation_beta0" \
            --seed "$SEED"
fi
phase_done D
fi

# ═══════════════════════════════════════════════════════════════════════
#  Phase D2: Wrong-Teacher Falsification Control
# ═══════════════════════════════════════════════════════════════════════
if ! is_phase_done D2; then
    run_timed "04b_wrong_teacher" \
        python scripts/run_spsi.py \
            --config "$CONFIG" \
            --regime oracle \
            --num_inits 3 \
            --num_suffix_blocks "$NUM_SUFFIX" \
            --wrong_teacher \
            --output_dir "$RESULTS_DIR/spsi_wrong_teacher" \
            --seed "$SEED"
    phase_done D2
fi

# ═══════════════════════════════════════════════════════════════════════
#  Phase E: Depth Boundary Analysis
# ═══════════════════════════════════════════════════════════════════════
if ! is_phase_done E; then
    DEPTH_VALS=(1 2 4 6)
    for d in "${DEPTH_VALS[@]}"; do
        run_timed "05_depth_${d}" \
            python scripts/run_spsi.py \
                --config "$CONFIG" \
                --regime oracle \
                --num_inits 3 \
                --num_suffix_blocks "$d" \
                --max_steps 5000 \
                --output_dir "$RESULTS_DIR/spsi_depth/depth_${d}" \
                --seed "$SEED"
    done
    phase_done E
fi

# ═══════════════════════════════════════════════════════════════════════
#  Phase F: Recovery Evaluation
# ═══════════════════════════════════════════════════════════════════════
if ! is_phase_done F; then
    if [ -d "$RESULTS_DIR/kd_baseline/best_student" ]; then
        KD_MODEL="$RESULTS_DIR/kd_baseline/best_student"
    else
        KD_MODEL="$RESULTS_DIR/kd_baseline/final_student"
    fi
    INV_MODEL="$RESULTS_DIR/spsi_oracle/regime_oracle/init_0/recovered_model"

    run_timed "06_eval_recovery" \
        $TORCHRUN scripts/eval_recovery_quality.py \
            --config "$CONFIG" \
            --kd_model "$KD_MODEL" \
            --inversion_model "$INV_MODEL" \
            --output_dir "$RESULTS_DIR/recovery_evaluation" \
            --num_output_samples 2000 \
            --downstream_samples 200
    phase_done F
fi

# ── Summary ──────────────────────────────────────────────────────────
log "============================================="
log "S-PSI: All experiments complete."
log "Results directory: $RESULTS_DIR"
log "Logs directory:    $LOG_DIR"
log ""
log "Key outputs:"
log "  KD baseline:       $RESULTS_DIR/kd_baseline/"
log "  S-PSI Oracle:      $RESULTS_DIR/spsi_oracle/"
log "  S-PSI Pure-Logits: $RESULTS_DIR/spsi_pure_logits/"
log "  Ablation β=0:      $RESULTS_DIR/spsi_ablation_beta0/"
log "  Depth analysis:    $RESULTS_DIR/spsi_depth/"
log "  Recovery eval:     $RESULTS_DIR/recovery_evaluation/"
log "============================================="

DONE_FILE="$PROJ_DIR_ROOT/results/.pipeline_done"
mkdir -p "$(dirname "$DONE_FILE")"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "S-PSI",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo ""
echo "[PIPELINE_COMPLETE] S-PSI experiments finished successfully."
