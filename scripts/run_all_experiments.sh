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
elif [ -d "$PROJ_DIR_ROOT/.conda_env" ] && command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$PROJ_DIR_ROOT/.conda_env"
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

CONFIG="${SPSI_CONFIG:-configs/inversion_config.yaml}"
RESULTS_DIR="results"
LOG_DIR="logs"
SEED=42
SKIP_KD=false
SKIP_ABLATION=false
NUM_INITS=5
NUM_SUFFIX=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)        CONFIG="$2"; shift 2 ;;
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
    local rc=0
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log" || rc=$?
    local elapsed=$(( SECONDS - start ))
    if [ $rc -ne 0 ]; then
        log "FAILED: $name (${elapsed}s, exit=$rc)"
        return $rc
    fi
    log "DONE:  $name (${elapsed}s)"
}

run_spsi_parallel() {
    local regime="$1"; local out_dir="$2"; shift 2
    local extra_args="$@"
    if [ "$NUM_GPUS" -le 1 ] || [ "$NUM_INITS" -le 1 ]; then
        python scripts/run_spsi.py \
            --config "$CONFIG" --regime "$regime" \
            --num_inits "$NUM_INITS" --num_suffix_blocks "$NUM_SUFFIX" \
            --output_dir "$out_dir" --seed "$SEED" $extra_args
        return
    fi
    log "  Multi-GPU S-PSI: $NUM_INITS inits across $NUM_GPUS GPUs"
    local pids=() labels=() fail=0
    for ((init=0; init<NUM_INITS; init++)); do
        local gpu_id=$((init % NUM_GPUS))
        local init_seed=$((SEED + init * 1000))
        CUDA_VISIBLE_DEVICES=$(gpu_at_index $gpu_id) python scripts/run_spsi.py \
            --config "$CONFIG" --regime "$regime" \
            --num_inits 1 --init_offset "$init" --num_suffix_blocks "$NUM_SUFFIX" \
            --output_dir "$out_dir" --seed "$init_seed" $extra_args \
            > "$LOG_DIR/spsi_${regime}_init${init}.log" 2>&1 &
        pids+=($!)
        labels+=("init_${init}/gpu_${gpu_id}")
        if (( (init + 1) % NUM_GPUS == 0 )) && (( init + 1 < NUM_INITS )); then
            log "  Waiting for GPU batch..."
            for j in "${!pids[@]}"; do
                wait "${pids[$j]}" || { log "WARN: ${labels[$j]} failed"; fail=1; }
            done
            pids=() labels=()
        fi
    done
    for j in "${!pids[@]}"; do
        wait "${pids[$j]}" || { log "WARN: ${labels[$j]} failed"; fail=1; }
    done
    if [ $fail -ne 0 ]; then
        log "ERROR: Some S-PSI inits failed — aborting (check logs/)"
        return 1
    fi

    log "  Aggregating multi-GPU results..."
    python scripts/run_spsi.py \
        --config "$CONFIG" --regime "$regime" \
        --output_dir "$out_dir" --aggregate_only
}

log "=== S-PSI Experiment Pipeline ==="
log "Using $NUM_GPUS GPU(s). TORCHRUN: $TORCHRUN"
log "Inits: $NUM_INITS | Suffix blocks: $NUM_SUFFIX | Seed: $SEED"

# Save resolved config and run metadata for reproducibility
mkdir -p "$RESULTS_DIR"
cp "$CONFIG" "$RESULTS_DIR/resolved_config.yaml" 2>/dev/null || true
cat > "$RESULTS_DIR/run_metadata.json" << METAEOF
{
  "start_time": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": $NUM_GPUS,
  "seed": $SEED,
  "num_inits": $NUM_INITS,
  "num_suffix_blocks": $NUM_SUFFIX,
  "config": "$CONFIG",
  "force_rerun": "$FORCE_RERUN"
}
METAEOF

# Warn about existing GPU processes (do NOT kill — unsafe on shared machines)
STALE_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u || true)
if [ -n "$STALE_PIDS" ]; then
    log "WARNING: Found existing GPU processes: $STALE_PIDS"
    log "  If these are from a previous run, kill them manually before re-running."
    log "  Continuing in 5 seconds..."
    sleep 5
fi

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
            --seed "$SEED" \
            --resume_from_checkpoint auto
fi
phase_done A
fi

# ═══════════════════════════════════════════════════════════════════════
#  Phase B: S-PSI Oracle-Prefix
# ═══════════════════════════════════════════════════════════════════════
if ! is_phase_done B; then
    run_timed "02_spsi_oracle" \
        run_spsi_parallel oracle "$RESULTS_DIR/spsi_oracle"
    phase_done B
fi

# ═══════════════════════════════════════════════════════════════════════
#  Phase C: S-PSI Pure-Logits
# ═══════════════════════════════════════════════════════════════════════
if ! is_phase_done C; then
    run_timed "03_spsi_pure_logits" \
        run_spsi_parallel pure_logits "$RESULTS_DIR/spsi_pure_logits"
    phase_done C
fi

# ═══════════════════════════════════════════════════════════════════════
#  Phase D: Ablation — β=0 (no sensitivity matching)
# ═══════════════════════════════════════════════════════════════════════
if ! is_phase_done D; then
if [ "$SKIP_ABLATION" = false ]; then
    run_timed "04_ablation_beta0" \
        run_spsi_parallel both "$RESULTS_DIR/spsi_ablation_beta0" --beta 0.0
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
    elif [ -d "$RESULTS_DIR/kd_baseline/final_student" ]; then
        KD_MODEL="$RESULTS_DIR/kd_baseline/final_student"
    else
        log "WARNING: No KD student model found; skipping Phase F"
        phase_done F
        KD_MODEL=""
    fi

    INV_MODEL="$RESULTS_DIR/spsi_oracle/regime_oracle/init_0/recovered_model"
    if [ ! -d "$INV_MODEL" ]; then
        log "WARNING: No S-PSI recovered model at $INV_MODEL; skipping Phase F"
        phase_done F
        INV_MODEL=""
    fi

    if [ -n "$KD_MODEL" ] && [ -n "$INV_MODEL" ]; then
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
