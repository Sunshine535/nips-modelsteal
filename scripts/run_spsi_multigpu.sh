#!/usr/bin/env bash
# Multi-GPU parallel S-PSI: distribute random initializations across GPUs.
#
# Each GPU runs one init independently. This is the most efficient strategy
# for S-PSI since the per-block inversion is sequential but init-parallel.
#
# Usage:
#   bash scripts/run_spsi_multigpu.sh
#   bash scripts/run_spsi_multigpu.sh --regime oracle --num_inits 8 --gpus 4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

REGIME="${REGIME:-oracle}"
NUM_INITS="${NUM_INITS:-5}"
NUM_SUFFIX="${NUM_SUFFIX:-2}"
BETA="${BETA:-0.1}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-results/spsi_${REGIME}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-0.8B}"
CONFIG="${SPSI_CONFIG:-configs/inversion_config.yaml}"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)

while [[ $# -gt 0 ]]; do
    case $1 in
        --regime)     REGIME="$2"; shift 2 ;;
        --num_inits)  NUM_INITS="$2"; shift 2 ;;
        --num_suffix) NUM_SUFFIX="$2"; shift 2 ;;
        --beta)       BETA="$2"; shift 2 ;;
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model)      MODEL_NAME="$2"; shift 2 ;;
        --config)     CONFIG="$2"; shift 2 ;;
        *)            echo "Unknown arg: $1"; shift ;;
    esac
done

mkdir -p "$OUTPUT_DIR" logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== S-PSI Multi-GPU Parallel ==="
log "Regime: $REGIME | Inits: $NUM_INITS | GPUs: $NUM_GPUS"
log "Model: $MODEL_NAME | Suffix blocks: $NUM_SUFFIX | Beta: $BETA"

PIDS=()
LABELS=()

for ((init=0; init<NUM_INITS; init++)); do
    GPU_ID=$((init % NUM_GPUS))
    INIT_SEED=$((SEED + init * 1000))
    INIT_DIR="${OUTPUT_DIR}/regime_${REGIME}/init_${init}"
    LOG_FILE="logs/spsi_${REGIME}_init${init}.log"

    log "  GPU $GPU_ID <- Init $init (seed=$INIT_SEED)"

    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/run_spsi.py \
        --config "$CONFIG" \
        --model_name "$MODEL_NAME" \
        --regime "$REGIME" \
        --num_inits 1 \
        --init_offset "$init" \
        --num_suffix_blocks "$NUM_SUFFIX" \
        --beta "$BETA" \
        --seed "$INIT_SEED" \
        --output_dir "${OUTPUT_DIR}" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    LABELS+=("init_$init")

    if (( (init + 1) % NUM_GPUS == 0 )) && (( init + 1 < NUM_INITS )); then
        log "  Waiting for batch of $NUM_GPUS inits to finish..."
        for j in "${!PIDS[@]}"; do
            if ! wait "${PIDS[$j]}"; then
                log "WARNING: ${LABELS[$j]} failed"
            fi
        done
        PIDS=()
        LABELS=()
    fi
done

FAIL=0
for j in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$j]}"; then
        log "ERROR: ${LABELS[$j]} failed"
        FAIL=1
    fi
done

if [ $FAIL -ne 0 ]; then
    log "Some inits failed. Check logs/ for details."
    exit 1
fi

log "=== All $NUM_INITS inits complete ==="
log "Results in: $OUTPUT_DIR"
log "Aggregate with: python scripts/run_spsi.py --regime $REGIME (for cross-init stats)"
