#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Auto-dispatch all experiment waves when GPUs free up
# Model: Qwen/Qwen2.5-0.5B (standard transformer, 24 blocks, 896 hidden)
# Run with: nohup bash scripts/dispatch_remaining_waves.sh > logs/dispatch.log 2>&1 &
# ═══════════════════════════════════════════════════════════════════════════════
set -e
PROJ="/mnt/ae22ef98-e02a-4dfa-acf4-a307e8941cd9/nwh/nips/nips-modelsteal"
LOG_DIR="$PROJ/logs"
RESULTS="$PROJ/results"
mkdir -p "$LOG_DIR"

export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Common arguments — Qwen2.5-0.5B (494M, standard attention)
COMMON="--model_name Qwen/Qwen2.5-0.5B --num_inits 1 --max_steps 3000 --lm_head_steps 1500 --batch_size 8 --query_budget 500000 --pool_size 2048 --max_seq_len 128 --logit_suffix_positions 8 --allow_synthetic --patience 500"

launch() {
    local NAME=$1 GPU=$2 EXTRA=$3
    local OUT="$RESULTS/$NAME"
    local LOG="$LOG_DIR/${NAME//\//_}.log"
    mkdir -p "$OUT"
    echo "[$(date '+%Y-%m-%d %H:%M')] Launching $NAME on GPU $GPU"
    nohup bash -c "
        cd $PROJ && source .venv/bin/activate
        export HF_ENDPOINT=https://hf-mirror.com
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        CUDA_VISIBLE_DEVICES=$GPU python -u scripts/run_spsi.py $COMMON $EXTRA --output_dir $OUT
    " > "$LOG" 2>&1 &
    local PID=$!
    echo "  PID=$PID -> $LOG"
    echo "$PID" > "$LOG_DIR/${NAME//\//_}.pid"
}

wait_gpu_free() {
    local GPU=$1
    echo "[$(date '+%H:%M')] Waiting for GPU $GPU to free up..."
    # First wait at least 60s to let experiments start
    sleep 60
    while true; do
        MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 99999)
        if [ "$MEM" -lt 500 ] 2>/dev/null; then
            echo "[$(date '+%H:%M')] GPU $GPU is free (${MEM}MiB)"
            return 0
        fi
        sleep 120
    done
}

wait_all_gpus() {
    for GPU in 0 1 2 3; do
        wait_gpu_free "$GPU"
    done
}

# Check if all PIDs from a wave are still alive; if not, the wave already completed
check_wave_pids() {
    local ALIVE=0
    for PIDFILE in "$@"; do
        if [ -f "$PIDFILE" ]; then
            local PID=$(cat "$PIDFILE")
            if kill -0 "$PID" 2>/dev/null; then
                ALIVE=$((ALIVE + 1))
            fi
        fi
    done
    echo "$ALIVE"
}

echo "═══════════════════════════════════════════════"
echo " Dispatch: All Experiment Waves"
echo " Model: Qwen/Qwen2.5-0.5B"
echo " Started: $(date)"
echo "═══════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────
# Wave 1: random(s42,s123,s777) + alg_clean(s42)
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Wave 1: random(s42,s123,s777) + alg_clean(s42) ==="
launch "exp1_algebraic_init/random/seed_42"     0 "--regime oracle --num_suffix_blocks 2 --init_method random --save_gramian_metrics --seed 42"
launch "exp1_algebraic_init/random/seed_123"    1 "--regime oracle --num_suffix_blocks 2 --init_method random --save_gramian_metrics --seed 123"
launch "exp1_algebraic_init/random/seed_777"    2 "--regime oracle --num_suffix_blocks 2 --init_method random --save_gramian_metrics --seed 777"
launch "exp1_algebraic_init/alg_clean/seed_42"  3 "--regime oracle --num_suffix_blocks 2 --init_method alg_clean --save_gramian_metrics --seed 42"
echo "Wave 1 launched."

# ─────────────────────────────────────────────────────────────────────
# Wave 2: alg_clean(s123,s777) + alg_aug(s42,s123)
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Waiting for Wave 1 to complete ==="
wait_all_gpus

echo "=== Wave 2: alg_clean(s123,s777) + alg_aug(s42,s123) ==="
launch "exp1_algebraic_init/alg_clean/seed_123" 0 "--regime oracle --num_suffix_blocks 2 --init_method alg_clean --save_gramian_metrics --seed 123"
launch "exp1_algebraic_init/alg_clean/seed_777" 1 "--regime oracle --num_suffix_blocks 2 --init_method alg_clean --save_gramian_metrics --seed 777"
launch "exp1_algebraic_init/alg_aug/seed_42"    2 "--regime oracle --num_suffix_blocks 2 --init_method alg_aug --save_gramian_metrics --seed 42"
launch "exp1_algebraic_init/alg_aug/seed_123"   3 "--regime oracle --num_suffix_blocks 2 --init_method alg_aug --save_gramian_metrics --seed 123"
echo "Wave 2 launched."

# ─────────────────────────────────────────────────────────────────────
# Wave 3: alg_aug(s777) + Exp 2 pure-logits + Exp 3 wrong-teacher
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Waiting for Wave 2 to complete ==="
wait_all_gpus

echo "=== Wave 3: alg_aug(s777) + pure-logits + controls ==="
launch "exp1_algebraic_init/alg_aug/seed_777"  0 "--regime oracle --num_suffix_blocks 2 --init_method alg_aug --save_gramian_metrics --seed 777"
launch "exp2_gramian/pure_logits/seed_42"      1 "--regime pure_logits --num_suffix_blocks 3 --init_method random --seed 42"
launch "exp2_gramian/pure_logits/seed_123"     2 "--regime pure_logits --num_suffix_blocks 3 --init_method random --seed 123"
launch "exp3_controls/wrong_teacher/seed_42"   3 "--regime oracle --num_suffix_blocks 2 --init_method random --wrong_teacher --seed 42"
echo "Wave 3 launched."

# ─────────────────────────────────────────────────────────────────────
# Wave 4: Ablations + more pure-logits
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Waiting for Wave 3 to complete ==="
wait_all_gpus

echo "=== Wave 4: Ablations ==="
launch "exp3_controls/beta0_ablation/seed_42"  0 "--regime oracle --num_suffix_blocks 2 --init_method random --beta 0.0 --seed 42"
launch "exp3_controls/no_gauge/seed_42"        1 "--regime oracle --num_suffix_blocks 2 --init_method alg_clean --no_gramian_project_gauge --save_gramian_metrics --seed 42"
launch "exp2_gramian/pure_logits/seed_777"     2 "--regime pure_logits --num_suffix_blocks 3 --init_method random --seed 777"
launch "exp2_gramian/pure_logits/seed_999"     3 "--regime pure_logits --num_suffix_blocks 3 --init_method random --seed 999"
echo "Wave 4 launched."

# ─────────────────────────────────────────────────────────────────────
# Wave 5: Gramian offline evaluation
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Waiting for Wave 4 to complete ==="
wait_all_gpus

echo "=== Wave 5: Gramian offline analysis ==="
mkdir -p "$RESULTS/exp2_gramian/analysis"
nohup bash -c "
    cd $PROJ && source .venv/bin/activate
    export HF_ENDPOINT=https://hf-mirror.com
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CUDA_VISIBLE_DEVICES=0 python -u scripts/run_gramian_eval.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --results_dir $RESULTS \
        --output_dir $RESULTS/exp2_gramian/analysis \
        --num_probes 128 \
        --project_gauge
" > "$LOG_DIR/gramian_eval.log" 2>&1 &
echo "  PID=$!"
echo "Wave 5 (Gramian eval) launched."

echo ""
echo "═══════════════════════════════════════════════"
echo " ALL WAVES DISPATCHED"
echo " Finished: $(date)"
echo "═══════════════════════════════════════════════"
