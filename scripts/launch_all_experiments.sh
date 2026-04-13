#!/bin/bash
# =============================================================================
# Transformer Tomography: Full Experiment Pipeline
# Hardware: 4× A100-80GB
# Budget: ~100 GPU-hours (~25 wall-clock hours)
# =============================================================================
set -e

PROJ="/mnt/ae22ef98-e02a-4dfa-acf4-a307e8941cd9/nwh/nips/nips-modelsteal"
RESULTS="$PROJ/results"
VENV="$PROJ/.venv/bin/activate"
LOG_DIR="$PROJ/logs"
mkdir -p "$LOG_DIR" "$RESULTS"

export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Shared settings
MODEL="Qwen/Qwen3.5-0.8B"
POOL_SIZE=4096
MAX_SEQ_LEN=192
LOGIT_K=8
BATCH=16
LM_HEAD_STEPS=1500
MAX_STEPS=3000
PATIENCE=500
QUERY_BUDGET=500000

echo "============================================"
echo " Transformer Tomography Experiments"
echo " $(date)"
echo "============================================"

# ─────────────────────────────────────────────────────────────────────────────
# EXP 1: Algebraic Init vs Random Init (MUST-RUN)
# 3 init methods × 3 seeds × 2 suffix blocks = 18 runs
# GPU allocation: 4 GPUs, 3 methods dispatched in waves
# ─────────────────────────────────────────────────────────────────────────────
launch_exp1() {
    local INIT_METHOD=$1
    local SEED=$2
    local GPU=$3
    local TAG="exp1_${INIT_METHOD}_s${SEED}"
    local OUT="$RESULTS/exp1_algebraic_init/${INIT_METHOD}/seed_${SEED}"
    mkdir -p "$OUT"

    echo "[$(date +%H:%M)] GPU $GPU: $TAG"
    CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "
        source $VENV
        python $PROJ/scripts/run_spsi.py \
            --model_name $MODEL \
            --regime oracle \
            --num_suffix_blocks 2 \
            --num_inits 1 \
            --init_offset 0 \
            --max_steps $MAX_STEPS \
            --lm_head_steps $LM_HEAD_STEPS \
            --batch_size $BATCH \
            --query_budget $QUERY_BUDGET \
            --pool_size $POOL_SIZE \
            --max_seq_len $MAX_SEQ_LEN \
            --logit_suffix_positions $LOGIT_K \
            --allow_synthetic \
            --patience $PATIENCE \
            --init_method $INIT_METHOD \
            --save_gramian_metrics \
            --output_dir $OUT \
            --seed $SEED
    " > "$LOG_DIR/${TAG}.log" 2>&1 &
    echo $! > "$LOG_DIR/${TAG}.pid"
}

echo ""
echo "=== EXP 1: Algebraic Init vs Random Init ==="
echo "Wave 1: random(s42,s123,s777) + alg_clean(s42) on GPU 0,1,2,3"

launch_exp1 random     42  0
launch_exp1 random    123  1
launch_exp1 random    777  2
launch_exp1 alg_clean  42  3

echo "Wave 1 launched. Remaining waves will be dispatched by monitor."
echo "PIDs saved to $LOG_DIR/exp1_*.pid"

# ─────────────────────────────────────────────────────────────────────────────
# Monitor script: launches remaining waves when GPUs free up
# ─────────────────────────────────────────────────────────────────────────────
cat > "$PROJ/scripts/dispatch_waves.sh" << 'DISPATCH_EOF'
#!/bin/bash
# Auto-dispatch remaining experiment waves when GPUs become available
PROJ="/mnt/ae22ef98-e02a-4dfa-acf4-a307e8941cd9/nwh/nips/nips-modelsteal"
RESULTS="$PROJ/results"
VENV="$PROJ/.venv/bin/activate"
LOG_DIR="$PROJ/logs"
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen3.5-0.8B"
POOL_SIZE=4096
MAX_SEQ_LEN=192
LOGIT_K=8
BATCH=16
LM_HEAD_STEPS=1500
MAX_STEPS=3000
PATIENCE=500
QUERY_BUDGET=500000

launch() {
    local TAG=$1 GPU=$2 EXTRA_ARGS=$3
    local OUT="$RESULTS/$TAG"
    mkdir -p "$OUT"
    echo "[$(date +%H:%M)] GPU $GPU: $TAG"
    CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "
        source $VENV
        python $PROJ/scripts/run_spsi.py \
            --model_name $MODEL \
            --regime oracle \
            --num_suffix_blocks 2 \
            --num_inits 1 \
            --max_steps $MAX_STEPS \
            --lm_head_steps $LM_HEAD_STEPS \
            --batch_size $BATCH \
            --query_budget $QUERY_BUDGET \
            --pool_size $POOL_SIZE \
            --max_seq_len $MAX_SEQ_LEN \
            --logit_suffix_positions $LOGIT_K \
            --allow_synthetic \
            --patience $PATIENCE \
            $EXTRA_ARGS \
            --output_dir $OUT
    " > "$LOG_DIR/${TAG//\//_}.log" 2>&1 &
    echo $! > "$LOG_DIR/${TAG//\//_}.pid"
}

wait_for_gpu() {
    local GPU=$1
    while true; do
        local MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null)
        if [ "$MEM" -lt 1000 ] 2>/dev/null; then
            return 0
        fi
        sleep 60
    done
}

echo "=== Wave 2: alg_clean(s123,s777) + alg_aug(s42,s123) ==="
for GPU in 0 1 2 3; do wait_for_gpu $GPU; done

launch "exp1_algebraic_init/alg_clean/seed_123" 0 "--init_method alg_clean --seed 123"
launch "exp1_algebraic_init/alg_clean/seed_777" 1 "--init_method alg_clean --seed 777"
launch "exp1_algebraic_init/alg_aug/seed_42"    2 "--init_method alg_aug --seed 42"
launch "exp1_algebraic_init/alg_aug/seed_123"   3 "--init_method alg_aug --seed 123"

echo "=== Wave 3: alg_aug(s777) + Exp 2 pure_logits runs ==="
for GPU in 0 1 2 3; do wait_for_gpu $GPU; done

launch "exp1_algebraic_init/alg_aug/seed_777" 0 "--init_method alg_aug --seed 777"

# Exp 2: Pure-logits regime (for Gramian correlation)
launch "exp2_gramian/pure_logits/seed_42"  1 "--init_method random --regime pure_logits --num_suffix_blocks 3 --seed 42"
launch "exp2_gramian/pure_logits/seed_123" 2 "--init_method random --regime pure_logits --num_suffix_blocks 3 --seed 123"

# Exp 3: Wrong-teacher control
launch "exp3_controls/wrong_teacher/seed_42" 3 "--init_method random --wrong_teacher --seed 42"

echo "=== Wave 4: More controls + ablations ==="
for GPU in 0 1 2 3; do wait_for_gpu $GPU; done

# Exp 3: beta=0 ablation
launch "exp3_controls/beta0_ablation/seed_42" 0 "--init_method random --beta 0.0 --seed 42"

# Exp 3: No gauge projection ablation
launch "exp3_controls/no_gauge/seed_42" 1 "--init_method alg_clean --no_gramian_project_gauge --save_gramian_metrics --seed 42"

# Exp 2: Extra pure-logits
launch "exp2_gramian/pure_logits/seed_777" 2 "--init_method random --regime pure_logits --num_suffix_blocks 3 --seed 777"
launch "exp2_gramian/pure_logits/seed_999" 3 "--init_method random --regime pure_logits --num_suffix_blocks 3 --seed 999"

echo "=== Wave 5: Gramian eval (offline) ==="
for GPU in 0 1 2 3; do wait_for_gpu $GPU; done

CUDA_VISIBLE_DEVICES=0 nohup bash -c "
    source $VENV
    python $PROJ/scripts/run_gramian_eval.py \
        --model_name $MODEL \
        --results_dir $RESULTS \
        --output_dir $RESULTS/exp2_gramian/analysis \
        --num_probes 128 \
        --project_gauge
" > "$LOG_DIR/gramian_eval.log" 2>&1 &

echo ""
echo "=== ALL EXPERIMENTS DISPATCHED ==="
echo "Monitor with: tail -f $LOG_DIR/*.log"
DISPATCH_EOF
chmod +x "$PROJ/scripts/dispatch_waves.sh"

echo ""
echo "=== Experiment launcher ready ==="
echo "Wave 1 running on GPUs 0-3"
echo "Auto-dispatch remaining waves: nohup bash $PROJ/scripts/dispatch_waves.sh > $LOG_DIR/dispatch.log 2>&1 &"
echo ""
echo "Monitor: tail -f $LOG_DIR/exp1_*.log"
echo "Status:  nvidia-smi"
