#!/bin/bash
# V4 Experiment Dispatch — 4x A100-80GB
# Addresses 4 critical weaknesses: KD baseline, Llama, active query, claim narrowing
#
# GPU allocation:
#   GPU 0: KD suffix baseline (Qwen2.5-0.5B)
#   GPU 1: Llama-3.2-1B S-PSI (oracle, alg_clean, 1 seed)
#   GPU 2: Llama-3.2-1B Gramian diagnostic
#   GPU 3: Active query experiment (Qwen2.5-0.5B)

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com

LOGDIR="logs"
mkdir -p "$LOGDIR"

echo "=== V4 Experiment Dispatch ==="
echo "Time: $(date)"
echo ""

# --- GPU 0: KD Suffix Baseline ---
echo "[GPU 0] Launching KD suffix baseline..."
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/run_kd_suffix_baseline.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --num_suffix_blocks 2 \
    --pool_size 512 \
    --max_steps 2000 \
    --learning_rate 1e-4 \
    --logit_suffix_positions 32 \
    --batch_size 4 \
    --num_inits 3 \
    --seed 42 \
    --output_dir results/v4_kd_suffix_baseline \
    > "$LOGDIR/v4_kd_suffix_baseline.log" 2>&1 &
PID_KD=$!
echo "  PID=$PID_KD"

sleep 5

# --- GPU 1: Llama-3.2-1B S-PSI ---
echo "[GPU 1] Launching Llama-3.2-1B S-PSI..."
CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/run_spsi.py \
    --model_name meta-llama/Llama-3.2-1B \
    --regime oracle \
    --num_inits 1 \
    --num_suffix_blocks 2 \
    --query_budget 500000 \
    --pool_size 512 \
    --max_seq_len 128 \
    --batch_size 4 \
    --max_steps 2000 \
    --learning_rate 1e-4 \
    --alpha 1.0 --beta 0.0 --gamma 1e-5 \
    --num_perturbation_positions 1 \
    --num_replacement_tokens 1 \
    --logit_suffix_positions 32 \
    --init_method alg_clean \
    --init_num_probes 128 \
    --init_truncation_rank 64 \
    --heldout_fraction 0.2 \
    --allow_synthetic \
    --seed 42 \
    --output_dir results/v4_llama_spsi \
    > "$LOGDIR/v4_llama_spsi.log" 2>&1 &
PID_LLAMA=$!
echo "  PID=$PID_LLAMA"

sleep 5

# --- GPU 2: Llama-3.2-1B Gramian Diagnostic ---
echo "[GPU 2] Launching Llama-3.2-1B Gramian diagnostic..."
CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/diagnose_gramian_rank.py \
    --model_name meta-llama/Llama-3.2-1B \
    --block_idx 15 \
    --K_values 8 32 \
    --k_values 128 \
    --pool_size 256 \
    --max_seq_len 128 \
    --query_subsample 128 \
    --seed 42 \
    --output_dir results/v4_llama_gramian \
    > "$LOGDIR/v4_llama_gramian.log" 2>&1 &
PID_GRAM=$!
echo "  PID=$PID_GRAM"

sleep 5

# --- GPU 3: Active Query Experiment ---
echo "[GPU 3] Launching active query experiment..."
CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/run_active_query.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --target_block 23 \
    --pool_size 2048 \
    --active_select 256 \
    --max_seq_len 128 \
    --batch_size 4 \
    --max_steps 2000 \
    --learning_rate 1e-4 \
    --logit_suffix_positions 32 \
    --num_probes 128 \
    --gramian_query_subsample 256 \
    --num_target_directions 16 \
    --seed 42 \
    --output_dir results/v4_active_query \
    > "$LOGDIR/v4_active_query.log" 2>&1 &
PID_AQ=$!
echo "  PID=$PID_AQ"

echo ""
echo "All experiments launched."
echo "PIDs: KD=$PID_KD LLAMA=$PID_LLAMA GRAM=$PID_GRAM AQ=$PID_AQ"
echo ""
echo "Monitor with:"
echo "  tail -f $LOGDIR/v4_kd_suffix_baseline.log"
echo "  tail -f $LOGDIR/v4_llama_spsi.log"
echo "  tail -f $LOGDIR/v4_llama_gramian.log"
echo "  tail -f $LOGDIR/v4_active_query.log"
echo ""
echo "Check completion:"
echo "  ps aux | grep 'run_kd_suffix\|run_spsi\|run_gramian\|run_active'"
