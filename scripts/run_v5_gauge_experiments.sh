#!/bin/bash
# V5 Gauge Experiment Dispatch — 4x A100-80GB
# Evaluates the full attention gauge symmetry (V/O GL(d_head) + Q/K RoPE C*)
#
# GPU allocation:
#   GPU 0: Gauge-invariant eval on v2_random_s42 results
#   GPU 1: Gauge-invariant eval on v2_alg_clean_s42 results
#   GPU 2: Gramian diagnostic with full gauge (Block 23, K=32, k=128)
#   GPU 3: Gramian diagnostic with full gauge (Block 22, K=32, k=128)
#
# Each saves to results/v5_gauge_*/ and logs to logs/v5_*.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com

MODEL="Qwen/Qwen2.5-0.5B"
LOGDIR="logs"
mkdir -p "$LOGDIR"

echo "=== V5 Gauge Experiment Dispatch ==="
echo "Time: $(date)"
echo "Model: $MODEL"
echo ""

# --- GPU 0: Gauge-invariant eval on v2_random_s42 ---
echo "[GPU 0] Launching gauge-invariant eval on v2_random_s42..."
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/eval_gauge_invariant.py \
    --results_dir results/v2_random \
    --model_name "$MODEL" \
    --block_indices 22,23 \
    --output_dir results/v5_gauge_eval_random_s42 \
    > "$LOGDIR/v5_gauge_eval_random.log" 2>&1 &
PID_G0=$!
echo "  PID=$PID_G0"

sleep 3

# --- GPU 1: Gauge-invariant eval on v2_alg_clean_s42 ---
echo "[GPU 1] Launching gauge-invariant eval on v2_alg_clean_s42..."
CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/eval_gauge_invariant.py \
    --results_dir results/v2_alg_clean_s42 \
    --model_name "$MODEL" \
    --block_indices 22,23 \
    --output_dir results/v5_gauge_eval_alg_clean_s42 \
    > "$LOGDIR/v5_gauge_eval_alg_clean.log" 2>&1 &
PID_G1=$!
echo "  PID=$PID_G1"

sleep 3

# --- GPU 2: Gramian with full gauge — Block 23 ---
echo "[GPU 2] Launching Gramian diagnostic with full gauge (Block 23)..."
CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/diagnose_gramian_rank.py \
    --model_name "$MODEL" \
    --block_idx 23 \
    --K_values 32 \
    --k_values 128 \
    --pool_size 2048 \
    --max_seq_len 128 \
    --query_subsample 256 \
    --seed 42 \
    --output_dir results/v5_gauge_gramian_block23 \
    > "$LOGDIR/v5_gramian_block23.log" 2>&1 &
PID_G2=$!
echo "  PID=$PID_G2"

sleep 3

# --- GPU 3: Gramian with full gauge — Block 22 ---
echo "[GPU 3] Launching Gramian diagnostic with full gauge (Block 22)..."
CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/diagnose_gramian_rank.py \
    --model_name "$MODEL" \
    --block_idx 22 \
    --K_values 32 \
    --k_values 128 \
    --pool_size 2048 \
    --max_seq_len 128 \
    --query_subsample 256 \
    --seed 42 \
    --output_dir results/v5_gauge_gramian_block22 \
    > "$LOGDIR/v5_gramian_block22.log" 2>&1 &
PID_G3=$!
echo "  PID=$PID_G3"

echo ""
echo "All V5 experiments launched."
echo "PIDs: EVAL_RANDOM=$PID_G0 EVAL_ALG=$PID_G1 GRAM_B23=$PID_G2 GRAM_B22=$PID_G3"
echo ""
echo "Monitor with:"
echo "  tail -f $LOGDIR/v5_gauge_eval_random.log"
echo "  tail -f $LOGDIR/v5_gauge_eval_alg_clean.log"
echo "  tail -f $LOGDIR/v5_gramian_block23.log"
echo "  tail -f $LOGDIR/v5_gramian_block22.log"
echo ""
echo "Check completion:"
echo "  ps aux | grep 'eval_gauge_invariant\|diagnose_gramian'"
echo ""
echo "Expected results:"
echo "  results/v5_gauge_eval_random_s42/gauge_invariant_eval.json"
echo "  results/v5_gauge_eval_alg_clean_s42/gauge_invariant_eval.json"
echo "  results/v5_gauge_gramian_block23/gramian_rank_diagnostic.json"
echo "  results/v5_gauge_gramian_block22/gramian_rank_diagnostic.json"
