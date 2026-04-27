#!/bin/bash
# Round 2 fixes: rerun baselines with fixed KL normalization + K sweep
# 4x A100 parallel
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs

echo "=============================="
echo "Round 2: Baseline Rerun + K Sweep"
echo "Start: $(date)"
echo "=============================="

# GPU 0: Rerun strict_topk_kd + full_logit_upper on Qwen2.5-0.5B (3 seeds)
CUDA_VISIBLE_DEVICES=0 nohup python3 scripts/run_tdart.py \
    --config configs/cdart_gate_v1.yaml \
    --seeds 0 1 2 \
    --output_dir results/cdart_qwen25_fixed \
    --variants strict_topk_kd full_logit_upper \
    --device cuda:0 \
    > logs/round2_qwen25_fixed.log 2>&1 &
echo "  GPU 0: Qwen2.5 baselines rerun (PID=$!)"

sleep 3

# GPU 1: Rerun on Llama-3.2-1B
CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/run_tdart.py \
    --config configs/cdart_llama_1b.yaml \
    --seeds 0 1 2 \
    --output_dir results/cdart_llama_1b_fixed \
    --variants strict_topk_kd full_logit_upper \
    --device cuda:0 \
    > logs/round2_llama1b_fixed.log 2>&1 &
echo "  GPU 1: Llama-1B baselines rerun (PID=$!)"

sleep 3

# GPU 2: K sweep on Qwen3-0.6B — K=5, K=10, K=50
CUDA_VISIBLE_DEVICES=2 nohup bash -c '
cd '"$(pwd)"'
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for K in 5 10 50; do
    echo "[$(date +%H:%M)] Running K=$K sweep on Qwen3-0.6B..."
    python3 scripts/run_tdart.py \
        --base_model models/Qwen3-0.6B \
        --teacher_checkpoint_path teachers/qwen3_0.6b_v1 \
        --seeds 0 1 2 \
        --output_dir results/cdart_qwen3_K${K} \
        --num_steps 5000 --batch_size 8 --seq_len 128 --lr 2e-5 \
        --topk $K --K_student $K --K_reference $K \
        --lambda_ce 0.3 --lambda_rank 0.3 --lambda_delta 0.2 --lambda_censor 0.2 \
        --min_margin 0.1 --eval_every 500 --eval_batches 50 \
        --finetune_steps 500 --finetune_lr 5e-5 \
        --variants ce_only bild_topk_delta cdart_full full_logit_upper \
        --device cuda:0
    echo "[$(date +%H:%M)] K=$K done."
done
echo "[$(date +%H:%M)] K sweep COMPLETE."
' > logs/round2_k_sweep.log 2>&1 &
echo "  GPU 2: K sweep K={5,10,50} on Qwen3-0.6B (PID=$!)"

sleep 3

# GPU 3: Rerun on Llama-3.2-3B + Qwen3-0.6B baselines
CUDA_VISIBLE_DEVICES=3 nohup bash -c '
cd '"$(pwd)"'
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date +%H:%M)] Llama-3B baselines..."
python3 scripts/run_tdart.py \
    --config configs/cdart_llama_3b.yaml \
    --seeds 0 1 2 \
    --output_dir results/cdart_llama_3b_fixed \
    --variants strict_topk_kd full_logit_upper \
    --device cuda:0

echo "[$(date +%H:%M)] Qwen3-0.6B baselines..."
python3 scripts/run_tdart.py \
    --base_model models/Qwen3-0.6B \
    --teacher_checkpoint_path teachers/qwen3_0.6b_v1 \
    --seeds 0 1 2 \
    --output_dir results/cdart_qwen3_fixed \
    --num_steps 5000 --batch_size 8 --seq_len 128 --lr 2e-5 \
    --topk 20 --K_student 20 --K_reference 20 \
    --eval_every 500 --eval_batches 50 \
    --finetune_steps 500 --finetune_lr 5e-5 \
    --variants strict_topk_kd full_logit_upper \
    --device cuda:0

echo "[$(date +%H:%M)] GPU 3 DONE."
' > logs/round2_gpu3.log 2>&1 &
echo "  GPU 3: Llama-3B + Qwen3 baselines rerun (PID=$!)"

echo ""
echo "=============================="
echo "All Round 2 jobs launched at $(date)"
echo "Monitor: tail -f logs/round2_*.log"
echo "=============================="
