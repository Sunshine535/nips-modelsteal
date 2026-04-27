#!/bin/bash
# Phase 1: C-DART Multi-Model Validation on Qwen3
# 4x A100-80GB parallel: GPU0=0.6B, GPU1=1.7B, GPU2-3=4B
# Usage: bash scripts/launch_phase1_qwen3.sh

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source .venv/bin/activate

mkdir -p logs teachers

echo "=============================="
echo "Phase 1: C-DART on Qwen3"
echo "Start: $(date)"
echo "=============================="

# Step 1: Download all models (sequential, network-bound)
echo "[$(date +%H:%M)] Downloading Qwen3 models..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
for m in ['Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B', 'Qwen/Qwen3-4B']:
    print(f'Downloading {m}...')
    AutoTokenizer.from_pretrained(m)
    AutoModelForCausalLM.from_pretrained(m, torch_dtype='auto')
    print(f'  {m} cached.')
print('All models downloaded.')
"
echo "[$(date +%H:%M)] All models cached."

# Step 2: Create delta teachers (sequential on GPU 0, fast)
echo "[$(date +%H:%M)] Creating delta teachers..."

for model_tag in "Qwen/Qwen3-0.6B:teachers/qwen3_0.6b_v1" \
                 "Qwen/Qwen3-1.7B:teachers/qwen3_1.7b_v1" \
                 "Qwen/Qwen3-4B:teachers/qwen3_4b_v1"; do
    MODEL="${model_tag%%:*}"
    OUTDIR="${model_tag##*:}"
    if [ -f "${OUTDIR}/config.json" ]; then
        echo "  Teacher exists: ${OUTDIR}, skipping."
    else
        echo "  Creating teacher: ${MODEL} -> ${OUTDIR}"
        CUDA_VISIBLE_DEVICES=0 python3 scripts/create_delta_teacher.py \
            --base_model "$MODEL" \
            --finetune_steps 500 \
            --finetune_lr 5e-5 \
            --output_dir "$OUTDIR" \
            --device cuda:0
    fi
done
echo "[$(date +%H:%M)] All teachers ready."

# Step 3: Launch C-DART experiments in parallel across GPUs
echo "[$(date +%H:%M)] Launching C-DART gate experiments..."

# GPU 0: Qwen3-0.6B (3 seeds, ~6h)
CUDA_VISIBLE_DEVICES=0 nohup python3 scripts/run_tdart.py \
    --config configs/cdart_qwen3_0.6b.yaml \
    --seeds 0 1 2 \
    --output_dir results/cdart_qwen3_0.6b \
    --device cuda:0 \
    > logs/cdart_qwen3_0.6b.log 2>&1 &
PID_06B=$!
echo "  GPU 0: Qwen3-0.6B (PID=$PID_06B)"

sleep 5

# GPU 1: Qwen3-1.7B (3 seeds, ~18h)
CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/run_tdart.py \
    --config configs/cdart_qwen3_1.7b.yaml \
    --seeds 0 1 2 \
    --output_dir results/cdart_qwen3_1.7b \
    --device cuda:0 \
    > logs/cdart_qwen3_1.7b.log 2>&1 &
PID_17B=$!
echo "  GPU 1: Qwen3-1.7B (PID=$PID_17B)"

sleep 5

# GPU 2: Qwen3-4B seed 0+1 (each ~12h)
CUDA_VISIBLE_DEVICES=2 nohup python3 scripts/run_tdart.py \
    --config configs/cdart_qwen3_4b.yaml \
    --seeds 0 1 \
    --output_dir results/cdart_qwen3_4b \
    --device cuda:0 \
    > logs/cdart_qwen3_4b_s01.log 2>&1 &
PID_4B_01=$!
echo "  GPU 2: Qwen3-4B seeds 0,1 (PID=$PID_4B_01)"

sleep 5

# GPU 3: Qwen3-4B seed 2
CUDA_VISIBLE_DEVICES=3 nohup python3 scripts/run_tdart.py \
    --config configs/cdart_qwen3_4b.yaml \
    --seeds 2 \
    --output_dir results/cdart_qwen3_4b \
    --device cuda:0 \
    > logs/cdart_qwen3_4b_s2.log 2>&1 &
PID_4B_2=$!
echo "  GPU 3: Qwen3-4B seed 2 (PID=$PID_4B_2)"

echo ""
echo "=============================="
echo "All jobs launched. PIDs:"
echo "  Qwen3-0.6B: $PID_06B"
echo "  Qwen3-1.7B: $PID_17B"
echo "  Qwen3-4B s0+1: $PID_4B_01"
echo "  Qwen3-4B s2: $PID_4B_2"
echo "=============================="
echo ""
echo "Monitor:"
echo "  tail -f logs/cdart_qwen3_0.6b.log"
echo "  tail -f logs/cdart_qwen3_1.7b.log"
echo "  tail -f logs/cdart_qwen3_4b_s01.log"
echo "  tail -f logs/cdart_qwen3_4b_s2.log"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "Estimated completion:"
echo "  Qwen3-0.6B:  ~6h  ($(date -d '+6 hours' +%H:%M 2>/dev/null || echo 'N/A'))"
echo "  Qwen3-1.7B:  ~18h ($(date -d '+18 hours' +%H:%M 2>/dev/null || echo 'N/A'))"
echo "  Qwen3-4B:    ~24h ($(date -d '+24 hours' +%H:%M 2>/dev/null || echo 'N/A'))"
