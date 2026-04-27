#!/bin/bash
# Phase 1: C-DART Multi-Model Validation — cached models
# Llama-3.2-1B (GPU0), Llama-3.2-3B (GPU1+2), Qwen3-0.6B (GPU3, if downloaded)
# Qwen3 download runs in background via modelscope
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source .venv/bin/activate
mkdir -p logs teachers

echo "=============================="
echo "Phase 1: C-DART Multi-Model"
echo "Start: $(date)"
echo "=============================="

# Step 1: Create delta teachers (offline, use cached models)
echo "[$(date +%H:%M)] Creating delta teachers..."

create_teacher() {
    local MODEL="$1" OUTDIR="$2" GPU="$3"
    if [ -f "${OUTDIR}/config.json" ]; then
        echo "  Teacher exists: ${OUTDIR}, skipping."
        return 0
    fi
    echo "  Creating teacher: ${MODEL} -> ${OUTDIR} on GPU ${GPU}"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/create_delta_teacher.py \
        --base_model "$MODEL" \
        --finetune_steps 500 \
        --finetune_lr 5e-5 \
        --output_dir "$OUTDIR" \
        --device cuda:0
}

create_teacher "meta-llama/Llama-3.2-1B" "teachers/llama_3.2_1b_v1" 0
create_teacher "meta-llama/Llama-3.2-3B" "teachers/llama_3.2_3b_v1" 1

echo "[$(date +%H:%M)] Teachers ready."

# Step 2: Launch experiments
echo "[$(date +%H:%M)] Launching C-DART experiments..."

# GPU 0: Llama-3.2-1B, 3 seeds (~8h)
CUDA_VISIBLE_DEVICES=0 nohup python3 scripts/run_tdart.py \
    --config configs/cdart_llama_1b.yaml \
    --seeds 0 1 2 \
    --output_dir results/cdart_llama_1b \
    --device cuda:0 \
    > logs/cdart_llama_1b.log 2>&1 &
PID_1B=$!
echo "  GPU 0: Llama-3.2-1B (PID=$PID_1B)"

sleep 5

# GPU 1: Llama-3.2-3B seed 0+1 (~20h)
CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/run_tdart.py \
    --config configs/cdart_llama_3b.yaml \
    --seeds 0 1 \
    --output_dir results/cdart_llama_3b \
    --device cuda:0 \
    > logs/cdart_llama_3b_s01.log 2>&1 &
PID_3B_01=$!
echo "  GPU 1: Llama-3.2-3B seeds 0,1 (PID=$PID_3B_01)"

sleep 5

# GPU 2: Llama-3.2-3B seed 2 (~10h)
CUDA_VISIBLE_DEVICES=2 nohup python3 scripts/run_tdart.py \
    --config configs/cdart_llama_3b.yaml \
    --seeds 2 \
    --output_dir results/cdart_llama_3b \
    --device cuda:0 \
    > logs/cdart_llama_3b_s2.log 2>&1 &
PID_3B_2=$!
echo "  GPU 2: Llama-3.2-3B seed 2 (PID=$PID_3B_2)"

# GPU 3: Try Qwen3 download + run (if available)
echo "[$(date +%H:%M)] GPU 3: Attempting Qwen3-0.6B download via modelscope..."
CUDA_VISIBLE_DEVICES=3 nohup bash -c '
cd '"$PROJECT_DIR"'
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Try to install modelscope and download
pip install -q modelscope 2>/dev/null
python3 -c "
from modelscope import snapshot_download
import os, shutil

print(\"Downloading Qwen3-0.6B from ModelScope...\")
local_dir = snapshot_download(\"Qwen/Qwen3-0.6B\", cache_dir=\"/tmp/modelscope_cache\")
print(f\"Downloaded to: {local_dir}\")

# Symlink to HF cache format so transformers can find it
hf_cache = os.path.expanduser(\"~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B\")
os.makedirs(hf_cache + \"/snapshots\", exist_ok=True)
os.makedirs(hf_cache + \"/blobs\", exist_ok=True)
# Copy config to let transformers find it
link_target = hf_cache + \"/snapshots/modelscope\"
if not os.path.exists(link_target):
    os.symlink(local_dir, link_target)
print(\"Linked to HF cache.\")
" 2>&1

if [ $? -eq 0 ]; then
    echo "[$(date +%H:%M)] Qwen3-0.6B downloaded. Creating teacher..."
    CUDA_VISIBLE_DEVICES=0 python3 scripts/create_delta_teacher.py \
        --base_model "/tmp/modelscope_cache/Qwen/Qwen3-0.6B" \
        --finetune_steps 500 --finetune_lr 5e-5 \
        --output_dir teachers/qwen3_0.6b_v1 --device cuda:0

    echo "[$(date +%H:%M)] Running C-DART on Qwen3-0.6B..."
    python3 scripts/run_tdart.py \
        --base_model "/tmp/modelscope_cache/Qwen/Qwen3-0.6B" \
        --teacher_checkpoint_path teachers/qwen3_0.6b_v1 \
        --seeds 0 1 2 \
        --output_dir results/cdart_qwen3_0.6b \
        --num_steps 5000 --batch_size 8 --seq_len 128 --lr 2e-5 \
        --topk 20 --K_student 20 --K_reference 20 \
        --lambda_ce 0.3 --lambda_rank 0.3 --lambda_delta 0.2 --lambda_censor 0.2 \
        --min_margin 0.1 --eval_every 500 --eval_batches 50 \
        --finetune_steps 500 --finetune_lr 5e-5 \
        --variants ce_only strict_topk_kd bild_topk_delta tdart_no_adaptive cdart_no_censor cdart_full full_logit_upper \
        --device cuda:0
else
    echo "[$(date +%H:%M)] Qwen3-0.6B download failed. GPU 3 idle."
fi
' > logs/cdart_qwen3_0.6b.log 2>&1 &
PID_Q3=$!
echo "  GPU 3: Qwen3-0.6B download+run attempt (PID=$PID_Q3)"

echo ""
echo "=============================="
echo "All jobs launched at $(date)"
echo "  Llama-3.2-1B:  PID=$PID_1B  (GPU 0, ~8h)"
echo "  Llama-3.2-3B:  PID=$PID_3B_01 + $PID_3B_2  (GPU 1+2, ~20h)"
echo "  Qwen3-0.6B:    PID=$PID_Q3  (GPU 3, download+~6h)"
echo "=============================="
echo ""
echo "Monitor:"
echo "  tail -f logs/cdart_llama_1b.log"
echo "  tail -f logs/cdart_llama_3b_s01.log"
echo "  tail -f logs/cdart_qwen3_0.6b.log"
echo "  nvidia-smi"
