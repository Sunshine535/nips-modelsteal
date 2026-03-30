#!/bin/bash
# ============================================================================
# S-PSI ACP Training — Startup Script
#
# Startup command:
#   bash /data/szs/250010072/nwh/nips-modelsteal/run_acp.sh
#
# Expected Docker: PyTorch 2.10, Transformers 5.3, TRL, PEFT, DeepSpeed,
#                  Accelerate (pip install if missing)
# ============================================================================
set -euo pipefail

PROJECT_DIR=/data/szs/250010072/nwh/nips-modelsteal
DATA_DIR=/data/szs/share/modelsteal
SHARE_DIR=/data/szs/share

LOG_DIR=${DATA_DIR}/logs
mkdir -p ${DATA_DIR}/{checkpoints,results,logs}
LOG="${LOG_DIR}/acp_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================"
echo " S-PSI ACP Training"
echo " $(date) | $(hostname)"
echo " GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "============================================"

export HF_HOME="${DATA_DIR}/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd ${PROJECT_DIR}

# ========== SYMLINKS ==========
for dir in results logs; do
    if [ ! -e "${PROJECT_DIR}/${dir}" ] || [ -L "${PROJECT_DIR}/${dir}" ]; then
        ln -sfn "${DATA_DIR}/${dir}" "${PROJECT_DIR}/${dir}"
        echo "[symlink] ${PROJECT_DIR}/${dir} -> ${DATA_DIR}/${dir}"
    else
        echo "[skip] ${PROJECT_DIR}/${dir} already exists (not a symlink)"
    fi
done

# ========== PRE-FLIGHT ==========
python -c "
import torch, sys
print(f'Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
n = torch.cuda.device_count()
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} ({p.total_memory / 1e9:.0f}GB)')
import transformers, peft, accelerate
print(f'Transformers {transformers.__version__}, PEFT {peft.__version__}, Accelerate {accelerate.__version__}')
"

MODEL_DIR="${SHARE_DIR}/Qwen3.5-0.8B"
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    for alt in Qwen3.5-0.8B Qwen2.5-0.5B Qwen3-0.6B; do
        if [ -f "${SHARE_DIR}/${alt}/config.json" ]; then
            MODEL_DIR="${SHARE_DIR}/${alt}"
            echo "[ok] Model found at alt path: ${MODEL_DIR}"
            break
        fi
    done
fi
ls "${MODEL_DIR}/config.json" 2>/dev/null \
    && echo "[ok] Model found: ${MODEL_DIR}" \
    || { echo "[FATAL] No model found under ${SHARE_DIR}!"; exit 1; }

# ========== RUN EXPERIMENTS ==========
export SPSI_CONFIG="${PROJECT_DIR}/configs/inversion_server.yaml"

echo ""
echo "Config: ${SPSI_CONFIG}"
echo "Output: ${DATA_DIR}"
echo ""

bash scripts/run_all_experiments.sh --config "${SPSI_CONFIG}" 2>&1

echo ""
echo "Done: $(date)"
echo "Log: ${LOG}"
