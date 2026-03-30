#!/bin/bash
# ============================================================================
# S-PSI ACP Training — Startup Script
#
# Startup command:
#   bash /data/szs/250010072/nwh/nips-modelsteal/run_acp.sh
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

# ========== PRE-FLIGHT ==========
python -c "
import torch, sys
print(f'Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
n = torch.cuda.device_count()
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} ({p.total_mem / 1e9:.0f}GB)')
import transformers, peft, accelerate
print(f'Transformers {transformers.__version__}, PEFT {peft.__version__}, Accelerate {accelerate.__version__}')
"

ls ${SHARE_DIR}/Qwen3.5-0.8B/config.json 2>/dev/null \
    && echo "[ok] Model Qwen3.5-0.8B found." \
    || echo "[WARN] Model not at ${SHARE_DIR}/Qwen3.5-0.8B!"

# ========== RUN EXPERIMENTS ==========
export SPSI_CONFIG="${PROJECT_DIR}/configs/inversion_server.yaml"

echo ""
echo "Config: ${SPSI_CONFIG}"
echo "Output: ${DATA_DIR}"
echo ""

bash scripts/run_all_experiments.sh 2>&1

echo ""
echo "Done: $(date)"
echo "Log: ${LOG}"
