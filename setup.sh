#!/bin/bash
set -euo pipefail

ENV_NAME="${1:-modelsteal}"
PYTHON_VERSION="3.10"

echo "============================================"
echo " Progressive Parameter Inversion — One-Click Setup"
echo " Conda env: $ENV_NAME | Python $PYTHON_VERSION"
echo "============================================"

# Step 1: Create conda environment
if conda info --envs 2>/dev/null | grep -q "^${ENV_NAME} "; then
    echo "[SKIP] Conda env '$ENV_NAME' already exists."
else
    echo "[1/4] Creating conda environment..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "  Python: $(python --version)"

# Step 2: Install PyTorch with CUDA 12.1
echo "[2/4] Installing PyTorch 2.4.0 + CUDA 12.1..."
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install project dependencies
echo "[3/4] Installing project dependencies..."
pip install -r requirements.txt

# Step 4: Optional flash-attn
echo "[4/4] Installing flash-attn (optional, may take a few minutes)..."
pip install flash-attn --no-build-isolation 2>/dev/null && \
    echo "  flash-attn installed successfully." || \
    echo "  [WARN] flash-attn installation failed — not critical, continuing."

# Verify CUDA
echo ""
echo "============================================"
echo " Verifying CUDA availability"
echo "============================================"
python -c "
import torch
print(f'  PyTorch version : {torch.__version__}')
print(f'  CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version    : {torch.version.cuda}')
    print(f'  GPU count       : {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}           : {torch.cuda.get_device_name(i)}')
else:
    print('  [WARN] No CUDA GPUs detected. CPU-only mode.')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo " Usage:"
echo "   conda activate $ENV_NAME"
echo "   bash scripts/run_all_experiments.sh"
echo "============================================"
