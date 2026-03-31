#!/bin/bash
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="nips-modelsteal"

echo "============================================"
echo " Environment Setup (PyTorch 2.10 + CUDA 12.8)"
echo "============================================"

USE_CONDA=false

# --- Detect environment: prefer uv, fall back to conda ---
if command -v uv &>/dev/null; then
    echo "[1/5] uv found: $(uv --version)"
elif command -v conda &>/dev/null; then
    echo "[1/5] uv not found; using conda instead"
    USE_CONDA=true
else
    echo "[1/5] Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ "$USE_CONDA" = true ]; then
    # ── Conda path ──────────────────────────────────────────────────────
    CONDA_ENV="$PROJ_DIR/.conda_env"
    if [ ! -d "$CONDA_ENV" ]; then
        echo "[2/5] Creating conda env ($ENV_NAME, Python 3.10) ..."
        conda create -y -p "$CONDA_ENV" python=3.10
    else
        echo "[2/5] Conda env exists: $CONDA_ENV"
    fi

    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"

    echo "[3/5] Installing PyTorch 2.10.0 + CUDA 12.8 (conda) ..."
    pip install "torch==2.10.0" "torchvision" "torchaudio" \
        --index-url https://download.pytorch.org/whl/cu128

    echo "[4/5] Installing project dependencies ..."
    pip install -r "$PROJ_DIR/requirements.txt" \
        --extra-index-url https://download.pytorch.org/whl/cu128

    echo "[5/5] Installing flash-attn (optional) ..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn skipped (optional)"

    ACTIVATE_CMD="conda activate $CONDA_ENV"
else
    # ── uv path ─────────────────────────────────────────────────────────
    VENV_DIR="$PROJ_DIR/.venv"
    if [ ! -d "$VENV_DIR" ]; then
        echo "[2/5] Creating Python 3.10 venv ..."
        uv venv "$VENV_DIR" --python 3.10 2>/dev/null || uv venv "$VENV_DIR"
    else
        echo "[2/5] Venv exists: $VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"

    echo "[3/5] Installing PyTorch 2.10.0 + CUDA 12.8 ..."
    uv pip install "torch==2.10.0" "torchvision" "torchaudio" \
        --index-url https://download.pytorch.org/whl/cu128

    echo "[4/5] Installing project dependencies ..."
    uv pip install -r "$PROJ_DIR/requirements.txt" \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match

    echo "[5/5] Installing flash-attn (optional) ..."
    uv pip install flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn skipped (optional)"

    ACTIVATE_CMD="source $VENV_DIR/bin/activate"
fi

# --- Verify ---
echo ""
echo "============================================"
python -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.version.cuda}')
print(f'  GPUs     : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo "============================================"
echo ""
echo "Setup complete!"
echo "  Activate:  $ACTIVATE_CMD"
echo "  Run:       bash scripts/run_all_experiments.sh"
