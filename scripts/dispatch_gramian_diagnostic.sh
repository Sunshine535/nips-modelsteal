#!/bin/bash
# Dispatch Gramian rank diagnostic to remote server
# Tests whether rank=32 is observation artifact or structural limit
#
# Usage: bash scripts/dispatch_gramian_diagnostic.sh
# Prerequisite: SSH access to remote server + code synced

set -euo pipefail

REMOTE_HOST="kemove@100.103.219.5"
REMOTE_DIR="/mnt/ae22ef98-e02a-4dfa-acf4-a307e8941cd9/nwh/nips/nips-modelsteal"
SSH_KEY="$HOME/.ssh/id_ed25519_windows_remote"

echo "=== Gramian Rank Diagnostic ==="
echo "Testing: Is rank=32 a structural limit or observation artifact?"
echo ""

# ---- Phase 1: Quick diagnostic (K=8 with more probes) ----
# Test whether k=64 was the bottleneck. If rank stays 32 with k=256, true rank IS 32.
# This takes ~30 minutes on 1 GPU.
echo "Phase 1: Probe scaling test (K=8, k={64,128,256})"
cat << 'PHASE1_CMD'
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/ae22ef98-e02a-4dfa-acf4-a307e8941cd9/nwh/nips/nips-modelsteal
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python -u scripts/diagnose_gramian_rank.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --output_dir results/gramian_diagnostic_phase1 \
    --K_values 8 \
    --k_values 64 128 256 \
    --query_subsample 256 \
    --pool_size 2048 \
    --max_seq_len 128 \
    --block_idx 23 \
    > logs/gramian_diag_phase1.log 2>&1
echo "Phase 1 complete. Exit code: $?"
PHASE1_CMD

# ---- Phase 2: Position expansion (K scaling, k=256) ----
# Tests whether more positions expand the Gramian rank.
# K=32 and K=64 should be GPU-feasible. K=128 uses memory-efficient path.
# This takes ~2-4 hours on 1 GPU (K=128 is slowest).
echo ""
echo "Phase 2: Position scaling test (K={8,32,64,128}, k=256)"
cat << 'PHASE2_CMD'
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/ae22ef98-e02a-4dfa-acf4-a307e8941cd9/nwh/nips/nips-modelsteal
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python -u scripts/diagnose_gramian_rank.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --output_dir results/gramian_diagnostic_phase2 \
    --K_values 8 32 64 128 \
    --k_values 256 \
    --query_subsample 128 \
    --pool_size 2048 \
    --max_seq_len 128 \
    --block_idx 23 \
    --also_block_22 \
    > logs/gramian_diag_phase2.log 2>&1
echo "Phase 2 complete. Exit code: $?"
PHASE2_CMD

echo ""
echo "To run on server:"
echo "  1. Sync code:  rsync -avz --exclude .venv --exclude __pycache__ -e 'ssh -i ${SSH_KEY}' ./ ${REMOTE_HOST}:${REMOTE_DIR}/"
echo "  2. SSH in:     ssh -i ${SSH_KEY} ${REMOTE_HOST}"
echo "  3. Run Phase 1 (copy the PHASE1_CMD block above)"
echo "  4. Run Phase 2 in parallel on GPU 1 (copy the PHASE2_CMD block above)"
echo ""
echo "Expected results:"
echo "  If rank scales with K: observation bottleneck confirmed → proceed to expanded training"
echo "  If rank stays ~32:     structural limit → pivot to active queries"
