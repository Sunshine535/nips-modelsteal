#!/bin/bash
# V3 Experiment Dispatch: Observation Space Expansion
#
# Tests whether rank=32 is an observation artifact or structural limit.
# Runs on 4×A100-80GB remote server.
#
# Wave 1 (GPU 0): Gramian diagnostic — probe scaling  (~30 min)
# Wave 2 (GPU 1): Gramian diagnostic — position scaling (~2-4 hours)
# Wave 3 (GPU 2): Expanded-K training (K=64, alg_clean, seed 42)  (~2 hours)
# Wave 4 (GPU 3): Expanded-K training (K=32, alg_clean, seed 42)  (~1 hour)
#
# After Wave 1-2 results: if rank increases → launch more training
#                          if rank stays 32 → pivot to active queries
#
# Usage:
#   1. Sync code: bash scripts/dispatch_v3_experiments.sh sync
#   2. Launch:    bash scripts/dispatch_v3_experiments.sh launch
#   3. Monitor:   bash scripts/dispatch_v3_experiments.sh monitor

set -euo pipefail

REMOTE_HOST="kemove@100.103.219.5"
REMOTE_DIR="/mnt/ae22ef98-e02a-4dfa-acf4-a307e8941cd9/nwh/nips/nips-modelsteal"
SSH_KEY="$HOME/.ssh/id_ed25519_windows_remote"
SSH_CMD="ssh -i ${SSH_KEY} ${REMOTE_HOST}"

ACTION="${1:-help}"

case "$ACTION" in
    sync)
        echo "=== Syncing code to remote server ==="
        rsync -avz --exclude .venv --exclude __pycache__ --exclude .git \
            --exclude 'results/' --exclude 'logs/' --exclude 'paper/' \
            -e "ssh -i ${SSH_KEY}" \
            ./ "${REMOTE_HOST}:${REMOTE_DIR}/"
        echo "Sync complete."
        ;;

    launch)
        echo "=== Launching V3 experiments ==="

        # Create logs directory
        ${SSH_CMD} "mkdir -p ${REMOTE_DIR}/logs"

        # Wave 1: Gramian diagnostic — probe scaling (GPU 0)
        echo "Launching Wave 1: Probe scaling diagnostic (GPU 0)..."
        ${SSH_CMD} "cd ${REMOTE_DIR} && nohup bash -c '
            export HF_ENDPOINT=https://hf-mirror.com
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            source .venv/bin/activate
            CUDA_VISIBLE_DEVICES=0 python -u scripts/diagnose_gramian_rank.py \
                --model_name Qwen/Qwen2.5-0.5B \
                --output_dir results/gramian_diagnostic_probes \
                --K_values 8 \
                --k_values 64 128 256 \
                --query_subsample 256 \
                --pool_size 2048 \
                --max_seq_len 128 \
                --block_idx 23
        ' > logs/v3_wave1_probe_diag.log 2>&1 &"
        echo "  Wave 1 launched (PID in nohup)"

        # Wave 2: Gramian diagnostic — position scaling (GPU 1)
        echo "Launching Wave 2: Position scaling diagnostic (GPU 1)..."
        ${SSH_CMD} "cd ${REMOTE_DIR} && nohup bash -c '
            export HF_ENDPOINT=https://hf-mirror.com
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            source .venv/bin/activate
            CUDA_VISIBLE_DEVICES=1 python -u scripts/diagnose_gramian_rank.py \
                --model_name Qwen/Qwen2.5-0.5B \
                --output_dir results/gramian_diagnostic_positions \
                --K_values 8 32 64 128 \
                --k_values 128 \
                --query_subsample 128 \
                --pool_size 2048 \
                --max_seq_len 128 \
                --block_idx 23 \
                --also_block_22
        ' > logs/v3_wave2_position_diag.log 2>&1 &"
        echo "  Wave 2 launched"

        # Wave 3: Expanded training K=64 (GPU 2)
        echo "Launching Wave 3: Expanded training K=64 (GPU 2)..."
        ${SSH_CMD} "cd ${REMOTE_DIR} && nohup bash -c '
            export HF_ENDPOINT=https://hf-mirror.com
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            source .venv/bin/activate
            CUDA_VISIBLE_DEVICES=2 python -u scripts/run_expanded_observation.py \
                --model_name Qwen/Qwen2.5-0.5B \
                --logit_suffix_positions 64 \
                --init_method alg_clean \
                --init_num_probes 128 \
                --init_truncation_rank 64 \
                --regime oracle \
                --max_steps 3000 \
                --pool_size 2048 \
                --batch_size 4 \
                --seed 42 \
                --output_dir results/v3_expanded_K64_s42
        ' > logs/v3_wave3_K64_s42.log 2>&1 &"
        echo "  Wave 3 launched"

        # Wave 4: Expanded training K=32 (GPU 3)
        echo "Launching Wave 4: Expanded training K=32 (GPU 3)..."
        ${SSH_CMD} "cd ${REMOTE_DIR} && nohup bash -c '
            export HF_ENDPOINT=https://hf-mirror.com
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            source .venv/bin/activate
            CUDA_VISIBLE_DEVICES=3 python -u scripts/run_expanded_observation.py \
                --model_name Qwen/Qwen2.5-0.5B \
                --logit_suffix_positions 32 \
                --init_method alg_clean \
                --init_num_probes 128 \
                --init_truncation_rank 64 \
                --regime oracle \
                --max_steps 3000 \
                --pool_size 2048 \
                --batch_size 4 \
                --seed 42 \
                --output_dir results/v3_expanded_K32_s42
        ' > logs/v3_wave4_K32_s42.log 2>&1 &"
        echo "  Wave 4 launched"

        echo ""
        echo "All 4 waves launched on 4 GPUs."
        echo "Monitor with: bash scripts/dispatch_v3_experiments.sh monitor"
        ;;

    monitor)
        echo "=== Monitoring V3 experiments ==="
        ${SSH_CMD} "
            echo '--- GPU Status ---'
            nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
            echo ''
            echo '--- Running Python processes ---'
            ps aux | grep 'python.*scripts/' | grep -v grep | awk '{print \$2, \$11, \$12, \$13}' || echo 'None running'
            echo ''
            echo '--- Log tails ---'
            for f in ${REMOTE_DIR}/logs/v3_wave*.log; do
                if [ -f \"\$f\" ]; then
                    echo \"=== \$(basename \$f) ===\"
                    tail -3 \"\$f\"
                    echo ''
                fi
            done
            echo '--- Result files ---'
            find ${REMOTE_DIR}/results -name '*.json' -newer ${REMOTE_DIR}/logs/v3_wave1_probe_diag.log 2>/dev/null | head -20 || echo 'No results yet'
        "
        ;;

    results)
        echo "=== Fetching V3 results ==="
        mkdir -p results/v3_remote
        rsync -avz -e "ssh -i ${SSH_KEY}" \
            "${REMOTE_HOST}:${REMOTE_DIR}/results/gramian_diagnostic_*/" \
            results/v3_remote/diagnostics/
        rsync -avz -e "ssh -i ${SSH_KEY}" \
            "${REMOTE_HOST}:${REMOTE_DIR}/results/v3_expanded_*/" \
            results/v3_remote/training/
        rsync -avz -e "ssh -i ${SSH_KEY}" \
            "${REMOTE_HOST}:${REMOTE_DIR}/logs/v3_*.log" \
            results/v3_remote/logs/
        echo "Results fetched to results/v3_remote/"
        ;;

    help|*)
        echo "Usage: bash scripts/dispatch_v3_experiments.sh {sync|launch|monitor|results}"
        echo ""
        echo "  sync    - Sync code to remote server"
        echo "  launch  - Launch all 4 waves on 4 GPUs"
        echo "  monitor - Check experiment status and GPU usage"
        echo "  results - Fetch results from remote server"
        ;;
esac
