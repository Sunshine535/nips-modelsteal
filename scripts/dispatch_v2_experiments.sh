#!/usr/bin/env bash
# Dispatch all v2 experiments (post-bugfix) on 4× A100 GPUs
# Usage: bash scripts/dispatch_v2_experiments.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen2.5-0.5B"
POOL=4096
SEQ=192
BATCH=8
LM_STEPS=1500
MAX_STEPS=3000
PROBES=64
QSUB=256
LOGIT_K=8
DATE=$(date +%Y%m%d_%H%M)

LOG_DIR="logs/v2_${DATE}"
mkdir -p "$LOG_DIR"

COMMON_ARGS="--model_name $MODEL --pool_size $POOL --max_seq_len $SEQ --batch_size $BATCH \
  --lm_head_steps $LM_STEPS --max_steps $MAX_STEPS \
  --init_num_probes $PROBES --init_query_subsample $QSUB \
  --gramian_num_probes $PROBES --gramian_query_subsample $QSUB \
  --save_gramian_metrics --logit_suffix_positions $LOGIT_K \
  --num_suffix_blocks 2 --heldout_fraction 0.2"

echo "=== V2 Experiment Dispatch (post-bugfix) ==="
echo "Date: $DATE"
echo "Logs: $LOG_DIR/"

# ----------------------------------------------------------------
# Wave 1: Random baseline (3 seeds) + alg_clean seed 42
# GPU 0: random seed 42
# GPU 1: random seed 123 (init_offset=1)
# GPU 2: random seed 777 (init_offset=2)
# GPU 3: alg_clean seed 42
# ----------------------------------------------------------------
echo "[Wave 1] Launching random ×3 + alg_clean ×1..."

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/run_spsi.py \
  $COMMON_ARGS --regime oracle --num_inits 1 --init_offset 0 --seed 42 \
  --init_method random \
  --output_dir results/v2_random \
  > "$LOG_DIR/random_s42.log" 2>&1 &
PID_R0=$!

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/run_spsi.py \
  $COMMON_ARGS --regime oracle --num_inits 1 --init_offset 0 --seed 123 \
  --init_method random \
  --output_dir results/v2_random_s123 \
  > "$LOG_DIR/random_s123.log" 2>&1 &
PID_R1=$!

CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/run_spsi.py \
  $COMMON_ARGS --regime oracle --num_inits 1 --init_offset 0 --seed 777 \
  --init_method random \
  --output_dir results/v2_random_s777 \
  > "$LOG_DIR/random_s777.log" 2>&1 &
PID_R2=$!

CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/run_spsi.py \
  $COMMON_ARGS --regime oracle --num_inits 1 --init_offset 0 --seed 42 \
  --init_method alg_clean --gramian_project_gauge \
  --output_dir results/v2_alg_clean_s42 \
  > "$LOG_DIR/alg_clean_s42.log" 2>&1 &
PID_AC0=$!

echo "  GPU 0: random s42    (PID $PID_R0)"
echo "  GPU 1: random s123   (PID $PID_R1)"
echo "  GPU 2: random s777   (PID $PID_R2)"
echo "  GPU 3: alg_clean s42 (PID $PID_AC0)"

echo "[Wave 1] Waiting for completion..."
wait $PID_R0 $PID_R1 $PID_R2 $PID_AC0
echo "[Wave 1] Done."

# ----------------------------------------------------------------
# Wave 2: alg_clean seeds 123, 777 + alg_aug seed 42 + controls (beta=0)
# GPU 0: alg_clean seed 123
# GPU 1: alg_clean seed 777
# GPU 2: alg_aug seed 42
# GPU 3: beta=0 ablation seed 42
# ----------------------------------------------------------------
echo "[Wave 2] Launching alg_clean ×2 + alg_aug ×1 + beta0 ×1..."

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/run_spsi.py \
  $COMMON_ARGS --regime oracle --num_inits 1 --init_offset 0 --seed 123 \
  --init_method alg_clean --gramian_project_gauge \
  --output_dir results/v2_alg_clean_s123 \
  > "$LOG_DIR/alg_clean_s123.log" 2>&1 &
PID_AC1=$!

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/run_spsi.py \
  $COMMON_ARGS --regime oracle --num_inits 1 --init_offset 0 --seed 777 \
  --init_method alg_clean --gramian_project_gauge \
  --output_dir results/v2_alg_clean_s777 \
  > "$LOG_DIR/alg_clean_s777.log" 2>&1 &
PID_AC2=$!

CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/run_spsi.py \
  $COMMON_ARGS --regime oracle --num_inits 1 --init_offset 0 --seed 42 \
  --init_method alg_aug --gramian_project_gauge \
  --output_dir results/v2_alg_aug_s42 \
  > "$LOG_DIR/alg_aug_s42.log" 2>&1 &
PID_AA0=$!

CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/run_spsi.py \
  $COMMON_ARGS --regime oracle --num_inits 1 --init_offset 0 --seed 42 \
  --init_method alg_clean --gramian_project_gauge \
  --beta 0.0 \
  --output_dir results/v2_beta0_s42 \
  > "$LOG_DIR/beta0_s42.log" 2>&1 &
PID_B0=$!

echo "  GPU 0: alg_clean s123 (PID $PID_AC1)"
echo "  GPU 1: alg_clean s777 (PID $PID_AC2)"
echo "  GPU 2: alg_aug s42    (PID $PID_AA0)"
echo "  GPU 3: beta0 s42      (PID $PID_B0)"

echo "[Wave 2] Waiting for completion..."
wait $PID_AC1 $PID_AC2 $PID_AA0 $PID_B0
echo "[Wave 2] Done."

# ----------------------------------------------------------------
# Wave 3: Controls (wrong_teacher, no_gauge) + pure_logits ×2
# GPU 0: wrong_teacher seed 42
# GPU 1: no_gauge seed 42
# GPU 2: pure_logits seed 42
# GPU 3: pure_logits seed 123
# ----------------------------------------------------------------
echo "[Wave 3] Launching wrong_teacher + no_gauge + pure_logits ×2..."

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/run_spsi.py \
  $COMMON_ARGS --regime oracle --num_inits 1 --init_offset 0 --seed 42 \
  --init_method alg_clean --gramian_project_gauge \
  --wrong_teacher --wrong_teacher_seed 9999 \
  --output_dir results/v2_wrong_teacher_s42 \
  > "$LOG_DIR/wrong_teacher_s42.log" 2>&1 &
PID_WT=$!

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/run_spsi.py \
  $COMMON_ARGS --regime oracle --num_inits 1 --init_offset 0 --seed 42 \
  --init_method alg_clean --no_gramian_project_gauge \
  --output_dir results/v2_no_gauge_s42 \
  > "$LOG_DIR/no_gauge_s42.log" 2>&1 &
PID_NG=$!

CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/run_spsi.py \
  $COMMON_ARGS --regime pure_logits --num_inits 1 --init_offset 0 --seed 42 \
  --init_method random \
  --output_dir results/v2_pure_logits_s42 \
  > "$LOG_DIR/pure_logits_s42.log" 2>&1 &
PID_PL0=$!

CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/run_spsi.py \
  $COMMON_ARGS --regime pure_logits --num_inits 1 --init_offset 0 --seed 123 \
  --init_method random \
  --output_dir results/v2_pure_logits_s123 \
  > "$LOG_DIR/pure_logits_s123.log" 2>&1 &
PID_PL1=$!

echo "  GPU 0: wrong_teacher (PID $PID_WT)"
echo "  GPU 1: no_gauge      (PID $PID_NG)"
echo "  GPU 2: pure_logits s42  (PID $PID_PL0)"
echo "  GPU 3: pure_logits s123 (PID $PID_PL1)"

echo "[Wave 3] Waiting for completion..."
wait $PID_WT $PID_NG $PID_PL0 $PID_PL1
echo "[Wave 3] Done."

# ----------------------------------------------------------------
# Wave 4: pure_logits ×2 more seeds
# GPU 0: pure_logits seed 777
# GPU 1: pure_logits seed 999
# ----------------------------------------------------------------
echo "[Wave 4] Launching pure_logits ×2 more..."

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/run_spsi.py \
  $COMMON_ARGS --regime pure_logits --num_inits 1 --init_offset 0 --seed 777 \
  --init_method random \
  --output_dir results/v2_pure_logits_s777 \
  > "$LOG_DIR/pure_logits_s777.log" 2>&1 &
PID_PL2=$!

CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/run_spsi.py \
  $COMMON_ARGS --regime pure_logits --num_inits 1 --init_offset 0 --seed 999 \
  --init_method random \
  --output_dir results/v2_pure_logits_s999 \
  > "$LOG_DIR/pure_logits_s999.log" 2>&1 &
PID_PL3=$!

echo "  GPU 0: pure_logits s777 (PID $PID_PL2)"
echo "  GPU 1: pure_logits s999 (PID $PID_PL3)"

echo "[Wave 4] Waiting for completion..."
wait $PID_PL2 $PID_PL3
echo "[Wave 4] Done."

echo ""
echo "=== ALL V2 EXPERIMENTS COMPLETE ==="
echo "Results in: results/v2_*/"
echo "Logs in: $LOG_DIR/"
