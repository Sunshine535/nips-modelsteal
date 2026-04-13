#!/usr/bin/env bash
# Wave 3 auto-dispatch: queue experiments on GPUs as they become available
# All experiments use FIXED code (sparse gauge, FlatParamSpec, untie lm_head, ground_truth)

set -euo pipefail

PROJ="/mnt/ae22ef98-e02a-4dfa-acf4-a307e8941cd9/nwh/nips/nips-modelsteal"
cd "$PROJ"
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_ARGS="--model_name Qwen/Qwen2.5-0.5B --num_inits 1 --batch_size 8 --query_budget 500000 --pool_size 2048 --max_seq_len 128 --logit_suffix_positions 8 --allow_synthetic --patience 500"

# Define experiments as: "name|extra_args|output_dir"
declare -a QUEUE=(
  "alg_clean_s123|--max_steps 3000 --lm_head_steps 1500 --regime oracle --num_suffix_blocks 2 --init_method alg_clean --save_gramian_metrics --seed 123|results/exp1_algebraic_init/alg_clean/seed_123"
  "alg_clean_s777|--max_steps 3000 --lm_head_steps 1500 --regime oracle --num_suffix_blocks 2 --init_method alg_clean --save_gramian_metrics --seed 777|results/exp1_algebraic_init/alg_clean/seed_777"
  "alg_aug_s42|--max_steps 3000 --lm_head_steps 1500 --regime oracle --num_suffix_blocks 2 --init_method alg_aug --save_gramian_metrics --seed 42|results/exp1_algebraic_init/alg_aug/seed_42"
  "alg_aug_s123|--max_steps 3000 --lm_head_steps 1500 --regime oracle --num_suffix_blocks 2 --init_method alg_aug --save_gramian_metrics --seed 123|results/exp1_algebraic_init/alg_aug/seed_123"
  "alg_aug_s777|--max_steps 3000 --lm_head_steps 1500 --regime oracle --num_suffix_blocks 2 --init_method alg_aug --save_gramian_metrics --seed 777|results/exp1_algebraic_init/alg_aug/seed_777"
  "pure_logits_s777|--max_steps 3000 --lm_head_steps 1500 --regime pure_logits --num_suffix_blocks 3 --init_method random --seed 777|results/exp2_gramian/pure_logits/seed_777"
  "no_gauge_s42|--max_steps 3000 --lm_head_steps 1500 --regime oracle --num_suffix_blocks 2 --init_method alg_clean --no_gramian_project_gauge --save_gramian_metrics --seed 42|results/exp3_controls/no_gauge/seed_42"
  "beta0_s42|--max_steps 3000 --lm_head_steps 1500 --regime oracle --num_suffix_blocks 2 --init_method alg_clean --beta 0.0 --save_gramian_metrics --seed 42|results/exp3_controls/beta0_ablation/seed_42"
)

get_free_gpus() {
  # Return GPU indices with <1GB memory used
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '$2 < 1000 {print $1}'
}

QUEUE_IDX=0
TOTAL=${#QUEUE[@]}

echo "[$(date +%T)] Wave 3 dispatch starting. $TOTAL experiments queued."

while [ $QUEUE_IDX -lt $TOTAL ]; do
  FREE_GPUS=$(get_free_gpus)
  if [ -z "$FREE_GPUS" ]; then
    sleep 60
    continue
  fi

  for GPU in $FREE_GPUS; do
    if [ $QUEUE_IDX -ge $TOTAL ]; then break; fi

    IFS='|' read -r NAME EXTRA_ARGS OUTDIR <<< "${QUEUE[$QUEUE_IDX]}"
    LOGFILE="logs/wave3_${NAME}.log"

    # Clean up any old results
    rm -rf "$OUTDIR"

    echo "[$(date +%T)] Launching $NAME on GPU $GPU (${QUEUE_IDX}+1/$TOTAL)"

    nohup bash -c "
      cd $PROJ && source .venv/bin/activate
      export HF_ENDPOINT=https://hf-mirror.com
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      CUDA_VISIBLE_DEVICES=$GPU python -u scripts/run_spsi.py $BASE_ARGS $EXTRA_ARGS --output_dir $PROJ/$OUTDIR
    " > "$LOGFILE" 2>&1 &

    QUEUE_IDX=$((QUEUE_IDX + 1))
    sleep 5  # stagger launches
  done

  # Wait before checking again
  sleep 120
done

echo "[$(date +%T)] All $TOTAL experiments dispatched. Monitoring..."

# Wait for all background jobs
wait
echo "[$(date +%T)] Wave 3 complete."
