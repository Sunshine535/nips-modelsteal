# Project: nips-modelsteal

## Project goal

Progressive Parameter Inversion: Recovering LLM Weights from Black-Box Access — 渐进式参数反演，从黑盒 logits 访问中恢复 LLM 权重，包括 KD baseline、多策略查询 PPI、recovery 评估与 defense 评估。

## Key models

- `Qwen/Qwen2.5-0.5B` — 教师/学生模型（同构，标准 transformer，黑盒 logits）
  - 24 blocks, d=896, d_ff=4864, 14 heads (2 KV heads GQA), RMSNorm, SiLU-gated MLP
  - vocab=151936, tied embeddings

## Key datasets

- WikiText（缓存预热用）
- 其他数据集由配置文件指定

## Repo map

- `scripts/` — 实验脚本
  - `run_all_experiments.sh` — 全阶段编排（Phase A/B/C）
  - `run_kd_baseline.py` — KD baseline
  - `run_progressive_inversion.py` — 渐进式参数反演
  - `eval_recovery_quality.py` — Recovery 质量评估
  - `run_defense_eval.py` — 防御评估
  - `distill_student.py` — 学生蒸馏
  - `invert_parameters.py` — 参数反演
  - `eval_extraction.py` — 抽取评估
  - `defense_evaluation.py` — 防御实验
  - `gpu_utils.sh` — GPU 分配工具
- `src/` — 核心模块
  - `parameter_inverter.py` — 参数反演器
  - `active_query.py` — 主动查询策略
- `configs/inversion_config.yaml` — 反演配置（含 teacher 模型路径）
- `results/` — 实验输出

## Common commands

```bash
bash setup.sh
source .venv/bin/activate

# 一键全流程（~570 GPU-hours）
bash run.sh

# 后台运行
nohup bash run.sh > run.log 2>&1 &

# 强制重跑
FORCE_RERUN=1 bash run.sh

# 打包结果
bash collect_results.sh
```

## Experiment phases

| Phase | 内容 |
|-------|------|
| A | KD baseline（全 GPU torchrun） |
| B | 并行: 3 种 inversion 策略 + 4 个 budget scaling + defense eval |
| C | Recovery 质量评估 |

## Data and outputs

- KD baseline: `results/kd_baseline/`
- 渐进式反演: `results/progressive_inversion/`
- Budget scaling: `results/scaling/budget_*/`
- 防御评估: `results/defense_eval/`
- Recovery 评估: `results/recovery_evaluation/`
- 日志: `logs/`

## Environment

- Python 3.10, PyTorch 2.6.0+cu124 (CUDA 13.0 driver)
- 关键依赖: transformers, datasets, accelerate, scipy, matplotlib, huggingface_hub
- 不使用 wandb
- 需设置: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 可选: flash-attn

## Project-specific rules

- Phase B 使用 `CUDA_VISIBLE_DEVICES=$GPU_IDX` 做并行实验
- Phase B 使用 `STAGGER_SECS` 错开启动避免 GPU 冲突
- `configs/inversion_config.yaml` 指定 teacher 模型，默认为 `Qwen/Qwen2.5-0.5B`
- 缓存预热在 Phase B 开始前执行

## Remote server

- SSH: `ssh -i ~/.ssh/id_ed25519_windows_remote kemove@100.103.219.5`
- GPU: 4x A100-80GB
- Activate: `source .venv/bin/activate`
- Code dir: `/mnt/ae22ef98-e02a-4dfa-acf4-a307e8941cd9/nwh/nips/nips-modelsteal`
- Background: `nohup bash -c '...' > logs/xxx.log 2>&1 &`
- HF mirror: `export HF_ENDPOINT=https://hf-mirror.com`
- No screen/tmux, no sudo
