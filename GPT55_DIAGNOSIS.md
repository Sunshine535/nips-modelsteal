下面是基于公开仓库快照、可读源码、论文草稿、结果 JSON、实验记录与审稿/自审记录得到的诊断报告。我没有运行大规模训练，也没有把缺失的私有日志当成已验证证据。当前最重要的结论是：**仓库里存在三个相互竞争且互相污染的主线：S-PSI 参数反演、Moment-CP 权重泄漏、Logit Completion 功能克隆；真正缺失的不是再调一个已有正面分支，而是一个“严格黑盒、计入查询预算、带不确定性门控的 logit-space completion 机制”。**

---

# 0. 仓库可读性判断

公开 GitHub 仓库可访问，根目录、`src/`、`scripts/`、`paper/`、`results/`、`research-wiki/`、`review-stage/`、`refine-logs/` 等目录可读；README 说明当前名义方法为 S-PSI，论文草稿当前题目却是 Moment-CP，实验进展记录又把 Logit Completion 作为最新强正面结果。根 README 给出 `run_spsi.py` 与 `run_kd_baseline.py` 快速命令，`scripts/` 下存在 S-PSI、KD、Moment、Carlini、logit-bias、memory、Jacobian、evaluation 等入口，`results/` 公开快照主要包含 v7–v14 的 KD / SCRD / Logit Completion JSON。([GitHub][1])

| Item               |    Found? | Location                                                                            | Notes                                                                                      |
| ------------------ | --------: | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 仓库是否可访问            |       Yes | root                                                                                | Public repo，可浏览文件树。([GitHub][1])                                                           |
| 完整代码               | Partially | `src/`, `scripts/`                                                                  | 关键源码可读；部分 raw 文件被压成单行但内容可读。([GitHub][2])                                                   |
| README             |       Yes | `README.md`                                                                         | 目前主叙事仍是 S-PSI。([GitHub][3])                                                                |
| 论文草稿               |       Yes | `paper/main.tex`                                                                    | 当前论文标题/摘要已转向 Moment-CP。([GitHub][4])                                                       |
| 训练脚本               |       Yes | `scripts/run_spsi.py`, `scripts/enhanced_kd_clone.py`, `scripts/run_kd_baseline.py` | S-PSI 与 KD/Logit Completion 都有入口。([GitHub][5])                                             |
| 评估脚本               |       Yes | `scripts/eval_*`, `functional_kl_eval.py`                                           | 有功能 KL、recovery quality、gauge invariant 等评估脚本。([GitHub][5])                                |
| configs            |       Yes | `configs/inversion_config.yaml`                                                     | 主要是 S-PSI 配置，默认 Qwen3.5-0.8B，与部分实验 Qwen2.5-0.5B 不一致。([GitHub][6])                          |
| 日志和结果              | Partially | `results/v7_*`–`v14_*`, docs                                                        | v7/v8/v9/v13/v14 raw JSON 可读；许多论文引用路径未出现在公开 results。([GitHub][7])                          |
| baseline           |       Yes | `matched_kd_baseline.py`, `run_kd_baseline.py`, KD variants                         | 但 official baseline / Clone 2025 reproduction 尚未充分证明。([GitHub][5])                         |
| 失败实验记录             |       Yes | `findings.md`, `EXPERIMENT_PROGRESS.md`, `ATTACK_4WAY_SUMMARY.md`, review docs      | 失败记录很有价值，尤其是 active-query bug、hidden MSE failure、S-PSI negative。([GitHub][8])              |
| ablation           |       Yes | results JSON + claims docs                                                          | 有 beta=0、K expansion、warmstart、wrong teacher、hidden-MSE/KD variants，但原始日志不完整。([GitHub][9]) |
| requirements / env |       Yes | `requirements.txt`, `configs/*`                                                     | requirements 简单，PyTorch 由 setup 单独安装。([GitHub][10])                                        |

| Missing Item                                                      | Why Needed                                                     | What You Should Upload                                                         |
| ----------------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `results/v5_attack_moments_v2/results.json` 与 CP factor artifacts | 当前论文主 claim 依赖 Moment-CP W_lm top-5 0.813，但公开 results 未看到对应目录。 | 完整 `results/v5_attack_moments_v2/`、`cp_real_factors.pt`、运行日志、命令、seed、config。   |
| 所有 S-PSI v1–v6 / warmstart / Gramian / functional KL 原始结果         | `CLAIMS_FROM_RESULTS.md` 与 paper 多处引用这些实验。                     | `results/v1*`–`v6*`, `warmstart_sweep`, `functional_eval_v2`, `gramian_*` 全目录。 |
| checkpoint manifest                                               | 判断 stale checkpoint / wrong config resume 是否污染结果。              | 每次实验的 checkpoints、resolved args、git hash、命令、环境。                                |
| paper figures/tables generation scripts                           | 判断论文表格是否直接来自日志。                                                | 生成 `paper/figures` 与所有表格的 scripts/notebooks。                                   |
| official / external baseline reproduction logs                    | 避免弱 baseline。                                                  | Clone 2025、Carlini、Finlayson、top-k KD 等 official 或 faithful reproduction logs。 |
| 多 seed Logit Completion 结果                                        | 当前 v13/v14 是 seed=42 单 seed，不能支撑稳定 claim。                      | seeds 0/1/2/3/4 的 v13/v14 exact config results。                                |

---

# 1. Repository Map

| Component                  | Path                                                                                        |                                                                     Purpose | Importance            | Notes                                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------: | --------------------- | ---------------------------------------------------------------------------------------------------- |
| README / project statement | `README.md`                                                                                 |                  声称 S-PSI：black-box logit access 下恢复 transformer suffix 参数。 | High                  | README 与当前 paper/logit-completion 叙事不一致。([GitHub][3])                                                |
| S-PSI core                 | `src/parameter_inverter.py`                                                                 | Teacher cache、sensitivity loss、oracle boundary、progressive block inversion。 | High                  | 当前 S-PSI 主线的实现核心。([GitHub][11])                                                                      |
| Active query               | `src/active_query.py`, `scripts/run_active_query.py`                                        |                                               选择 queries / active recovery。 | Medium                | `findings.md` 已确认一个 active-query 实验 bug，不能作为正证据。([GitHub][12])                                       |
| Algebraic init / Gramian   | `src/algebraic_init.py`, `src/gramian.py`                                                   |                                              局部线性化、observability、warmstart。 | High for diagnosis    | 证明“可观测但不可恢复”很关键，但多处结果缺原始日志。([GitHub][9])                                                             |
| Gauge / symmetry           | `src/symmetry_gauge.py`, `src/permutation_alignment.py`                                     |                                      RMSNorm/MLP/attention gauge、对齐 cosine。 | High                  | 论文 claim 与实现覆盖范围需要核对；attention gauge 曾被审稿记录指出为遗漏/不完整。([GitHub][13])                                  |
| S-PSI launcher             | `scripts/run_spsi.py`                                                                       |                                 训练入口、query pool、randomize suffix、aggregate。 | High                  | 纯 logits regime 实际保留 pretrained embedding/prefix，不等价于 docstring“random frozen prefix”。([GitHub][14]) |
| Enhanced KD / SCRD / LC    | `scripts/enhanced_kd_clone.py`                                                              |              A/B/C/D/E variants：top-K KD、hidden-state MSE、Logit Completion。 | Very High             | v13/v14 正面结果来自这里；但 completion 使用 full teacher logits 模拟 probe，需要严格黑盒重写。([GitHub][15])                |
| Carlini reproduction       | `scripts/reproduce_carlini.py`                                                              |                                     SVD recover output projection subspace。 | High                  | 是所有 logit-space / W_lm subspace 工作的强 baseline。([GitHub][16])                                         |
| Moment attack              | `scripts/attack_higher_order_moments.py`                                                    |                                       CP decomposition cross-moment attack。 | High but inconsistent | 文件头标注“quarantined / killed branches / not cited”，但 paper 当前主打 Moment-CP。([GitHub][17])               |
| Functional KL eval         | `scripts/functional_kl_eval.py`                                                             |                       Carlini+CP / random / oracle lm_head functional test。 | High                  | 比 parameter cosine 更接近真实功能 theft，但结果目录缺失。([GitHub][18])                                              |
| Public results             | `results/v7_*`–`v14_*`                                                                      |                                              SCRD/KD/logit completion JSON。 | Very High             | 有 raw JSON；大多单 seed。([GitHub][19])                                                                   |
| Experiment notes           | `EXPERIMENT_PROGRESS.md`, `CLAIMS_FROM_RESULTS.md`, `ATTACK_4WAY_SUMMARY.md`, `findings.md` |                                                              正负结果、bug、叙事变迁。 | Very High             | 这是现象库主来源。([GitHub][8])                                                                               |
| Paper draft                | `paper/main.tex`                                                                            |                                 当前论文 claim：Moment CP recovers W_lm columns。 | Very High             | 与 README/S-PSI 和 enhanced KD path 不一致。([GitHub][20])                                                 |
| Review / audit             | `review-stage/*`, `PAPER_CLAIM_AUDIT*.md`                                                   |                                                           自动审稿、claim audit。 | Very High             | 已多次标记 paper claim FAIL / 数字错配 / evidence gap。([GitHub][21])                                          |
| configs/env                | `configs/*.yaml`, `requirements.txt`                                                        |                                                                S-PSI 配置与依赖。 | Medium                | 配置与实验 model_name 不统一。([GitHub][10])                                                                  |
| tests                      | `tests/test_modelsteal.py`                                                                  |                                                                       基础测试。 | Medium                | 不足以覆盖数据、metric、black-box oracle。([GitHub][22])                                                       |

**当前仓库试图解决的问题**：从 black-box LLM API 输出中恢复模型信息，早期目标是 suffix 参数反演，后期变成 output projection / moment leakage / top-K API functional cloning。
**当前已有方法**：S-PSI、Gramian/algebraic init、active query、Moment-CP、Carlini SVD reproduction、SCRD hidden-state KD、Logit Completion。
**核心假设演化**：从“局部 sensitivity 可以反演 suffix 参数”转向“higher-order moments / low-rank logits 可以破 gauge”，再转向“top-K + logit-bias 可以完成 dense logits 并提升 KD”。
**当前 claim 的 prior limitation**：Carlini 只恢复输出投影子空间；仓库希望进一步恢复列/内部权重或在 top-K 限制下克隆功能。
**主要训练入口**：`scripts/run_spsi.py` 和 `scripts/enhanced_kd_clone.py`。
**主要评估入口**：`scripts/functional_kl_eval.py`, `scripts/eval_*`, results JSON。
**数据处理**：`build_query_pool`, `WikiTextDataset`，主要用 WikiText-103 train/validation；有 synthetic fallback。
**模型核心**：HuggingFace causal LM，Qwen2.5-0.5B / Qwen3.5-0.8B 等。
**loss/objective**：S-PSI 用 logit MSE + sensitivity MSE + L2；KD path 用 KL/top-K KD + CE + optional hidden MSE / completion。
**baseline**：KD variants、matched KD、Carlini SVD、random/oracle lm_head。
**dead / historical code risk**：Moment script 标注 quarantined；active-query 结果确认 bug；S-PSI 与 paper 主线冲突。
**会影响实验结论的文件**：`enhanced_kd_clone.py`, `parameter_inverter.py`, `run_spsi.py`, `functional_kl_eval.py`, `attack_higher_order_moments.py`, result aggregation scripts, paper tables。

---

# 2. Result Reliability Audit

| Result ID | Result Name                                     | Dataset          | Metric                          |                                  Claimed Value |                                       Logged Value | Config       |            Seed | Command                 | Checkpoint | Status                | Reliability                                 | Issue                                                                                                                                      |
| --------- | ----------------------------------------------- | ---------------- | ------------------------------- | ---------------------------------------------: | -------------------------------------------------: | ------------ | --------------: | ----------------------- | ---------- | --------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| R1        | v13 Logit Completion topK20                     | WikiText eval    | PPL/KL/top1                     |                         PPL 137.26 vs A 178.09 | raw JSON: E PPL 137.2579, KL 1.9310; A PPL 178.086 | args in JSON |              42 | implicit script args    | none       | Possibly Contaminated | medium for signal, low for strict black-box | `complete_logits` receives full teacher logits and recovers probe-token logits from them; logit-bias query cost not counted.([GitHub][23]) |
| R2        | v14 Logit Completion topK5                      | WikiText eval    | PPL/KL/top1                     |                         PPL 136.41 vs A 190.04 |                raw JSON: E PPL 136.4105; A 190.041 | args in JSON |              42 | implicit script args    | none       | Possibly Contaminated | medium/low                                  | Same strict-black-box issue; single seed.([GitHub][24])                                                                                    |
| R3        | v8 pretrained full-logit KD                     | WikiText eval    | PPL/KL                          |                      A/D/B2 all around 134–142 |            raw JSON: A 135.40, D 134.75, B2 135.89 | args in JSON |              42 | implicit                | none       | Partially Verified    | medium                                      | Shows hidden MSE not useful under pretrained+noise; full-logit setting not black-box top-K.([GitHub][25])                                  |
| R4        | v7 random 5k fails                              | WikiText eval    | PPL/KL                          |                                     PPL ≈ 2800 |                     raw JSON: A 2833.82, D 2861.68 | args in JSON |              42 | implicit                | none       | Verified              | medium                                      | Good negative: random-init KD does not clone with small budget.([GitHub][7])                                                               |
| R5        | v9 random 50k still weak                        | WikiText eval    | PPL/KL                          |                             still insufficient |                     raw JSON: A 2035.32, D 1985.41 | args in JSON |              42 | implicit                | none       | Verified              | medium                                      | Strong negative against “just more steps from random”.([GitHub][26])                                                                       |
| R6        | Hidden MSE catastrophic under topK20/5          | WikiText eval    | PPL                             |                                  B ≈ 753 / 736 |            raw v13/v14: B_h_oracle 753.15 / 736.24 | args in JSON |              42 | implicit                | none       | Verified              | medium                                      | Reliable as same-script comparison; mechanism interpretation high value.([GitHub][23])                                                     |
| R7        | S-PSI Gramian full rank / well-conditioned      | Qwen suffix      | rank/cond                       |                            8 configs full rank |                                          docs only | partial      |           mixed | missing                 | missing    | Missing Log           | low–medium                                  | `CLAIMS_FROM_RESULTS.md` claims; raw artifacts mostly absent from public results.([GitHub][9])                                             |
| R8        | K expansion 32→128 no recovery                  | Qwen             | cosine                          |                                 no improvement |                                          docs only | partial      |         unclear | missing                 | missing    | Missing Log           | low–medium                                  | Useful negative but not fully auditable.([GitHub][9])                                                                                      |
| R9        | Warmstart negligible                            | Qwen             | Δ cosine                        |                                  |Δcos| < 0.01 |                                      docs + audits | partial      |    single/mixed | missing                 | missing    | Partially Verified    | low–medium                                  | Paper audits show numeric mismatches in earlier versions.([GitHub][9])                                                                     |
| R10       | beta=0 same as sensitivity                      | Qwen             | cosine                          |                             beta=0 ≈ alg clean |                                          docs only | partial      |         unclear | missing                 | missing    | Missing Log           | low–medium                                  | Important because it refutes sensitivity objective, but raw logs needed.([GitHub][9])                                                      |
| R11       | Active query cos≈0.999                          | active-query exp | cosine                          |                                strong positive |                                        invalidated | unknown      |         unknown | missing                 | missing    | Contradicted          | unusable                                    | `findings.md` confirms student loaded same pretrained weights as teacher.([GitHub][12])                                                    |
| R12       | KD suffix baseline valid negative               | Qwen             | cosine                          |              block≈0.13, lm_head aligned≈0.063 |                                          docs only | partial      | 3 seeds claimed | missing                 | missing    | Partially Verified    | medium-low                                  | Good negative but raw per-seed logs absent.([GitHub][12])                                                                                  |
| R13       | Moment-CP W_lm top5 0.813                       | Qwen             | top-k cosine/null margin        |                W_lm.cols 0.8128 vs null 0.1626 |                                    docs/paper only | missing      | seed 42 claimed | missing public artifact | missing    | Missing Log           | medium-low                                  | Main paper claim, but `results/v5_attack_moments_v2` not visible; script says quarantined.([GitHub][27])                                   |
| R14       | Moment-CP weak internal signals                 | Qwen             | cosine over W_v/W_up/W_q/W_gate |                                 weak positives |                                    docs/paper only | missing      |              42 | missing                 | missing    | Missing Log           | low–medium                                  | Useful as signal, not proof.([GitHub][27])                                                                                                 |
| R15       | Jacobian FD negative                            | Qwen             | subspace/cosine                 | sees common architecture, not teacher-specific |                                          docs only | missing      |         unclear | missing                 | missing    | Missing Log           | medium-low                                  | Valuable negative mechanism clue.([GitHub][27])                                                                                            |
| R16       | Logit-bias precision fail but Carlini SVD works | Qwen             | h_L cos/subspace                |              h_L fail 0.0015; Phase5 h_L 0.957 |                                          docs only | missing      |         unclear | missing                 | missing    | Unclear               | low–medium                                  | Important for oracle design; implementation break not fully auditable.([GitHub][27])                                                       |
| R17       | Current paper Moment-CP thesis                  | Qwen             | paper table                     |                                W_lm top5 0.813 |                                              paper | missing raw  |              42 | missing                 | missing    | Partially Verified    | low–medium                                  | Paper and script quarantine conflict.([GitHub][20])                                                                                        |

**结论**：v7/v8/v9/v13/v14 JSON 里的功能克隆结果可作为“机制线索”，但 v13/v14 不能直接支撑“strict top-K black-box” claim。Moment-CP 可作为“可能的 gauge-breaking signal”，但在公开快照中缺少原始 artifacts，且核心 script 标注 quarantined，不能作为主证明。Active-query 正面结果必须从证据库中删除。

---

# 3. 代码正确性审查：Suspected Bug Table

| Priority | File                                                                   | Function/Class                       | Code Region                         | Suspicion                                                                                                                   | Evidence                                                                                                                                                                                   | How to Verify                                                                                | Proposed Fix for Claude Code                                                                                  | Expected Effect                  | Confidence |
| -------: | ---------------------------------------------------------------------- | ------------------------------------ | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------- | ---------- |
|       P0 | `scripts/enhanced_kd_clone.py`                                         | `complete_logits`, `train_variant`   | Logit Completion path               | **Top-K/Logit Completion 实验使用 full teacher logits 来恢复 probe token logits**，不是严格 top-K API；如果 claim 是 top-K black-box，则结果污染。 | `train_variant` 调 `teacher(...).logits` 得到 full logits，再把 full `t_logits` 传给 `complete_logits`; `complete_logits` 从 `probe_ids` 取 logits 做 lstsq，并只把 top-K exact scatter 回去。([GitHub][15]) | 加 strict oracle，只返回 top-K；让 `complete_logits` 只能访问已查询的 probe logits；若无 logit-bias probe，应失败。 | 新增 `src/oracles.py`, `src/logit_completion.py`；禁止 completion 直接读取 full `t_logits`；计入 logit-bias query budget。 | v13/v14 可能下降；但结果变成学术可用。          | high       |
|       P0 | `scripts/run_active_query.py`                                          | active-query init                    | reported lines 339/364/411 in notes | Active query 结果确认污染：student 与 teacher 同权重。                                                                                  | `findings.md` 明确“BUG CONFIRMED”，所有 conditions cos≈0.999 不能用。([GitHub][12])                                                                                                                 | 重跑 init sanity：训练前 student/teacher suffix cosine 应接近 random/null，而非 0.999。                   | archive active-query old results；新增 pre-run `assert_initial_cosine_below_threshold`。                          | 删除 false positive；保护后续实验。        | high       |
|       P0 | `scripts/attack_higher_order_moments.py` vs `paper/main.tex`           | repo state                           | file header / paper title           | 当前 paper 主打 Moment-CP，但脚本头写 quarantined / killed / not cited。                                                               | Paper title/abstract claim Moment-CP；script header 否定其作为新 claim。([GitHub][20])                                                                                                             | grep all paper-cited commands and scripts; check artifact provenance.                        | 将 Moment-CP 降级为 `historical_signal`，直到 raw artifact + non-quarantined reproduction script pass。               | 避免论文主线建立在不一致代码上。                 | high       |
|       P1 | `src/parameter_inverter.py`                                            | `TeacherCache.get_batch`             | perturbed fallback                  | 如果 perturb index 不合法，代码静默用 clean repeat 替代，可能让 sensitivity loss 假装存在。                                                       | local inspection: invalid pert_indices 时 `pi=ids.repeat`, `pl=cl.repeat`。                                                                                                                  | 设置 P/R/indices 边界测试，确认非法时 raise。                                                             | fallback 改成 `raise RuntimeError`；若 `P*R==0` 显式走 no-perturb path。                                              | 防止 beta ablation 被静默污染。          | high       |
|       P1 | `src/parameter_inverter.py`, `src/algebraic_init.py`, `src/gramian.py` | autocast                             | training/diagnostics                | 多处 `torch.autocast("cuda")` 无条件使用，CPU smoke test 可能失败。                                                                      | parameter inverter/algebraic/gramian 都有硬编码 CUDA autocast。                                                                                                                                  | `CUDA_VISIBLE_DEVICES="" pytest`。                                                            | 用 `device.type` 或 helper `autocast_for(device)`。                                                              | smoke tests 可在 CPU/tiny model 跑。 | high       |
|       P1 | `src/parameter_inverter.py`                                            | `invert_block`                       | train/eval mode                     | 反演过程中 `student.train()`，但目标是 deterministic logit matching；dropout/随机层会污染。                                                   | local code step 505。                                                                                                                                                                       | 同 batch eval/train 输出 diff test。                                                             | 使用 `student.eval()` + grad enabled；仅优化 selected params。                                                       | 降低随机性，提升 reproducibility。        | medium     |
|       P1 | `src/parameter_inverter.py`                                            | `invert_block`                       | regularization                      | `loss_reg=sum(norm^2)` 未按参数量 normalize；gamma 可能压倒 logit/sensitivity。                                                        | local code line 548–553；日志未记录 reg term。                                                                                                                                                    | log loss_logit/sens/reg magnitudes。                                                          | 改为 per-parameter mean norm或 AdamW；记录 reg。                                                                     | 解释部分 optimization barrier。       | medium     |
|       P1 | `scripts/run_spsi.py`                                                  | `randomize_suffix`, `_untie_lm_head` | pure_logits regime                  | docstring 说 pure logits student prefix random frozen；实际 embedding 保留 pretrained，suffix/lm_head randomize。                   | `parameter_inverter.py` docstring 与 `run_spsi.py` `_untie_lm_head` 注释冲突。([GitHub][11])                                                                                                     | 输出 trainable/randomized param manifest。                                                      | 改 README/claim，或新增真正 random-prefix config；所有结果标注 regime。                                                      | 避免 threat model 夸大。              | high       |
|       P1 | `src/parameter_inverter.py`                                            | `_save_results`                      | result manifest                     | summary 不保存 command、git hash、resolved config、data split、checkpoint path、loss terms、query accounting。                        | `_save_results` 只存 regime/seed/blocks/model。                                                                                                                                               | inspect result JSON。                                                                         | 新增 `manifest.json` schema。                                                                                    | 提升可复现性，支持 audit。                 | high       |
|       P2 | `src/parameter_inverter.py`                                            | checkpoint resume                    | `_load_dir`                         | 可能从旧 config/model checkpoint 静默 resume；无 config hash guard。                                                                 | loads latest `ckpt_{block}_step*.pt` only by name.                                                                                                                                         | 尝试不同 config resume，确认未阻止。                                                                    | checkpoint 存 model/config/data hash；resume mismatch hard fail。                                                | 避免 stale checkpoint 污染。          | medium     |
|       P2 | `scripts/enhanced_kd_clone.py`                                         | evaluation                           | `evaluate`                          | teacher baseline uses teacher vs teacher KL=0，student CE 用 ground-truth next token；OK，但未记录 train/eval split hashes。         | raw JSON args 无 split hash/command.                                                                                                                                                        | manifest check。                                                                              | 保存 dataset name/split/hash/max_samples/eval_batches。                                                          | 防止 train/test/split 混淆。          | medium     |
|       P2 | `src/algebraic_init.py` / `run_spsi`                                   | alg metrics                          | alg init                            | `alg_result` 只 log，不进入 `BlockResult.init_metrics`；后续 claim 难验证。                                                             | run_spsi logs alg_result but br 不保存。                                                                                                                                                       | inspect summary JSON.                                                                        | 把 predicted/post/init singular values写入 result。                                                               | 让 algebraic ablation 可审计。        | high       |

---

# 4. Claim-Code-Result Matrix

| Claim                                                  | Source File                                        | Implementation File                                       | Result Evidence                                                       | Status              | Problem                                                                                         | Confidence |
| ------------------------------------------------------ | -------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------- | ------------------- | ----------------------------------------------------------------------------------------------- | ---------- |
| S-PSI recovers suffix parameters from black-box logits | `README.md`, `src/parameter_inverter.py` docstring | `parameter_inverter.py`, `run_spsi.py`                    | `CLAIMS_FROM_RESULTS.md` mostly negative; KD suffix baseline negative | Unsupported         | 公开证据显示 parameter recovery weak；核心 sensitivity beta 无明显贡献。([GitHub][3])                          | medium     |
| Sensitivity matching is core useful mechanism          | README/method                                      | `loss_sensitivity`                                        | beta=0 control ≈ same; v7/v9 negative                                 | Contradicted        | sensitivity objective未提供稳定增益。([GitHub][9])                                                      | medium     |
| Observability full-rank but recoverability fails       | `CLAIMS_FROM_RESULTS.md`, older narrative          | `gramian.py`, `algebraic_init.py`                         | docs claim K expansion/warmstart fail                                 | Partially Supported | raw logs missing，但负面现象一致。([GitHub][9])                                                          | medium     |
| Active query gives near-perfect recovery               | old active-query results                           | `run_active_query.py`                                     | `findings.md` bug confirmed                                           | Contradicted        | must delete from claim set。([GitHub][12])                                                       | high       |
| Moment-CP recovers W_lm columns beyond Carlini gauge   | `paper/main.tex`, `ATTACK_4WAY_SUMMARY.md`         | `attack_higher_order_moments.py`, `functional_kl_eval.py` | docs/paper claim W_lm top5 0.813                                      | Partially Supported | raw artifact missing; script says quarantined / not cited。([GitHub][20])                        | low–medium |
| Moment-CP gives internal layer leakage map             | paper                                              | moment script                                             | docs claim weak W_v/W_up/W_q/W_gate                                   | Partially Supported | missing raw, weak signal; should be secondary only。([GitHub][27])                               | low        |
| Logit Completion improves top-K KD                     | `EXPERIMENT_PROGRESS.md`, `enhanced_kd_clone.py`   | `complete_logits`                                         | v13/v14 raw JSON                                                      | Partially Supported | algorithmic signal real in simulator, but strict black-box access not implemented。([GitHub][8]) | medium     |
| Hidden-state MSE helps clone                           | SCRD variants                                      | `enhanced_kd_clone.py`                                    | v13/v14 B/C/D worse                                                   | Contradicted        | representation objective conflicts with behavior objective。([GitHub][23])                       | high       |
| Current paper claims are audit-clean                   | paper + review docs                                | N/A                                                       | PAPER_CLAIM_AUDIT R1/R2 FAIL                                          | Contradicted        | number mismatches/evidence gaps persist。([GitHub][21])                                          | high       |
| Strict SOTA / apples-to-apples                         | paper/notes                                        | missing official baselines                                | EXPERIMENT_PROGRESS says Clone 2025 pending                           | Unsupported         | Cannot claim SOTA.([GitHub][8])                                                                 | high       |

---

# 5. Phenomenon Ledger

| ID  | Observation                                                          | Type                       | Where Found        | Setting                           | Metric                  | Compared To      | Reliability | What It Suggests                                                                 | What It Rules Out                              | Confidence |
| --- | -------------------------------------------------------------------- | -------------------------- | ------------------ | --------------------------------- | ----------------------- | ---------------- | ----------- | -------------------------------------------------------------------------------- | ---------------------------------------------- | ---------- |
| P1  | Pretrained+noise logit KD already strong: v8 A PPL 135.4             | Positive                   | v8 JSON            | full logits, pretrained perturbed | PPL/KL                  | teacher PPL 24.2 | medium      | functional cloning depends heavily on initialization and dense logit supervision | random-init recovery as main route             | high       |
| P2  | Logit Completion improves topK20/5 over sparse top-K KD              | Positive but contaminated  | v13/v14 JSON       | topK simulated                    | PPL                     | A baseline       | medium/low  | dense tail supervision is useful                                                 | pure top-K sparse KD as final method           | medium     |
| P3  | Hidden-state oracle/recovered MSE hurts under top-K                  | Negative                   | v13/v14            | topK5/20                          | PPL                     | A/E              | medium      | representation matching conflicts with behavior under gauge/RMSNorm              | hidden MSE/SCRD as main path                   | high       |
| P4  | v7 random 5k fails badly                                             | Negative                   | v7 JSON            | random init                       | PPL ~2800               | teacher          | medium      | from-scratch distillation budget too small                                       | claim that KD alone recovers model             | high       |
| P5  | v9 random 50k still bad                                              | Negative                   | v9 JSON            | random init 50k                   | PPL ~2000               | teacher          | medium      | not just insufficient steps                                                      | simple scaling of random KD                    | high       |
| P6  | K expansion / full-rank Gramian doesn’t recover parameters           | Negative/mixed             | claims docs        | S-PSI                             | cosine                  | K=32 vs 128      | low–medium  | observation volume not bottleneck; objective/geometry is                         | “more probes fixes recovery”                   | medium     |
| P7  | sensitivity beta=0 no worse                                          | Negative                   | claims docs        | S-PSI                             | cosine                  | beta>0           | low–medium  | local perturbation sensitivity is not the missing signal                         | sensitivity-guided as core novelty             | medium     |
| P8  | algebraic init marginal over random                                  | Mixed                      | claims docs        | Qwen                              | cosine                  | random           | low–medium  | linearized Gramian has weak useful direction but not enough                      | purely algebraic single-step recovery          | medium     |
| P9  | Moment-CP W_lm signal > null                                         | Positive but under-audited | ATTACK_4WAY/paper  | Qwen                              | top5 cosine             | null             | low–medium  | higher-order moments may break output-projection gauge                           | CP as fully verified main result               | medium-low |
| P10 | Moment internal signals weak/sparse                                  | Mixed                      | ATTACK_4WAY        | Qwen                              | cosine                  | null             | low         | depth/RMSNorm screen internal weights; need gating                               | full internal recovery claim                   | medium     |
| P11 | Memory probing expands rank but no recovery                          | Negative                   | ATTACK_4WAY        | rich queries                      | effective rank          | baseline queries | low–medium  | rank alone insufficient; need structure/targeted objective                       | “observability=attack success”                 | medium     |
| P12 | Jacobian FD sees architecture subspace, not teacher-specific         | Negative                   | ATTACK_4WAY        | FD queries                        | subspace                | student null     | low–medium  | first-order signal not identifiable enough                                       | Jacobian-only attack                           | medium     |
| P13 | Logit-bias precision implementation broke; Carlini SVD control works | Anomalous                  | ATTACK_4WAY        | logit-bias/Carlini                | h cos                   | oracle           | low–medium  | API simulator/precision must be hardened                                         | relying on broken binary search implementation | medium     |
| P14 | Active-query positive invalid                                        | Anomalous/bug              | findings           | active query                      | cos≈0.999               | random           | high        | init sanity checks are mandatory                                                 | any active-query claim from old run            | high       |
| P15 | Paper claim audits repeatedly fail                                   | Anomalous                  | review-stage docs  | paper                             | number/evidence matches | logs             | high        | result provenance and narrative unstable                                         | submit-ready claim set                         | high       |
| P16 | Moment script quarantined but paper uses Moment thesis               | Anomalous                  | script vs paper    | repo state                        | N/A                     | N/A              | high        | repository governance broken; main path must separate historical from active     | treating paper as source of truth              | high       |
| P17 | Top-K logit completion uses full logits internally                   | Anomalous/code             | enhanced KD script | topK                              | API access              | black-box        | high        | strict oracle abstraction missing                                                | top-K black-box claim from v13/v14             | high       |
| P18 | Llama/S-PSI values conflict across docs                              | Unstable                   | findings vs paper  | Llama                             | cosine/KL               | Qwen             | low         | cross-model generalization not settled                                           | broad architecture claim                       | medium     |

---

# 6. Design Constraints

| Constraint ID | Derived From Observation | Constraint Type    | Meaning                                                             | Implication for New Method                                                      | Confidence |
| ------------- | ------------------------ | ------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ---------- |
| C1            | P2/P17                   | Must Fix           | Completion must be strict black-box and query-accounted.            | Build oracle that exposes only top-K plus optional counted logit-bias probes.   | high       |
| C2            | P2/P1                    | Must Preserve      | Dense logit-space supervision is the strongest functional signal.   | Keep logit completion principle, but not its current leakage.                   | high       |
| C3            | P3                       | Must Avoid         | Hidden-state MSE conflicts with functional objective/gauge.         | No direct h-MSE in main method; hidden estimates only for logit reconstruction. | high       |
| C4            | P6/P7/P11                | Must Explain       | More observations/rank/sensitivity do not imply parameter recovery. | Method must target behavior-aligned logit objective, not raw parameter cosine.  | high       |
| C5            | P9/P10                   | Must Stabilize     | Moment signals are sparse/noisy and need null-margin gating.        | Use Moment-CP only as confidence-weighted gauge/completion prior.               | medium     |
| C6            | P14/P15/P16              | Must Control       | Provenance bugs create false positives.                             | Add manifest, init sanity, strict result schema, no silent archive.             | high       |
| C7            | P4/P5                    | Must Avoid         | Random-from-scratch KD is not viable under current budget.          | Use pretrained/noise or explicitly compare as weak baseline, not main.          | high       |
| C8            | P13/P17                  | Must Test          | logit-bias simulator itself is a core dependency.                   | Add unit tests for probe recovery accuracy and counted budget.                  | high       |
| C9            | P8/P9                    | Must Differentiate | Existing CP/Carlini signals are not enough for novelty.             | New contribution must be calibrated completion + uncertainty/gauge control.     | medium     |
| C10           | P18                      | Must Generalize    | Qwen-only evidence insufficient.                                    | Minimal second-model gate before broad claim.                                   | medium     |
| C11           | R13/R17                  | Must Not Claim     | Cannot claim full internal parameter theft now.                     | Reframe paper as query-budgeted functional extraction under top-K+probe access. | high       |
| C12           | v13/v14 single seed      | Must Stabilize     | Positive signal lacks multi-seed stability.                         | 3–5 seed gate before paper claim.                                               | high       |
| C13           | P15                      | Must Control       | Paper numbers must be generated from checked artifacts.             | One script must regenerate all tables.                                          | high       |

---

# 7. Negative-to-Insight Analysis

| Negative Observation                           | Failed Assumption                               | Why the Assumption Failed                                                                                            | What Mechanism Is Missing                          | New Design Requirement                                                      |
| ---------------------------------------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------- |
| Hidden MSE worsens PPL under top-K             | Matching representation helps behavior          | h-space has gauge/RMSNorm scale ambiguity; behavior lives in logits, not raw hidden coordinates.                     | Gauge-invariant functional alignment               | Use h only to reconstruct logits; no direct h-MSE.                          |
| beta sensitivity no effect                     | Local perturbation response identifies suffix   | Perturbation deltas are first-order and entangled with frozen/pretrained prefix; signal not teacher-specific enough. | Higher-order / structural signal with null control | Use moment/Carlini signal only where above null margin.                     |
| K expansion no recovery                        | More observed outputs solve recovery            | Non-convex/gauge barrier dominates; objective mismatched.                                                            | Objective alignment and calibration                | Optimize calibrated dense logit KL, not cosine.                             |
| Random KD fails even 50k                       | More steps can replace initialization           | Student lacks pretrained manifold; distillation signal too sparse/noisy.                                             | Manifold-preserving initialization                 | Main method uses pretrained-perturbed or fair pretrained clone baseline.    |
| Moment internal signals weak                   | CP decomposition recovers transformer internals | Deep residual/RMSNorm/attention fold pathways; internal weights not separable by simple moment.                      | Confidence gating and output-focused use           | Do not claim internal recovery; use W_lm/gauge prior only.                  |
| Memory probing rank increase no recovery       | Effective rank equals useful identifiability    | Rank lacks alignment with recoverable teacher parameters.                                                            | Task-aware target                                  | Query selection must optimize completion calibration, not rank alone.       |
| Jacobian FD sees common subspace               | First-order Jacobian distinguishes teacher      | Architecture subspace dominates; teacher-specific info weak.                                                         | Teacher-specific higher-order/statistical signal   | Moment/low-rank calibration with null baselines.                            |
| Active query bug                               | Positive cosine validates method                | Student initialized as teacher; no sanity check.                                                                     | Pre-run identity/null controls                     | Hard fail if initial similarity is too high.                                |
| Logit Completion top-K result uses full logits | Simulated top-K equals API top-K+logit_bias     | Probe logits were taken from full teacher tensor without counted API queries.                                        | Strict oracle abstraction                          | Implement top-K and logit-bias access as separate query-counted interfaces. |

---

# 8. Method Synthesis Table

| Evidence Fragment                 | Source in Repo                       | What It Reveals                               | Generalized Principle                            | Use in New Method?   | How to Transform It                                                   |
| --------------------------------- | ------------------------------------ | --------------------------------------------- | ------------------------------------------------ | -------------------- | --------------------------------------------------------------------- |
| v13/v14 Logit Completion positive | `enhanced_kd_clone.py`, results JSON | Dense tail logits improve top-K KD            | Behavior-aligned dense logit supervision         | Yes                  | Rewrite as strict, query-counted completion with uncertainty weights. |
| hidden MSE negative               | v13/v14 B/C/D                        | h-space objective conflicts                   | Avoid representation-space losses                | Yes, as anti-design  | Remove h-MSE; h estimates only intermediate for logits.               |
| Carlini SVD subspace              | `reproduce_carlini.py`               | logits lie in low-rank output image           | Output subspace is recoverable                   | Yes                  | Use as basis/completion prior and baseline.                           |
| Moment-CP W_lm signal             | paper/ATTACK_4WAY                    | higher-order moments may break gauge          | Gauge-breaking signal can be statistical         | Yes, but gated       | Use only dimensions/factors with null margin; not direct main claim.  |
| Moment internal weak              | ATTACK_4WAY                          | internals not reliably recoverable            | Depth-screened signals need gating               | Yes                  | Tail/gauge confidence, not full internal recovery.                    |
| S-PSI negative                    | CLAIMS_FROM_RESULTS                  | sensitivity/objective mismatch                | Parameter cosine not primary objective           | As negative evidence | Archive as historical diagnostic; keep as ablation.                   |
| active-query bug                  | findings                             | false positives possible                      | Sanity guards required                           | Yes                  | Add init cosine and oracle access tests.                              |
| review-stage audits               | PAPER_CLAIM_AUDIT                    | paper/result mismatch                         | provenance is part of method validity            | Yes                  | Add manifest + table regeneration pipeline.                           |
| random KD negative                | v7/v9                                | from-scratch budget insufficient              | need pretrained manifold or stronger supervision | Yes                  | Baseline only; not main path.                                         |
| full-logit KD positive            | v8                                   | dense full distribution is strong upper bound | completion tries approximate this                | Yes                  | Use full-logit KD as upper bound, not black-box baseline.             |

---

# 9. Missing Mechanism Diagnosis

1. **Missing Mechanism Name**: Query-Budgeted Uncertainty-Gated Logit Completion under Gauge Ambiguity.

2. **One-Sentence Diagnosis**: 当前方法失败的核心不是“观测不够”或“训练不够”，而是把有 gauge ambiguity 的 hidden/parameter-space 信号直接当监督，而没有一个严格黑盒、查询计入、可校准不确定性的 logit-space completion 机制来把 sparse/top-K API 输出转成可信的 dense behavioral target。

3. **Evidence From Positive Results**: v13/v14 显示 dense completion 后 PPL 明显优于 sparse top-K KD；v8 显示 dense/full-logit supervision 在 pretrained manifold 上很强；Moment-CP 与 Carlini 说明 output image / higher-order moments 可能提供 gauge/subspace signal。([GitHub][23])

4. **Evidence From Negative Results**: hidden MSE/SCRD 在 top-K 下灾难性变差；random KD 5k/50k 都失败；S-PSI sensitivity/K expansion/warmstart 不能转化为参数恢复。([GitHub][7])

5. **Evidence From Unstable Results**: active query 伪阳性、paper claim audit FAIL、Moment script quarantine 与 paper 主线冲突，说明缺少严格结果治理。([GitHub][12])

6. **Evidence From Failed Ablations**: beta=0 不坏、K expansion 不好、hidden MSE 反坏，说明旧方法解释机制错误。

7. **Why Existing Method Cannot Solve It**: S-PSI 直接优化参数/局部 sensitivity；SCRD 直接对齐 hidden state；Moment-CP 直接声称权重泄漏。这些都没有把“可观察 top-K/side-channel 信息 → calibrated dense logits → functional student objective”作为主机制。

8. **Why Simple Tuning Cannot Solve It**: v9 50k random KD 仍很差；hidden-MSE 在两个 top-K 设置下稳定变坏；K expansion 不解决。调参无法修复 API leakage、gauge mismatch、uncertainty weighting 的结构缺失。

9. **Why Existing Best Positive Fragment Is Insufficient**: 现有 Logit Completion 的 E variant 使用 full teacher logits 来恢复 probe logits，不能直接支撑 strict black-box top-K claim；它也没有不确定性门控、null margin、query accounting、moment confidence。([GitHub][15])

10. **What New Mechanism Must Do**: 严格限制 API 可见信息，计入 top-K 与 logit-bias/probe 查询预算；把 observed top-K exact logits 与 probe-recovered / low-rank / moment-guided tail logits 合成 dense target；对 tail 使用 calibration uncertainty 和 moment null-margin gating；只优化 logit-space KL/CE，不做 hidden MSE。

11. **Confidence**: medium。正面功能信号明确，但当前 strongest result 有 strict-black-box contamination；需要最小实验验证。

---

# 10. New MAIN METHOD PATH

## Method Name Placeholder

**Q-UMC: Query-budgeted Uncertainty-gated Moment-guided Logit Completion**

1. **One-Sentence Core Idea**: 在严格 top-K + 可选 logit-bias/probe API 下，先用计入预算的 probe logits 与 Carlini/Moment 信号构建带不确定性的 dense logit completion，再用 uncertainty-weighted dense KL 训练 student，完全避免 hidden-state MSE。

2. **Core Missing Mechanism It Adds**: query-accounted completion + uncertainty gating + gauge-aware moment/Carlini prior。

3. **What Phenomena It Explains**: 解释 v13/v14 为什么有效：dense tail target 比 sparse top-K 更接近 full-logit KD；解释 hidden MSE 为什么失败：h-space gauge 与 behavior 不对齐；解释 Moment-CP 为什么有局部信号但不能单独成文：它只能作为 gauge/confidence prior。

4. **What Negative Results It Fixes**: 修复 top-K contamination、hidden MSE conflict、random KD insufficiency、rank≠recoverability、Moment internal weak signal。

5. **What Existing Positive Signals It Generalizes**: generalizes Logit Completion E + Carlini output subspace + Moment-CP W_lm null-margin signal。

6. **Why Existing Best Path Is Not Enough**: Existing E path 没有 strict oracle，没有 query accounting，没有 tail uncertainty，没有 moment/null gating；因此它是线索，不是最终方法。

7. **Core Mechanism**:

   * Strict oracle returns only observed top-K logits.
   * Optional logit-bias/probe oracle returns selected token logits with counted cost.
   * Recover `h_hat` / low-rank coordinates from probe logits.
   * Reconstruct tail logits using `W_hat` from Carlini/Moment-gated prior.
   * Merge exact top-K values.
   * Weight every pseudo-tail token by calibration uncertainty and CP/null confidence.
   * Train with weighted dense logit KL + CE; no hidden MSE.

8. **New Objective / Loss**: uncertainty-weighted completed-logit KL plus optional moment-consistency regularizer only on high-confidence factors.

9. **New Architecture or Module**: no new model architecture; new modules are `StrictTopKOracle`, `LogitBiasProbeOracle`, `CalibratedLogitCompleter`, `MomentConfidenceGate`.

10. **New Training Procedure**: calibration phase → completion phase → KD phase → ablation/falsification phase.

11. **New Evaluation Protocol**: compare A/B/C triad:

* A. Existing Best Positive Fragment Only: old E-style completion, but tagged as simulator.
* B. Q-UMC without new mechanism: strict top-K KD or strict completion without uncertainty/moment gate.
* C. Full Q-UMC.

12. **Existing Components Reused**: KD training loop, Carlini SVD code, functional KL eval, Moment code only after unquarantine/provenance check.

13. **Existing Components Deleted/Archived**: active-query positive runs, contaminated top-K claims, paper Moment-CP main claim until artifact restored.

14. **Existing Components Rewritten**: `enhanced_kd_clone.py` logic should be split into strict oracle/completer/trainer; result logging.

15. **Existing Components Kept Only as Ablation**: hidden MSE/SCRD, S-PSI sensitivity, algebraic init, Moment-only CP.

16. **Existing Components Kept Only as Baseline**: full-logit KD upper bound, sparse top-K KD, old Logit Completion simulator, Carlini SVD.

17. **Why Not Merely Existing Best Path**: Full Q-UMC must beat both sparse top-K KD and old E-style fragment under strict oracle; if it only matches old E under leaked full logits, it fails.

18. **Why This Could Produce Real Positive Results**: It keeps the strongest observed inductive bias—dense logit-space supervision—while fixing the black-box access and uncertainty problems.

19. **Why Mechanism-Level Different from Prior Work**: Carlini/Finlayson recover output subspace/image; Q-UMC uses query-budgeted partial-logit probes plus uncertainty/moment gating to train a functional clone under top-K constraints.

20. **Main Risk**: Once full-logit leakage is removed and probe costs counted, completion may not outperform sparse top-K KD enough to justify novelty.

21. **Minimal Falsification Experiment**: Replace `complete_logits(t_logits_full, ...)` with strict oracle completion; if Full Q-UMC fails to beat strict top-K KD by a pre-registered margin over 3 seeds, stop/pivot.

22. **Confidence**: medium-low as paper route; medium as next experiment path.

---

# 11. Formal Method Description

## Problem Setup

Teacher `T` is accessible through:

* `O_topk(x, K) -> {(i, z_i^T(x))}_{i in I_K(x)}`
* optional `O_probe(x, P) -> {z_j^T(x)}_{j in P}` via logit-bias / probe API, with explicit query cost.

Student `S_θ` is initialized either pretrained-perturbed or from a fair baseline. Goal: minimize teacher-student functional divergence on held-out text under fixed query budget, not recover all internal parameters.

## Existing Method Failure

Current Logit Completion takes full teacher logits and then simulates top-K masking; S-PSI and SCRD optimize hidden/parameter objectives that conflict with gauge-invariant behavior.

## New Insight

The recoverable object is not raw hidden state or internal weights; it is a **calibrated dense behavioral target** that is exact where API reveals logits and uncertainty-weighted elsewhere.

## Method Overview

1. Query top-K logits on train/calibration prompts.
2. Query a small probe set `P` under logit-bias/probe API; count all costs.
3. Estimate low-rank output coordinates `h_hat(x)` via least squares:
   [
   \hat h(x)=\arg\min_h |W_{P}h-z_{P}^{T}(x)|_2^2+\lambda|h|_2^2
   ]
4. Build base completion:
   [
   \hat z_v(x)=\hat W_v^\top \hat h(x)+\hat b_v
   ]
5. Merge exact top-K:
   [
   \tilde z_v(x)=
   \begin{cases}
   z_v^T(x), & v\in I_K(x)\
   \hat z_v(x), & v\notin I_K(x)
   \end{cases}
   ]
6. Estimate uncertainty on held-out probe logits:
   [
   \sigma_v^2=\mathbb E_{x\in C}[(\hat z_v(x)-z_v^T(x))^2]
   ]
7. Gate pseudo-tail:
   [
   w_v(x)=
   \begin{cases}
   1, & v\in I_K(x)\
   g_v / (\sigma_v^2+\epsilon), & v\notin I_K(x)
   \end{cases}
   ]
   where `g_v` comes from Moment/Carlini confidence and null margin; if no verified moment confidence, set `g_v=1` or ablate it.

## Objective

[
L_{\text{total}} =
L_{\text{uc-kl}}

* \lambda_{\text{ce}} L_{\text{CE}}
* \lambda_{\text{mom}} L_{\text{moment}}
* \lambda_{\text{cal}} L_{\text{cal}}
  ]

[
L_{\text{uc-kl}} =
\mathbb E_x
\sum_v
w_v(x),
q_v(\tilde z(x);\tau)
\left[
\log q_v(\tilde z(x);\tau)
--------------------------

\log p_v(S_\theta(x);\tau)
\right]
]

[
L_{\text{cal}} =
\mathbb E_{x,v\in P_{\text{heldout}}}
\left[
\hat z_v(x)-z_v^T(x)
\right]^2
]

[
L_{\text{moment}} =
\left|
\Pi_{\mathcal C}
\left(
M_3(S_\theta)-\hat M_3(T)
\right)
\right|_F^2
]

where `Π_C` projects only onto high-confidence CP/Moment factors whose null margin passed a pre-registered threshold. `L_moment` is optional and must be ablated; it must not be the only source of improvement.

## Algorithm: Q-UMC

**Input**: prompts `D_train`, calibration prompts `D_cal`, top-K API `O_topk`, optional probe API `O_probe`, student `S_θ`, completion basis `W_hat`, budget `B`.
**Output**: trained student, manifest, result table, calibration diagnostics.

Steps:

1. Build strict query oracle; hard fail if any training code accesses full teacher logits outside oracle.
2. Collect top-K logits for `D_train` and `D_cal`.
3. Collect probe logits for selected token set `P`; count every probe query.
4. Estimate `h_hat`, dense completion `z_tilde`, and uncertainty weights `w`.
5. Train student with `L_total`; log top-K exact error, tail calibration MSE, query count, KL, PPL, top1.
6. Run ablations: sparse top-K only, completion without uncertainty, completion without moment gate, full Q-UMC.
7. Evaluate on held-out WikiText and at least one second model gate before paper claim.

**Logging required**:

* `query_count_topk`, `query_count_probe`, `effective_budget`
* `tail_completion_mse`, `tail_completion_kl`, `topk_exact_mae`
* `weight_distribution`, `fraction_tail_active`
* `moment_null_margin`, `moment_gate_tokens`
* `student_ppl`, `teacher_student_KL`, `top1_agreement`
* config, command, git hash, seed, data split hash

---

# 12. Related Work and Novelty Risk

| Paper                                                                                                     | Year / Venue          | Code                   | Mechanism                                                            | Why Close                                                     | Difference from Q-UMC                                                                                               | Novelty Risk | Required Differentiation Experiment                                                                                          |
| --------------------------------------------------------------------------------------------------------- | --------------------- | ---------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| Carlini et al., *Stealing Part of a Production Language Model*                                            | 2024 / ICML-arXiv     | not relied on          | SVD recovers output projection / hidden dimension from logit outputs | Q-UMC uses same low-rank output image premise                 | Q-UMC trains a student under top-K/probe access with calibrated completion, not just recovering projection subspace | high         | Compare against Carlini-only completion and Carlini subspace KD.([arXiv][28])                                                |
| Finlayson et al., *Logits of API-Protected LLMs Leak Proprietary Information*                             | 2024 / COLM           | check official         | Softmax bottleneck / model image; efficient output extraction        | Directly close to output-image and logit-bias attack surface  | Q-UMC must be framed as functional extraction with uncertainty-gated completion, not “image discovery”              | very high    | Strict API query budget vs Finlayson-style image extraction; completion quality under top-K only / top-K+probe.([arXiv][29]) |
| Golowich et al., *Sequences of Logits Reveal Low Rank Structure* / *Provably Learning via Low Logit Rank* | 2025                  | unknown                | Low-rank logit matrices and provable learning                        | Very close to dense completion / low-rank functional learning | Q-UMC uses partial top-K/probe logits and moment confidence for cloning, not general low-rank generation theorem    | high         | Show Q-UMC gains when only partial logits/probe budget are available.([arXiv][30])                                           |
| Nazir et al., PILS                                                                                        | 2025 / NeurIPS        | GitHub listed in paper | Compress next-token distributions for inversion                      | Uses low-dimensional logprob representation                   | Different task: prompt inversion, not model cloning; still close mechanism                                          | medium       | Include as related, not baseline unless prompt inversion task used.([arXiv][31])                                             |
| Dionicio et al., Teacher Scrambling                                                                       | 2025 NeurIPS workshop | unknown                | Tail logits can be scrambled to defeat distillation                  | Directly challenges tail-logit distillation                   | Q-UMC must show uncertainty gate avoids harmful tail supervision                                                    | medium       | Robustness test with scrambled/noisy tail; Q-UMC should downweight bad tail.([OpenReview][32])                               |
| Bhaskara et al., robust Kruskal tensor decomposition                                                      | 2014                  | theory                 | Robust uniqueness of tensor decompositions                           | Supports moment/CP identifiability framing                    | Q-UMC does not claim CP fully identifies transformer weights; uses moment confidence only                           | medium       | Null-margin and synthetic tensor sanity tests.([Proceedings of Machine Learning Research][33])                               |
| Tramèr et al., prediction API model stealing                                                              | 2016 / USENIX         | yes                    | classical prediction API extraction                                  | Broad extraction baseline                                     | Q-UMC specialized to LLM top-K/logit-bias outputs                                                                   | low          | Cite for threat model; no SOTA claim from vision/classical tasks.([USENIX][34])                                              |
| Jagielski et al., high accuracy/high fidelity extraction                                                  | 2020 / USENIX         | some reproductions     | functionality vs fidelity limitations                                | Reviewer may invoke fidelity limits                           | Q-UMC should claim functional clone under limited threat model, not exact fidelity                                  | medium       | Report accuracy/fidelity separately; no overclaim.([USENIX][35])                                                             |
| Orekondy et al., Knockoff Nets                                                                            | 2019 / CVPR           | official repo exists   | query-transfer-set + surrogate training                              | Classic functionality stealing                                | Different modality; relevant as baseline philosophy                                                                 | low          | cite only as functional extraction prior.([CVF开放获取][36])                                                                     |

**Novelty risk summary**: Highest risk is that reviewers say “this is Carlini/Finlayson + KD.” Q-UMC survives only if experiments prove: strict partial-logit/probe access, query accounting, uncertainty gating, and moment/null confidence materially improve over Carlini-only, old Logit Completion, and sparse top-K KD.

---

# 13. Keep / Delete / Rewrite / Archive Plan

| Item                     | Type        | File / Directory / Claim / Experiment   | Current Role                  | Problem Under New MAIN PATH             | Action                                      | Reason                                           |
| ------------------------ | ----------- | --------------------------------------- | ----------------------------- | --------------------------------------- | ------------------------------------------- | ------------------------------------------------ |
| Logit Completion code    | script      | `scripts/enhanced_kd_clone.py`          | strongest functional positive | full-logit leakage in completion        | REWRITE                                     | Split into strict oracle + completer + trainer.  |
| v13/v14 results          | results     | `results/v13_lc_topk20`, `v14_lc_topk5` | positive signal               | not strict black-box                    | KEEP ONLY AS HISTORICAL NEGATIVE EVIDENCE   | Keep as phenomenon; not final evidence.          |
| Sparse top-K KD A        | baseline    | A_logit_only variants                   | baseline                      | okay as baseline                        | KEEP ONLY AS BASELINE                       | Required A-control.                              |
| Hidden MSE/SCRD          | method      | B/C/D variants                          | old proposed improvement      | consistently worse                      | KEEP ONLY AS ABLATION                       | Demonstrates representation conflict.            |
| S-PSI core               | method      | `src/parameter_inverter.py`             | old main                      | not new main; bugs/manifests need fixes | ARCHIVE + KEEP ONLY AS ABLATION             | Preserve as negative evidence/diagnostic.        |
| S-PSI sensitivity claim  | claim       | README/method                           | old novelty                   | contradicted by beta=0/no recovery      | DELETE                                      | Do not claim main mechanism.                     |
| Gramian diagnostics      | analysis    | `src/gramian.py`                        | observability evidence        | raw logs incomplete                     | FREEZE                                      | Use only to motivate gap, not main result.       |
| Algebraic init           | method      | `src/algebraic_init.py`                 | init/warmstart                | marginal                                | KEEP ONLY AS ABLATION                       | Not main method.                                 |
| Active query old results | experiment  | active-query logs                       | false positive                | confirmed bug                           | ARCHIVE                                     | Mark invalid, never cite as success.             |
| `run_active_query.py`    | script      | active query launcher                   | historical                    | needs init sanity                       | REWRITE                                     | Add sanity checks before any future use.         |
| Moment-CP script         | script      | `attack_higher_order_moments.py`        | paper main but quarantined    | provenance conflict                     | ARCHIVE until reproduced                    | Cannot be active main without unquarantine.      |
| Moment-CP claim          | paper claim | W_lm top5 0.813                         | current paper thesis          | raw result missing                      | FREEZE                                      | Only restore after artifact audit.               |
| Carlini reproduction     | baseline    | `reproduce_carlini.py`                  | subspace baseline             | needed                                  | KEEP                                        | Required close baseline.                         |
| Functional KL eval       | eval        | `functional_kl_eval.py`                 | functional metric             | useful but path-specific                | REWRITE                                     | Add Q-UMC variants and strict oracle.            |
| README                   | doc         | `README.md`                             | S-PSI narrative               | inconsistent with new main              | REWRITE                                     | New thesis and clear deprecated branches.        |
| Paper                    | doc         | `paper/main.tex`                        | Moment-CP paper               | current evidence insufficient           | REWRITE                                     | New Q-UMC thesis after minimal experiments pass. |
| Claim audits             | docs        | `PAPER_CLAIM_AUDIT*`                    | self-review                   | valuable                                | KEEP                                        | Supports integrity and reviewer-risk control.    |
| Result aggregation       | scripts     | aggregate functions                     | partial                       | lacks manifest                          | REWRITE                                     | Single table generator from manifest.            |
| Missing remote results   | artifacts   | v1–v6/v5_moments etc.                   | claimed evidence              | absent                                  | ARCHIVE if uploaded; otherwise mark missing | Avoid hidden evidence.                           |

---

# 14. Claude Code Implementation Plan

## Task 1: Quarantine invalid and historical branches

**Purpose**: prevent old false positives from driving new method.
**Which Phenomenon / Constraint It Addresses**: P14/P16, C6/C11.
**Why It Supports New MAIN METHOD PATH**: Q-UMC needs clean evidence base.
**Files to Inspect**: `findings.md`, `ATTACK_4WAY_SUMMARY.md`, `scripts/attack_higher_order_moments.py`, `paper/main.tex`, `README.md`.
**Files to Edit**: `README.md`, new `docs/evidence_registry.md`.
**Files to Delete / Archive**: none silently; add `archived_reason` metadata.
**Functions / Classes**: N/A.
**Exact Change**: create evidence registry labeling active-query positive as invalid, Moment-CP as under-audited/quarantined, v13/v14 as simulator-positive not strict-black-box.
**Do Not Change**: raw results JSON.
**Verification Command**: `grep -R "active query.*success\\|top-5 0.813\\|SOTA" -n README.md paper docs || true`
**Expected Result**: all strong claims point to registry caveats.
**Failure Means**: old claims remain active.
**Rollback Condition**: none; documentation-only.
**Priority**: P0.
**Confidence**: high.

## Task 2: Add strict oracle abstraction

**Purpose**: remove full-logit leakage.
**Addresses**: P17, C1/C8.
**Files to Inspect**: `scripts/enhanced_kd_clone.py`.
**Files to Edit**: new `src/oracles.py`, new tests.
**Exact Change**: implement `StrictTopKOracle`, `ProbeLogitOracle`, `QueryBudget`, with hard prohibition on raw full logits in training path.
**Do Not Change**: teacher model loading semantics except oracle wrapping.
**Verification Command**: `python -m pytest tests/test_oracles.py -q`
**Expected Result**: top-K oracle returns only K ids/values; probe oracle increments query count; accessing unqueried logits raises.
**Failure Means**: black-box simulation unsafe.
**Rollback Condition**: tests cannot be made deterministic on toy logits.
**Priority**: P0.
**Confidence**: high.

## Task 3: Extract and rewrite logit completion module

**Purpose**: implement Q-UMC core.
**Addresses**: C2/C3/C5.
**Files to Inspect**: `scripts/enhanced_kd_clone.py`, `scripts/reproduce_carlini.py`, `functional_kl_eval.py`.
**Files to Edit**: new `src/logit_completion.py`.
**Exact Change**: implement `CalibratedLogitCompleter.fit_calibration`, `complete`, `compute_uncertainty_weights`, `merge_topk_exact`; no full teacher logits except in offline unit tests.
**Do Not Change**: KD trainer yet.
**Verification Command**: `python -m pytest tests/test_logit_completion.py -q`
**Expected Result**: toy linear model completion exact when probes span hidden dim; uncertainty downweights noisy tail.
**Failure Means**: mechanism not reliable.
**Rollback Condition**: completion cannot beat sparse top-K on toy.
**Priority**: P0.
**Confidence**: high.

## Task 4: Add strict Q-UMC training script

**Purpose**: minimal trainable implementation.
**Addresses**: C1–C5.
**Files to Edit**: new `scripts/run_qumc.py`.
**Exact Change**: reuse dataset/evaluate from `enhanced_kd_clone.py`, but all teacher access goes through oracles; implement variants `topk_only`, `completion_no_uncertainty`, `completion_uncertainty`, `full_qumc`.
**Do Not Change**: old `enhanced_kd_clone.py` results.
**Verification Command**: `python scripts/run_qumc.py --num_steps 2 --batch_size 1 --seq_len 16 --eval_batches 1 --topk 5 --probe_tokens 64 --output_dir results/smoke_qumc --allow_synthetic`
**Expected Result**: manifest saved; no direct full-logit access in train loop.
**Failure Means**: strict path not executable.
**Rollback Condition**: smoke cannot run after oracle/unit tests pass.
**Priority**: P0.
**Confidence**: medium-high.

## Task 5: Add manifest and provenance schema

**Purpose**: make results auditable.
**Addresses**: P15, C6/C13.
**Files to Edit**: new `src/result_manifest.py`; update `run_qumc.py`; optionally update `run_spsi.py`.
**Exact Change**: save command, git hash, config, seed, split, query counts, model name, checkpoint hash, code version.
**Do Not Change**: metrics definitions.
**Verification Command**: `python -m pytest tests/test_manifest.py -q`
**Expected Result**: `manifest.json` present and complete.
**Failure Means**: results remain non-reproducible.
**Rollback Condition**: none.
**Priority**: P1.
**Confidence**: high.

## Task 6: Add metric/data sanity tests

**Purpose**: protect against leakage and split errors.
**Addresses**: C6/C13.
**Files to Edit**: `tests/test_metrics.py`, `tests/test_data_splits.py`.
**Exact Change**: verify teacher-vs-teacher KL=0, random student KL>0, top-K masks size K, train/eval split names saved.
**Verification Command**: `python -m pytest tests/test_metrics.py tests/test_data_splits.py -q`
**Expected Result**: deterministic pass on toy model.
**Failure Means**: no trustworthy experiment.
**Priority**: P1.
**Confidence**: high.

## Task 7: Add A/B/C ablation configs

**Purpose**: prove not just old positive fragment.
**Files to Edit**: `configs/qumc_smoke.yaml`, `configs/qumc_minimal.yaml`.
**Exact Change**: configs for A existing fragment only, B strict completion without uncertainty/moment, C full Q-UMC.
**Verification Command**: `python scripts/run_qumc.py --config configs/qumc_smoke.yaml --dry_run`
**Expected Result**: prints exact planned variants and budgets.
**Failure Means**: comparisons not pre-registered.
**Priority**: P1.
**Confidence**: high.

## Task 8: Add Moment confidence gate as optional module

**Purpose**: use CP signal without overclaim.
**Files to Edit**: new `src/moment_gate.py`.
**Exact Change**: load CP factors only if artifact exists; compute null margin; otherwise default gate=1 and log `moment_gate_available=false`.
**Do Not Change**: Q-UMC should run without CP artifacts.
**Verification Command**: `python -m pytest tests/test_moment_gate.py -q`
**Expected Result**: missing artifacts do not crash; unverified CP cannot silently affect results.
**Failure Means**: provenance risk.
**Priority**: P2.
**Confidence**: medium.

## Task 9: Reproduce current positive and negative fragments under old script

**Purpose**: establish anchors as baselines, not final method.
**Files to Edit**: none unless script broken.
**Command**: `python scripts/enhanced_kd_clone.py --num_steps 500 --batch_size 4 --eval_every 500 --eval_batches 10 --topk_kd 5 --init_mode pretrained_perturbed --output_dir results/repro_old_lc_small`
**Expected Result**: old E-style completion beats A in simulator.
**Failure Means**: old positive not reproducible even as signal.
**Priority**: P2.
**Confidence**: medium.

## Task 10: Run strict minimal Q-UMC experiment

**Purpose**: validate new mechanism.
**Command**: `python scripts/run_qumc.py --config configs/qumc_minimal.yaml --seeds 0 1 2 --output_dir results/qumc_minimal`
**Expected Result**: Full Q-UMC beats strict top-K KD and completion without uncertainty by pre-registered margin.
**Failure Means**: stop or pivot to honest “strict top-K completion not enough”.
**Priority**: P0 after tasks 1–8.
**Confidence**: medium.

## Task 11: Update paper only after gate passes

**Purpose**: prevent narrative chasing.
**Files to Edit**: `paper/main.tex`, `README.md`.
**Exact Change**: replace Moment-CP/S-PSI thesis with Q-UMC thesis only if minimal tests pass; otherwise write negative report.
**Verification Command**: `python scripts/make_results_table.py --results results/qumc_minimal --out paper/tables/qumc_minimal.tex`
**Expected Result**: paper tables generated from manifests.
**Failure Means**: paper not ready.
**Priority**: P3.
**Confidence**: high.

---

# 15. Minimal Verification Experiments

| Priority | Experiment                              | Hypothesis                                 | Command                                                                                                   | Config       | Dataset      | Seeds   | Metric                  | Success Criterion                 | Failure Interpretation          |
| -------: | --------------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------- | ------------ | ------------ | ------- | ----------------------- | --------------------------------- | ------------------------------- |
|        0 | Smoke test                              | Q-UMC code runs without full logits        | `python scripts/run_qumc.py --num_steps 2 --batch_size 1 --seq_len 16 --eval_batches 1 --allow_synthetic` | smoke        | synthetic    | 0       | no crash/query manifest | manifest saved                    | implementation broken           |
|        0 | Data sanity                             | train/eval split distinct                  | `python -m pytest tests/test_data_splits.py -q`                                                           | N/A          | toy/WikiText | N/A     | split hash              | pass                              | leakage risk                    |
|        0 | Metric sanity                           | KL/CE/top1 correct                         | `python -m pytest tests/test_metrics.py -q`                                                               | N/A          | toy          | N/A     | KL=0 teacher/self       | pass                              | metric bug                      |
|        0 | One-batch overfit                       | student can fit completed logits           | `python scripts/run_qumc.py --overfit_one_batch --num_steps 100`                                          | toy          | synthetic    | 0       | train KL                | decreases >90%                    | optimization bug                |
|        0 | Checkpoint loading                      | no stale config resume                     | `python -m pytest tests/test_manifest.py -q`                                                              | N/A          | toy          | N/A     | hash guard              | mismatch hard fails               | stale contamination             |
|        1 | Reproduce current negative              | random KD remains bad                      | `python scripts/enhanced_kd_clone.py --num_steps 500 --init_mode random ...`                              | old-small    | WikiText     | 42      | PPL/KL                  | bad vs pretrained                 | old negative unstable           |
|        1 | Reproduce old positive fragment         | old E beats A in simulator                 | `python scripts/enhanced_kd_clone.py --num_steps 500 --topk_kd 5 ...`                                     | old-lc-small | WikiText     | 42      | PPL/KL                  | E < A                             | old signal absent               |
|        1 | Strict oracle activation                | no full logits accessible                  | `python -m pytest tests/test_oracles.py -q`                                                               | N/A          | toy          | N/A     | access exception        | pass                              | invalid black-box               |
|        1 | Completion calibration check            | tail uncertainty predicts error            | `python scripts/run_qumc.py --calibration_only ...`                                                       | calibration  | WikiText     | 0       | tail MSE/ECE            | high-error tails downweighted     | uncertainty useless             |
|        1 | New MAIN minimal test                   | Q-UMC beats strict top-K                   | `python scripts/run_qumc.py --config configs/qumc_minimal.yaml --variant full_qumc`                       | minimal      | WikiText     | 0,1,2   | PPL/KL/top1             | mean PPL lower by pre-reg margin  | stop/pivot                      |
|        1 | A. Existing Best Positive Fragment Only | old E-style simulator baseline             | old script command                                                                                        | old-E        | WikiText     | 0,1,2   | PPL                     | report only                       | shows old leakage ceiling       |
|        1 | B. New without new mechanism            | strict completion no uncertainty/moment    | `--variant completion_no_uncertainty`                                                                     | minimal      | WikiText     | 0,1,2   | PPL/KL                  | worse than full                   | if same, mechanism not needed   |
|        1 | C. Full New MAIN                        | full Q-UMC                                 | `--variant full_qumc`                                                                                     | minimal      | WikiText     | 0,1,2   | PPL/KL                  | beats A strict/B                  | validates mechanism             |
|        2 | Key ablation remove moment gate         | moment not necessary or useful             | `--disable_moment_gate`                                                                                   | minimal      | WikiText     | 0,1,2   | PPL/KL                  | full ≥ no-moment if claim moment  | if no gain, delete moment claim |
|        2 | Key ablation remove uncertainty         | uncertainty matters                        | `--disable_uncertainty`                                                                                   | minimal      | WikiText     | 0,1,2   | PPL/KL/calibration      | full beats no-uncertainty         | if not, simplify                |
|        2 | Small baseline comparison               | top-K KD / full-logit upper / Carlini-only | `--run_baselines`                                                                                         | minimal      | WikiText     | 0,1,2   | PPL/KL                  | Q-UMC between topK and full upper | if below topK, fail             |
|        2 | Multi-seed stability                    | not seed cherry-pick                       | `--seeds 0 1 2 3 4`                                                                                       | minimal      | WikiText     | 5       | mean±std                | CI excludes zero improvement      | instability                     |
|        3 | Expansion gate                          | larger steps maintain gain                 | `--config configs/qumc_medium.yaml`                                                                       | medium       | WikiText     | 0,1,2   | PPL/KL                  | gain persists                     | overfit small setup             |
|        3 | Official baseline reproduction          | fair external comparison                   | official/faithful scripts                                                                                 | baseline     | WikiText     | matched | PPL/KL                  | reproduce reported ranges         | no SOTA claim                   |
|        3 | Unified environment comparison          | all methods same env                       | `scripts/run_all_qumc_baselines.sh`                                                                       | unified      | WikiText     | 3       | table                   | same GPU/model/data               | unfair comparison               |
|        3 | Generalization test                     | not Qwen-only                              | `--model_name second_small_model`                                                                         | minimal      | second LM    | 0,1,2   | PPL/KL                  | same qualitative ordering         | architecture-specific           |
|        3 | Statistical significance                | robust claim                               | `python scripts/bootstrap_ci.py results/qumc_minimal`                                                     | N/A          | heldout      | 5       | CI/p-value              | significant pre-reg metric        | no paper claim                  |

---

# 16. Baseline and SOTA Plan

| Baseline                         | Why Required                    | Official Code                            | Dataset           | Metric              | Reproduction Requirement                 | Fairness Risk                   |
| -------------------------------- | ------------------------------- | ---------------------------------------- | ----------------- | ------------------- | ---------------------------------------- | ------------------------------- |
| Sparse top-K KD                  | simplest strong strict baseline | repo implementation after oracle rewrite | WikiText          | PPL/KL/top1         | same top-K oracle/budget                 | must not use full logits        |
| Full-logit KD upper bound        | measures ceiling                | repo                                     | WikiText          | PPL/KL              | full logits allowed only as oracle upper | cannot compare as black-box     |
| Old Logit Completion E           | existing best positive fragment | repo                                     | WikiText          | PPL/KL              | label as simulator/leaked-probe          | cannot be final baseline        |
| Strict completion no uncertainty | isolates uncertainty mechanism  | new                                      | WikiText          | PPL/KL              | same probe budget                        | if equal, Q-UMC novelty weak    |
| Strict completion no moment gate | isolates moment contribution    | new                                      | WikiText          | PPL/KL              | same budget                              | if equal, delete moment claim   |
| Carlini-only completion          | close mechanism baseline        | repo reproduction                        | WikiText          | PPL/KL/subspace     | same queries                             | do not use oracle W_lm          |
| Finlayson/image extraction style | closest mechanism prior         | need official/faithful                   | WikiText          | completion error/KL | reproduce image basis extraction         | novelty high risk               |
| Matched KD baseline              | current repo baseline           | repo                                     | WikiText          | PPL/KL              | same init, steps, data                   | avoid weak random-only baseline |
| Random KD                        | negative baseline               | repo                                     | WikiText          | PPL/KL              | include but not sole baseline            | too weak alone                  |
| Teacher-scrambled/noisy tail     | robustness baseline             | implement defense                        | WikiText          | PPL/KL              | same top-K utility                       | checks harmful tail logits      |
| Second-model replication         | generalization                  | repo/new config                          | non-Qwen small LM | PPL/KL              | same protocol                            | broad claim requires it         |

---

# 17. Paper Thesis Reconstruction

1. **New Paper Thesis**: Sparse top-K logprob APIs leak more functional information than top-K KD can use, but only if the attacker converts partial outputs into a calibrated dense logit target under a strict query budget.

2. **Main Technical Contribution**: Q-UMC, a query-budgeted uncertainty-gated logit completion method that combines exact top-K logits, counted probe logits, low-rank/Carlini output image, and optional moment confidence without hidden-state MSE.

3. **Main Empirical Claim**: Under the same top-K/probe budget, Q-UMC improves held-out teacher-student KL/PPL over sparse top-K KD and strict completion without uncertainty across multiple seeds.

4. **What Previous Failures Taught Us**: Parameter/hidden-space supervision is misaligned; observability is not recoverability; dense behavioral targets matter; false positives are easy without strict oracle and provenance.

5. **What We Should Not Claim**:

   * no full internal parameter theft;
   * no SOTA;
   * no strict top-K-only success unless logit-bias/probe-free variant passes;
   * no Moment-CP main claim until raw artifacts and scripts are restored;
   * no active-query success.

6. **What We Can Claim If Experiments Pass**:

   * Q-UMC beats sparse top-K KD under strict top-K+probe API;
   * uncertainty gating matters;
   * hidden-state MSE is harmful in this setting;
   * output-image / moment signals are useful only as calibrated priors.

7. **Required Baselines**: sparse top-K KD, full-logit KD upper bound, old E simulator, Carlini-only, strict completion no uncertainty, strict completion no moment, matched KD, second-model replication.

8. **Required Ablations**: remove uncertainty, remove probe, remove moment gate, remove exact top-K merge, vary top-K, vary probe budget, vary seed.

9. **Required Robustness Tests**: noisy probe logits, scrambled tails, lower top-K, different model, OOD prompts.

10. **Reviewer Likely Objections**:

* “This is Carlini/Finlayson + KD.”
* “The old results used full logits.”
* “Moment-CP artifact is missing.”
* “Only Qwen/seed 42.”
* “No official baselines.”

11. **How New MAIN METHOD Answers Them**: strict oracle, query accounting, A/B/C ablation, multi-seed, artifact manifest, direct comparison to Carlini/Finlayson-style baselines.

12. **What Would Make This NeurIPS-Strong**: strict black-box protocol, robust multi-seed improvement, clear mechanism logs showing uncertainty gate predicts tail error, fair close-baseline reproduction, second architecture.

13. **What Would Make This Rejected**: only old v13/v14 leaked-logit results; no strict oracle; missing Moment artifacts; overclaiming parameter theft or SOTA.

14. **What Would Be Required for Oral-Level Strength**: strong gains across several models/API restrictions, theoretical sample/query analysis for uncertainty-gated completion, public reproducible artifacts.

15. **What Would Be Required for Best-Paper-Level Strength**: broad new security implication with production-like API simulation, defenses, theory, official comparisons, and independently reproducible results.

---

# 18. Reviewer Risk Assessment

| Risk                       | Why Reviewer May Object                    | Evidence Needed                     | How Q-UMC Addresses It                        | Remaining Weakness                   |
| -------------------------- | ------------------------------------------ | ----------------------------------- | --------------------------------------------- | ------------------------------------ |
| novelty risk               | close to Carlini/Finlayson/low-rank logits | mechanism ablations                 | uncertainty-gated strict completion           | still close                          |
| incremental risk           | may look like KD engineering               | A/B/C proof                         | show strict oracle + calibration is essential | gains may be small                   |
| baseline weakness          | random KD too weak                         | strong top-K/full/Carlini baselines | required baseline table                       | official code burden                 |
| reproducibility            | current repo lacks manifests               | manifest + table generator          | Task 5                                        | old results still messy              |
| cherry-picking             | single seed v13/v14                        | 5 seed CI                           | minimal experiments                           | compute cost                         |
| negative hiding            | many failed branches                       | evidence registry                   | archive not delete                            | paper must discuss honestly          |
| overclaiming               | current paper too strong                   | claim-code-result matrix            | rewrite thesis                                | discipline needed                    |
| unclear mechanism          | old S-PSI explanations failed              | gate diagnostics                    | tail MSE/uncertainty logs                     | moment role may vanish               |
| ablation insufficiency     | need prove not old E                       | A/B/C core                          | required triad                                | old E not strict                     |
| dataset limitation         | WikiText-only                              | second model/dataset                | expansion gate                                | may fail generalization              |
| compute unfairness         | probe queries extra                        | query accounting                    | oracle budget                                 | comparing to full-logit upper tricky |
| implementation reliability | active-query bug                           | sanity tests                        | tests and hard guards                         | legacy code remains                  |
| related work omission      | many close LLM logit papers                | related-work table                  | cite Carlini/Finlayson/Golowich/PILS          | literature evolves fast              |

---

# 19. Final Decision

1. **One-Sentence Verdict**:
   从所有现象推导出的唯一推荐主线是 **Q-UMC: Query-budgeted Uncertainty-gated Moment-guided Logit Completion**，即把当前 Logit Completion 的正面信号重写成严格黑盒、查询计入、tail 不确定性门控、moment/Carlini 只作 gauge/completion prior 的功能克隆方法。

2. **Current Most Likely Root Cause**:
   主要是 **missing mechanism + evaluation/access bug**：旧方法缺少 gauge-aware calibrated logit completion；同时 v13/v14 completion path 对 strict top-K claim 有 full-logit leakage 风险。

3. **Why This Is Not Just the Existing Best Path**:
   现有 E variant 是 simulator positive；Q-UMC 新增 strict oracle、probe query accounting、uncertainty tail weighting、moment/null confidence gate、A/B/C falsification。它不是保留 E，而是把 E 的“dense logits 有用”现象改造成可发表机制。

4. **Phenomena Explained**:
   v13/v14 positive、v8 dense KD positive、hidden MSE failure、random KD failure、S-PSI observability/recoverability gap、Moment-CP weak-but-positive W_lm signal、active-query false positive。

5. **Mechanism Missing in Current Method**:
   严格黑盒下从 sparse/partial logits 到可信 dense target 的 calibrated completion mechanism。

6. **New Mechanism**:
   query-budgeted strict top-K/probe oracle + calibrated dense completion + uncertainty/moment confidence gate + weighted logit KL.

7. **What to Delete / Archive / Rewrite**:
   Delete/weaken S-PSI recovery claims and hidden-MSE claims; archive active-query positive and quarantined Moment-CP as historical until reproduced; rewrite `enhanced_kd_clone.py` path into strict modules; rewrite README/paper after minimal pass.

8. **First Five Claude Code Tasks**:

   1. create evidence registry and quarantine invalid claims;
   2. implement strict oracle/query budget;
   3. implement calibrated logit completer;
   4. add `run_qumc.py`;
   5. add manifest/provenance schema.

9. **Minimal Experiments**:
   smoke, data/metric sanity, one-batch overfit, checkpoint guard, reproduce old positive/negative, strict oracle activation, calibration check, A/B/C triad, multi-seed stability.

10. **Continue / Stop / Pivot Criteria**:

* Continue: Full Q-UMC beats strict top-K KD and no-uncertainty completion over ≥3 seeds with valid query accounting.
* Stop: strict completion loses all old advantage after full-logit leakage removal.
* Pivot: uncertainty helps but moment gate does not; drop Moment claim and make paper pure calibrated completion.

11. **NeurIPS-Level Gap**:
    strict black-box experiments, official close baselines, second model, multi-seed CI, reproducible manifests, honest related work.

12. **Oral / Best Paper Gap**:
    broader production-like API evidence, stronger theory, defenses, multiple architectures, independent reproducibility.

13. **Confidence**:
    **medium-low for publication success, medium for next experimental path**. The phenomenon direction is clear, but current strongest result must survive strict oracle rewriting.

---

# 20. Final Claude Code Instruction

```text
Claude Code, execute the following plan.

You must implement the New MAIN METHOD PATH defined in the GPT-5.5 Pro diagnosis report:

Q-UMC: Query-budgeted Uncertainty-gated Moment-guided Logit Completion.

Do not invent a different method.
Do not optimize for superficial positive results.
Do not weaken baselines.
Do not delete negative results silently.
Do not change metrics or datasets unless explicitly instructed.
Do not rewrite unrelated files.

Your tasks are:

1. Create docs/evidence_registry.md.
   - Mark active-query positive results as INVALID.
   - Mark v13/v14 Logit Completion as simulator-positive but not strict-black-box evidence.
   - Mark Moment-CP as under-audited/quarantined until raw artifacts are restored.
   - Do not delete raw results.

2. Implement src/oracles.py.
   - Add StrictTopKOracle, ProbeLogitOracle, QueryBudget.
   - The training path must not access full teacher logits except in explicit oracle-upper-bound baselines.
   - Accessing unqueried logits must raise an error.
   - Add tests/test_oracles.py.

3. Implement src/logit_completion.py.
   - Add CalibratedLogitCompleter.
   - Fit probe-based h_hat / low-rank coordinates.
   - Merge exact observed top-K logits.
   - Estimate held-out tail uncertainty.
   - Produce uncertainty weights.
   - Do not use direct full t_logits in strict mode.
   - Add tests/test_logit_completion.py.

4. Implement src/result_manifest.py.
   - Save command, git hash, model, dataset split, seed, config, query counts, checkpoint hashes, metrics, and code version.
   - Add tests/test_manifest.py.

5. Add scripts/run_qumc.py.
   - Variants:
     A. strict_topk_kd
     B. completion_no_uncertainty
     C. completion_uncertainty
     D. full_qumc
     E. full_logit_upper_bound only as oracle upper bound
     F. old_lc_simulator only as historical comparison
   - All teacher calls in A-D must go through strict oracles.
   - Save manifest.json and results.json.

6. Add configs/qumc_smoke.yaml and configs/qumc_minimal.yaml.
   - Include topk, probe_tokens, query budget, seeds, dataset, eval_batches, num_steps.
   - Add --dry_run support to print planned variants and budgets.

7. Add sanity tests.
   - Metric sanity: teacher-vs-teacher KL = 0.
   - Data sanity: train/eval split names and hashes are saved.
   - Init sanity: student must not equal teacher unless explicitly oracle upper bound.
   - Checkpoint sanity: config/model hash mismatch must hard fail.

8. Run verification commands in order:
   - python -m pytest tests/test_oracles.py -q
   - python -m pytest tests/test_logit_completion.py -q
   - python -m pytest tests/test_manifest.py -q
   - python -m pytest tests/test_metrics.py tests/test_data_splits.py -q
   - python scripts/run_qumc.py --config configs/qumc_smoke.yaml --dry_run
   - python scripts/run_qumc.py --config configs/qumc_smoke.yaml --output_dir results/smoke_qumc

9. Only after all smoke tests pass, run the minimal A/B/C queue:
   - python scripts/run_qumc.py --config configs/qumc_minimal.yaml --seeds 0 1 2 --output_dir results/qumc_minimal

10. Do not update paper/main.tex until results/qumc_minimal contains:
   - strict_topk_kd
   - completion_no_uncertainty
   - full_qumc
   - full_logit_upper_bound
   - old_lc_simulator
   - manifest for every run
   - mean/std over seeds

For every task:
- make the smallest necessary change;
- show the diff;
- run the specified verification command;
- save logs;
- report failures;
- stop if verification fails;
- do not proceed to full benchmark until minimal tests pass.

At the end, output:
- files changed;
- files archived;
- configs added;
- commands run;
- logs;
- result table;
- failed checks;
- unresolved issues;
- whether Full Q-UMC beats:
  A. Existing Best Positive Fragment Only,
  B. Q-UMC Without New Mechanism,
  C. Full Q-UMC.
```

[1]: https://github.com/Sunshine535/nips-modelsteal "GitHub - Sunshine535/nips-modelsteal: ModelSteal: Progressive Layer-wise Parameter Inversion of LLMs (NeurIPS 2026) · GitHub"
[2]: https://github.com/Sunshine535/nips-modelsteal/tree/main/src "nips-modelsteal/src at main · Sunshine535/nips-modelsteal · GitHub"
[3]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/README.md "raw.githubusercontent.com"
[4]: https://github.com/Sunshine535/nips-modelsteal/tree/main/paper "nips-modelsteal/paper at main · Sunshine535/nips-modelsteal · GitHub"
[5]: https://github.com/Sunshine535/nips-modelsteal/tree/main/scripts "https://github.com/Sunshine535/nips-modelsteal/tree/main/scripts"
[6]: https://github.com/Sunshine535/nips-modelsteal/tree/main/configs "nips-modelsteal/configs at main · Sunshine535/nips-modelsteal · GitHub"
[7]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/results/v7_scrd/results.json "https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/results/v7_scrd/results.json"
[8]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/EXPERIMENT_PROGRESS.md "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/CLAIMS_FROM_RESULTS.md "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/requirements.txt "https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/requirements.txt"
[11]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/src/parameter_inverter.py "raw.githubusercontent.com"
[12]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/findings.md "raw.githubusercontent.com"
[13]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/src/permutation_alignment.py "raw.githubusercontent.com"
[14]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/scripts/run_spsi.py?plain=1 "https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/scripts/run_spsi.py?plain=1"
[15]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/scripts/enhanced_kd_clone.py "https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/scripts/enhanced_kd_clone.py"
[16]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/scripts/reproduce_carlini.py "https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/scripts/reproduce_carlini.py"
[17]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/scripts/attack_higher_order_moments.py "https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/scripts/attack_higher_order_moments.py"
[18]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/scripts/functional_kl_eval.py "https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/scripts/functional_kl_eval.py"
[19]: https://github.com/Sunshine535/nips-modelsteal/tree/main/results "nips-modelsteal/results at main · Sunshine535/nips-modelsteal · GitHub"
[20]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/paper/main.tex "raw.githubusercontent.com"
[21]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/PAPER_CLAIM_AUDIT.md "raw.githubusercontent.com"
[22]: https://github.com/Sunshine535/nips-modelsteal/tree/main/tests "nips-modelsteal/tests at main · Sunshine535/nips-modelsteal · GitHub"
[23]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/results/v13_lc_topk20/results.json "https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/results/v13_lc_topk20/results.json"
[24]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/results/v14_lc_topk5/results.json "https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/results/v14_lc_topk5/results.json"
[25]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/results/v8_scrd_pretrained/results.json "https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/results/v8_scrd_pretrained/results.json"
[26]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/results/v9_scrd_long/results.json "https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/results/v9_scrd_long/results.json"
[27]: https://raw.githubusercontent.com/Sunshine535/nips-modelsteal/main/ATTACK_4WAY_SUMMARY.md "raw.githubusercontent.com"
[28]: https://arxiv.org/abs/2403.06634 "https://arxiv.org/abs/2403.06634"
[29]: https://arxiv.org/html/2403.09539v3 "https://arxiv.org/html/2403.09539v3"
[30]: https://arxiv.org/abs/2510.24966 "https://arxiv.org/abs/2510.24966"
[31]: https://arxiv.org/pdf/2506.17090 "https://arxiv.org/pdf/2506.17090"
[32]: https://openreview.net/pdf?id=g9vFg3O8YY "https://openreview.net/pdf?id=g9vFg3O8YY"
[33]: https://proceedings.mlr.press/v35/bhaskara14a.pdf "https://proceedings.mlr.press/v35/bhaskara14a.pdf"
[34]: https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer "https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer"
[35]: https://www.usenix.org/conference/usenixsecurity20/presentation/jagielski "https://www.usenix.org/conference/usenixsecurity20/presentation/jagielski"
[36]: https://openaccess.thecvf.com/content_CVPR_2019/html/Orekondy_Knockoff_Nets_Stealing_Functionality_of_Black-Box_Models_CVPR_2019_paper.html "https://openaccess.thecvf.com/content_CVPR_2019/html/Orekondy_Knockoff_Nets_Stealing_Functionality_of_Black-Box_Models_CVPR_2019_paper.html"
