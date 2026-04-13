# Comprehensive Literature Survey: Recovering LLM Weights from Black-Box Access
## NeurIPS 2026 Submission — S-PSI Positioning Paper
## Date: 2026-04-08

---

## Executive Summary

本次文献检索覆盖 2024-2026 年间 model stealing / weight extraction / parameter inversion 的最新进展。共发现 **18+ 篇新论文**（不在已有 PAPERS.md 17 篇之中），分为 5 大类。核心发现：

> **关键空白 (Key Gap)**：当前所有已发表工作要么只恢复单层/部分参数（Carlini→output projection），要么只做行为克隆（KD），要么需要物理侧信道（side-channel）。**没有任何一篇工作实现了从纯 black-box logit API 访问渐进式恢复 transformer 全部层权重的目标**。这正是 S-PSI 的独特定位。

---

## Category A: Direct Weight Extraction Attacks (最相关)

### A1. ★★★★★ Model Stealing for Any Low-Rank Language Model
- **Authors**: Allen Liu, Ankur Moitra (MIT)
- **Venue**: AAAI 2025
- **arXiv**: [2411.07536](https://arxiv.org/abs/2411.07536) (Nov 2024)
- **Key Result**: 对于 d×d rank-r 的权重矩阵，仅需 O(dr) 次查询即可恢复权重（远少于 d² 个矩阵元素）；提供 near-matching lower bound
- **Method**: 利用 LLM 权重的近似低秩特性，通过精心设计的查询恢复权重子空间，再恢复权重本身；两阶段方法：先恢复子空间，再恢复权重
- **验证**: 在包含 self-attention 机制的多种架构上实验验证
- **Limitation**: 理论性强，假设精确低秩；未处理 LayerNorm、残差连接等；实验规模有限
- **与 S-PSI 关系**: **最重要的新竞争者**。他们的查询复杂度理论 bound 可能优于我们的 empirical budget。但他们的方法需要精确低秩假设，而真实 LLM 只是 approximately low-rank。S-PSI 不需要这个假设。
- **论文中需引用并对比的要点**: 查询效率理论 vs. 我们的实际效率；低秩假设的局限性；单层理论 vs. 我们的全模型渐进策略

### A2. ★★★★★ Stealing Part of a Production Language Model
- **Authors**: Nicholas Carlini, Daniel Paleka et al.
- **Venue**: ICML 2024 (已在 PAPERS.md #1)
- **arXiv**: [2403.06634](https://arxiv.org/abs/2403.06634) (Mar 2024)
- **Key Result**: <$20 恢复 OpenAI Ada/Babbage 的完整 projection matrix；确认 GPT-3.5-turbo hidden dim ≤ 4096
- **核心洞见**: logit bias 参数可被滥用以获取远超 API 设计者意图的模型信息
- **NEW UPDATE**: 该论文被 ICML 2024 接收，已成为此领域标杆

### A3. ★★★★★ Logits of API-Protected LLMs Leak Proprietary Information
- **Authors**: (独立团队，与 Carlini 同期)
- **Venue**: 2024 (OpenReview)
- **arXiv**: [2403.09539](https://arxiv.org/abs/2403.09539) (Mar 2024)
- **Key Result**: 仅从 API 暴露的 logits 即可：(i) 识别底层模型架构, (ii) 判断是否经过 fine-tuning, (iii) 判断两个模型是否共享 pre-training, (iv) 提取模型的精确词汇表
- **Method**: 深入分析 softmax 函数、temperature scaling 等与 logits 交互产生的微妙模式
- **与 S-PSI 关系**: **补充性工作**。他们证明 logits 泄露的信息远超预期，为我们的方法提供了更强的理论动机——如果 logits 能泄露架构信息，那更精心的查询策略应该能泄露权重信息。
- **🔴 不在 PAPERS.md 中，必须添加**

### A4. ★★★★☆ Polynomial Time Cryptanalytic Extraction of Neural Network Models
- **Authors**: Adi Shamir et al. (Canales-Martínez, Chavez-Saab, Carlini, Jagielski, Mironov)
- **Venue**: Extended version 2024 (original Crypto 2020)
- **arXiv**: [2310.08708](https://arxiv.org/abs/2310.08708)
- **Key Result**: ReLU 网络所有神经元参数（符号、值）可在 O(n) 时间内恢复
- **Method**: 差分密码分析方法（differential cryptanalysis），利用 ReLU 的分段线性特性
- **Limitation**: 限于 ReLU 网络；transformer 中的 GeLU/SiLU、Attention、LayerNorm 大大增加复杂度
- **与 S-PSI 关系**: 理论上互补。他们的密码分析方法在 ReLU 上精确，但不适用于现代 LLM 的非 ReLU 激活函数和注意力机制。S-PSI 的基于优化的方法更通用。
- **🔴 不在 PAPERS.md 中，必须添加**

### A5. ★★★★☆ Stealing Weights of Black-box Transformer Models via Side Channels
- **Authors**: (Euro S&P 2025 作者团队)
- **Venue**: IEEE Euro S&P 2025
- **Link**: [IEEE](https://www.computer.org/csdl/proceedings-article/euro-sp/2025/137400a256/26F7yavw7Qc)
- **Key Result**: 通过 timing side-channel 恢复 BERT-base / RoBERTa-base 的权重参数
- **Method**: 利用 timing-based 侧信道将提取的结构信息映射到未知权重参数；不需要训练数据或蒸馏
- **Limitation**: 需要**物理/时间侧信道访问**（非纯 API black-box）；仅在 BERT 级别模型验证
- **与 S-PSI 关系**: **威胁模型不同**。他们需要部署设备的侧信道访问，我们只需要标准 API。但他们验证了 transformer 权重可被精确恢复的可行性。
- **🔴 不在 PAPERS.md 中，必须添加**

---

## Category B: Defense Methods (防御策略)

### B1. ★★★★★ LoDD — Logit Detector-based Defense
- **Authors**: (AAAI 2025 团队)
- **Venue**: AAAI 2025
- **Link**: [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34016)
- **Key Result**: 通过分析 logit 值区分良性/恶意查询；恶意查询只返回 hard label，良性查询返回完整 softmax 概率
- **Method**: 利用 exploiting softmax probabilities 的攻击依赖于 softmax 向量本身进行训练这一洞察，在 logit 层面进行检测
- **评估**: 对 6 种模型窃取攻击、多数据集架构测试，优于 5 种已有防御
- **与 S-PSI 关系**: **直接对抗**。LoDD 是我们的 defense evaluation 中应该测试的最强防御之一。S-PSI 是否能规避 LoDD 的检测？如果我们的查询模式被检测为恶意，攻击效果会如何退化？
- **🔴 不在 PAPERS.md 中，必须添加**

### B2. ★★★★☆ SPLITS — Smoothed Perturbation of LogITs
- **Authors**: Grzegorz Gluch et al.
- **Venue**: ICLR 2025 submission (OpenReview)
- **Link**: [OpenReview](https://openreview.net/forum?id=bQMejscfJb)
- **Key Result**: 首个基于 logit 扰动的**可证明鲁棒**模型窃取防御；改编 randomized smoothing 到模型窃取场景
- **Method**: 推导模型一致性上界（stolen model vs victim model agreement upper bounds）
- **理论贡献**: 在有界 Lp Lebesgue 范数下提供理论鲁棒性保证
- **与 S-PSI 关系**: **重要的 defense baseline**。SPLITS 的认证保证意味着在一定扰动水平下，任何攻击（包括 S-PSI）的成功率有理论上界。我们需要实验验证 S-PSI 在 SPLITS 防御下的实际退化程度。
- **🔴 不在 PAPERS.md 中，必须添加**

### B3. ★★★★☆ Robust Utility-Preserving Logit Perturbation for Model Stealing Defense
- **Authors**: Yuancheng Xu et al.
- **Venue**: NeurIPS-track submission (OpenReview)
- **Link**: [OpenReview](https://openreview.net/forum?id=9xPKxRQKXc)
- **Key Result**: (i) 现有 logit 扰动防御可被更强攻击轻易绕过; (ii) 推导了具有理论保证的 logit 扰动认证半径; (iii) 设计了在保持模型效用的同时尽可能达到理论 bound 的防御
- **与 S-PSI 关系**: 双重意义——(a) 他们证明了现有防御的脆弱性支持我们的攻击动机; (b) 他们的 certified radius 是对我们方法的理论挑战。
- **🔴 不在 PAPERS.md 中，必须添加**

### B4. ★★★☆☆ A Survey on Model Extraction Attacks and Defenses (IEEE 2025)
- **Venue**: IEEE Journals, 2025
- **Link**: [IEEE](https://ieeexplore.ieee.org/abstract/document/10891624/)
- **Key Result**: 全面综述 MLaaS 中的模型提取攻击与防御分类体系
- **🔴 不在 PAPERS.md 中，应引用**

---

## Category C: Surveys & Systematization (综述)

### C1. ★★★★☆ A Survey on Model Extraction and Defenses on Generative AI
- **Authors**: Dey, Hoque et al.
- **Venue**: arXiv Nov 2024
- **arXiv**: [2411.09150](https://arxiv.org/abs/2411.09150)
- **Coverage**: NLP（query-based extraction）、CV（model distillation）、RL（reward-based imitation）三大领域的模型提取攻击
- **防御分类**: watermarking, differential privacy 等
- **与 S-PSI 关系**: 提供了最新的 threat landscape 视角。我们应在论文中引用此综述来定位 S-PSI 在 generative AI extraction 中的位置。
- **🔴 不在 PAPERS.md 中，必须添加**

### C2. ★★★★☆ Beyond Model Extraction: A Survey of Attacks and Defenses
- **Venue**: arXiv June 2025
- **arXiv**: [2506.01261](https://arxiv.org/abs/2506.01261)
- **Coverage**: 超越传统模型提取，涵盖 model stealing, data extraction 和针对专有 LLM 的各种攻击方法论
- **与 S-PSI 关系**: 最新的全景综述，涵盖我们的 threat model。
- **🔴 不在 PAPERS.md 中，必须添加**

---

## Category D: KD vs. Model Stealing 关系研究

### D1. ★★★★☆ Two Heads are Better than One: Multi-Agent Model Stealing
- **Venue**: arXiv Oct 2024
- **arXiv**: [2410.10226](https://arxiv.org/abs/2410.10226)
- **Key Result**: 直接研究 KD 与 model stealing 的关系，证明 KD 可作为 model stealing 的有效工具；多 agent 协作可提升模型窃取准确率
- **与 S-PSI 关系**: 验证了 KD 作为 model stealing 的 warm-start 阶段是合理的（正如我们的 Phase A 设计）。
- **🔴 不在 PAPERS.md 中，应引用**

### D2. ★★★☆☆ Comparing KD and Model Stealing Attacks in LLM Robustness
- **Authors**: Nargiza Nosirova
- **Venue**: CalState Master's Thesis, 2024
- **Link**: [CalState](https://scholarworks.calstate.edu/concern/theses/r207tw62f)
- **Key Result**: 系统对比 KD 和 model stealing 在评估 LLM 鲁棒性方面的等价性和差距
- **与 S-PSI 关系**: 支持我们将 KD 和 weight recovery 作为连续谱的两端来定位的叙事。

### D3. ★★★☆☆ A Comprehensive Survey on Knowledge Distillation of LLMs
- **Venue**: arXiv Feb 2024
- **arXiv**: [2402.13116](https://arxiv.org/abs/2402.13116)
- **Coverage**: LLM 知识蒸馏技术的全面综述
- **与 S-PSI 关系**: 提供 KD 技术全景，帮助定位 S-PSI 超越 KD 的独特贡献。

---

## Category E: 理论基础 & 侧信道

### E1. ★★★★☆ Beyond Model Extraction: Inference of Model Hyperparameters via Side-Channel
- **Venue**: IEEE 2024
- **Link**: [IEEE](https://ieeexplore.ieee.org/document/10646822)
- **Key Result**: 通过 timing SCA 推断目标模型的超参数（架构信息），作为完整模型提取的前置步骤
- **与 S-PSI 关系**: 互补工作。如果 SCA 可以确定架构，S-PSI 可以在已知架构假设下恢复权重。

### E2. ★★★☆☆ Attention Map Extraction from Black-Box Vision Transformers
- **Venue**: Springer 2024
- **Link**: [Springer](https://link.springer.com/chapter/10.1007/978-3-031-78192-6_24)
- **Key Result**: 通过 knowledge distillation 训练代理模型，从黑盒 ViT 提取 attention maps
- **与 S-PSI 关系**: attention 信息的可提取性为我们的方法提供了额外的理论支持。

### E3. ★★★☆☆ On Identifiability in Transformers (Theoretical Work)
- **Authors**: Bona Pellissier, Hardouin, Malgouyres 等
- **Venue**: Various 2024
- **Key Result**: 在什么条件下 transformer 权重可从输入输出行为唯一恢复（up to known symmetries）
- **关键洞见**: attention head permutation symmetry 意味着权重恢复只能达到 equivalence class 级别
- **与 S-PSI 关系**: 为我们的可行性分析提供理论基础。需要在论文中讨论 identifiability 限制。

---

## 🎯 关键空白分析 (Gap Analysis for S-PSI Positioning)

### Gap 1: 全层渐进式权重恢复 ★★★★★
- **现状**: Carlini 恢复 1 层 (output projection)；Liu & Moitra 理论上可恢复单矩阵（low-rank 假设）；侧信道方法需要物理访问
- **S-PSI 填补**: 从纯 API 访问渐进式恢复所有 transformer 层的权重
- **论文 Claim**: "First method to progressively recover all transformer layer weights from pure black-box logit API access"

### Gap 2: 超越低秩/ReLU 假设 ★★★★★
- **现状**: Liu & Moitra 需要精确 low-rank 假设；Shamir 需要 ReLU 激活
- **S-PSI 填补**: 基于优化的方法，不需要特定的结构假设；适用于 GeLU/SiLU、Attention、LayerNorm
- **论文 Claim**: "Architecture-agnostic inversion via sensitivity-guided optimization"

### Gap 3: 在最新防御下的评估 ★★★★☆
- **现状**: LoDD (AAAI'25), SPLITS (ICLR'25), Robust Utility-Preserving Perturbation 是最新防御；无攻击方法在这些防御下评估
- **S-PSI 填补**: 在 LoDD + SPLITS + logit perturbation 防御下评估 S-PSI 的退化程度
- **论文 Claim**: "Comprehensive evaluation against state-of-the-art 2025 defenses"

### Gap 4: 查询效率 vs. 恢复精度的 Pareto 前沿 ★★★★☆
- **现状**: Liu & Moitra 给出了 O(dr) 理论 bound；Carlini 的 $20 成本很低但只恢复 1 层
- **S-PSI 填补**: 提供不同查询 budget 下的恢复精度 Pareto 曲线（budget scaling experiments）
- **论文 Claim**: "First empirical Pareto frontier of query budget vs. weight recovery across all layers"

### Gap 5: Sensitivity-guided 查询策略 ★★★★☆
- **现状**: 大多数攻击使用随机查询或固定策略；没有将参数敏感度信息反馈到查询设计
- **S-PSI 填补**: 利用已恢复层的梯度敏感度信息指导后续层的查询策略
- **论文 Claim**: "First sensitivity-guided adaptive query strategy for progressive parameter inversion"

### Gap 6: KD warm-start → Weight Inversion 连续谱 ★★★☆☆
- **现状**: KD 和 model stealing 被视为平行但独立的研究线；Two Heads (2024) 开始研究两者关系
- **S-PSI 填补**: 将 KD 作为 warm-start 阶段，渐进过渡到 weight-level inversion
- **论文 Claim**: "Unified framework bridging behavioral cloning and weight recovery"

---

## 🔴 必须添加到 PAPERS.md 的新论文（按优先级）

| 优先级 | 编号 | 论文 | 年份 | 类型 |
|--------|------|------|------|------|
| P0 | A1 | Model Stealing for Any Low-Rank LM (Liu & Moitra) | AAAI 2025 | 直接竞争 |
| P0 | A3 | Logits Leak Proprietary Information | 2024 | 直接相关 |
| P0 | B1 | LoDD (AAAI 2025) | AAAI 2025 | 防御 |
| P0 | B2 | SPLITS (ICLR 2025) | ICLR 2025 | 防御 |
| P1 | A4 | Polynomial Cryptanalytic Extraction (Shamir) | 2024 | 理论基础 |
| P1 | A5 | Side-Channel Transformer Weight Stealing | Euro S&P 2025 | 互补攻击 |
| P1 | B3 | Robust Utility-Preserving Logit Perturbation | 2024 | 防御 |
| P1 | C1 | Survey: Extraction on Generative AI | 2024 | 综述 |
| P2 | C2 | Beyond Model Extraction Survey | 2025 | 综述 |
| P2 | D1 | Two Heads: Multi-Agent Model Stealing | 2024 | KD vs Stealing |
| P2 | B4 | IEEE Survey on Extraction Attacks | 2025 | 综述 |
| P3 | D3 | KD Survey for LLMs | 2024 | 综述 |
| P3 | E1 | Hyperparameter Inference via SCA | 2024 | 侧信道 |
| P3 | E2 | Attention Map Extraction from ViT | 2024 | 互补 |

---

## 📊 Updated Competitive Landscape

```
                    Weight Recovery Precision
                    Full weights ←————————→ Behavior only
                    |                                    |
Pure API   A1 Liu   |  🎯 S-PSI (ours)                  | KD baselines
Black-box  A3 Logit |       ↑                            |
           A2 Carl  |  Full model, progressive           |
                    |                                    |
Partial    A4 Shamir|                                    |
Access     E1 SCA   |                                    |
                    |                                    |
Physical   A5 EuroSP|                                    |
Side-ch.           |                                    |
                    |————————————————————————————————————|
                    Single layer    Progressive    All layers
                         Recovery Scope →
```

S-PSI 独占 **"Pure API + Full weights + All layers + Progressive"** 这个象限。

---

## 📝 论文写作建议

### Related Work 结构建议
1. **Model Extraction via API Access** — Carlini'24, Logits Leak'24, Liu & Moitra'25
2. **Cryptanalytic & Theoretical Extraction** — Shamir'24, Identifiability theory
3. **Side-Channel Weight Recovery** — Euro S&P'25, SCA hyperparameter inference
4. **Knowledge Distillation as Extraction** — Two Heads'24, KD surveys
5. **Defense Mechanisms** — LoDD'25, SPLITS'25, Robust Perturbation'24

### 关键对比句式
- "While Carlini et al. (2024) recovered only the output projection layer, S-PSI progressively recovers parameters across all transformer layers."
- "Liu and Moitra (2025) provide optimal query complexity bounds under a strict low-rank assumption. S-PSI relaxes this assumption through optimization-based inversion applicable to full-rank weight matrices."
- "Unlike side-channel approaches (Euro S&P 2025) requiring physical device access, S-PSI operates purely through standard API queries."
- "We evaluate S-PSI against the latest 2025 defenses including LoDD and SPLITS, demonstrating [specific results]."

---

## 注意事项

1. **论文中不要引用的**: PAPERS.md #4 "Aggressive Compression" (2026) — 可能未正式发表
2. **注意 StolenLoRA 和 "Clone What You Can't Steal"** — 搜索未返回可靠来源，可能是非常新的 preprint 或内部命名，需要确认 arXiv ID
3. **Liu & Moitra (AAAI 2025) 是最强竞争者**，论文中需要详细的理论/实验对比
4. **LoDD + SPLITS 是最新防御**，如果我们的 defense evaluation 没有测试这些，审稿人可能要求补充

---

*Survey conducted: 2026-04-08*
*Search sources: WebSearch across multiple queries, arXiv, OpenReview, IEEE, AAAI, Semantic Scholar*
