# Related Work — Model Stealing & Parameter Extraction

## Direct Competitors

### 1. Carlini et al. — "Stealing Part of a Production Language Model"
- **Venue**: ICML 2024
- **Key Result**: Extracted the embedding projection layer of production OpenAI models for <$20 using API queries
- **Method**: Algebraic attack exploiting the structure of the final projection layer; recovers the output embedding matrix exactly
- **Limitation**: Only extracts the OUTPUT layer; no recovery of internal transformer layers
- **Relevance**: ★★★★★ — Most direct prior work. We extend beyond output layer to full progressive recovery.
- **URL**: https://arxiv.org/abs/2403.06634

### 2. "Clone What You Can't Steal" — SVD-Based Behavioral Extraction
- **Venue**: arXiv 2025
- **Key Result**: 97.6% hidden geometry recovery via SVD-based logit leakage + distillation
- **Method**: Exploit logit distributions to infer hidden layer geometry (not weights); combine with knowledge distillation
- **Limitation**: Recovers *behavioral clone* and geometric structure, NOT actual weight values
- **Relevance**: ★★★★★ — Claims geometry recovery but admits weights themselves are not recovered. We aim for actual weight matrices.

### 3. StolenLoRA — LoRA Adapter Extraction
- **Venue**: 2025
- **Key Result**: 96.6% LoRA adapter extraction fidelity with 10K queries
- **Method**: Query a LoRA-adapted model, recover the low-rank adaptation matrices via gradient matching
- **Limitation**: Only targets LoRA adapters (low-rank, small parameter count); does not attempt base model extraction
- **Relevance**: ★★★★☆ — Demonstrates feasibility of weight recovery for low-rank components. We attempt full-rank layer recovery.

### 4. "Aggressive Compression Enables Weight Theft"
- **Venue**: 2026
- **Key Result**: 16-100x compression enables exfiltration of model weights
- **Method**: Compress model weights to tiny payloads for covert extraction
- **Limitation**: ASSUMES the attacker already HAS access to the raw weights; solves exfiltration, not extraction
- **Relevance**: ★★★☆☆ — Complementary threat model; we address the harder problem of extraction from API access only.

### 5. "Beyond Slow Signs" — Efficient Cryptanalytic Extraction
- **Venue**: NeurIPS 2024
- **Key Result**: 14.8x extraction efficiency improvement over prior sign-based methods
- **Method**: Improved cryptanalytic techniques for extracting weight signs from neural network API access
- **Limitation**: Recovers sign bits only, not full-precision floating-point weights
- **Relevance**: ★★★★☆ — Advances extraction efficiency but limited to sign recovery. We target full-precision weight matrices.

### 6. Theoretical Bounds on Attention Parameter Extraction
- **Venue**: Various theoretical works 2024–2025
- **Key Result**: O(d²) queries sufficient to recover single-head attention parameters
- **Method**: Algebraic analysis of attention mechanism as polynomial in input
- **Limitation**: Theory only; single attention head; doesn't scale to multi-head multi-layer transformers
- **Relevance**: ★★★☆☆ — Provides theoretical grounding for feasibility; our empirical method complements.

## Knowledge Distillation (Behavioral Cloning Baselines)

### 7. Hinton et al. — "Distilling the Knowledge in a Neural Network"
- **Venue**: NeurIPS Workshop 2015
- **Key Result**: Foundational work on knowledge distillation via soft targets
- **Method**: Train student to match teacher's softmax distribution at temperature T
- **Relevance**: ★★★☆☆ — Our Stage 1 baseline; behavioral cloning without weight recovery.

### 8. Gou et al. — "Knowledge Distillation: A Survey"
- **Venue**: IJCV 2021
- **Key Result**: Comprehensive taxonomy of KD methods (logit, feature, relation-based)
- **Relevance**: ★★★☆☆ — Context for our distillation warm-start stage.

### 9. DistilBERT / TinyBERT / MiniLM
- **Venue**: Various 2019–2020
- **Key Result**: Efficient transformer distillation with layer mapping
- **Method**: Layer-wise distillation matching intermediate representations
- **Relevance**: ★★☆☆☆ — Layer-wise matching parallels our approach, but targets behavior not weight recovery.

## Model Extraction Attacks (Broader)

### 10. Tramèr et al. — "Stealing Machine Learning Models via Prediction APIs"
- **Venue**: USENIX Security 2016
- **Key Result**: Foundational model extraction for logistic regression, decision trees, neural networks
- **Method**: Equation-solving attacks using prediction API queries
- **Relevance**: ★★★☆☆ — Seminal work but targets simple models; LLM extraction is fundamentally harder.

### 11. Jagielski et al. — "High Accuracy and High Fidelity Extraction of Neural Networks"
- **Venue**: USENIX Security 2020
- **Key Result**: Learning-based extraction with fidelity guarantees
- **Method**: Fidelity extraction maximizing agreement with teacher on specific inputs
- **Relevance**: ★★★☆☆ — Fidelity-focused extraction; we extend from behavior to weight-level fidelity.

### 12. Krishna et al. — "Thieves on Sesame Street"
- **Venue**: ICLR 2020
- **Key Result**: BERT model extraction via fine-tuning on query results
- **Method**: Query BERT API, fine-tune local copy on responses
- **Relevance**: ★★☆☆☆ — Behavioral extraction of NLU models; predates LLM era.

## Defense Methods

### 13. Dziedzic et al. — "On the Difficulty of Defending Self-Supervised Learning against Model Extraction"
- **Venue**: ICML 2022
- **Key Result**: Defenses against model extraction are fundamentally difficult for SSL models
- **Relevance**: ★★☆☆☆ — Informs our defense evaluation design.

### 14. Watermarking for LLMs (Kirchenbauer et al.)
- **Venue**: ICML 2023
- **Key Result**: Statistical watermarking of LLM outputs via soft red/green token lists
- **Method**: Bias token generation toward "green" tokens detectable statistically
- **Relevance**: ★★★☆☆ — One of our defense baselines; does watermarking prevent parameter inversion?

### 15. Output Perturbation Defenses
- **Venue**: Various 2020–2024
- **Key Result**: Adding noise to API outputs degrades extraction quality
- **Method**: Calibrated noise, logit rounding, temperature randomization
- **Relevance**: ★★★★☆ — Primary defense mechanism we evaluate against PLPI.

## Gradient-Based Inversion (Parallel Domain)

### 16. Zhu et al. — "Deep Leakage from Gradients"
- **Venue**: NeurIPS 2019
- **Key Result**: Recover training data from shared gradients in federated learning
- **Method**: Gradient-matching optimization to reconstruct inputs
- **Relevance**: ★★★☆☆ — Parallel technique (gradient inversion); we invert weights from outputs rather than inputs from gradients.

### 17. Geiping et al. — "Inverting Gradients — How Easy Is It to Break Privacy in Federated Learning?"
- **Venue**: NeurIPS 2020
- **Key Result**: Improved gradient inversion with cosine similarity loss
- **Relevance**: ★★★☆☆ — Optimization techniques applicable to our weight inversion objective.

## Summary Table

| # | Paper | Year | Extracts | Full Weights? | Black-Box? |
|---|-------|------|----------|--------------|------------|
| 1 | Carlini et al. | 2024 | Output projection | Partial (1 layer) | ✓ |
| 2 | Clone What You Can't Steal | 2025 | Hidden geometry | ✗ (behavioral) | ✓ |
| 3 | StolenLoRA | 2025 | LoRA adapters | Partial (adapters) | ✓ |
| 4 | Aggressive Compression | 2026 | Full model | ✓ (but needs access) | ✗ |
| 5 | Beyond Slow Signs | 2024 | Weight signs | Partial (signs only) | ✓ |
| 6 | Theoretical bounds | 2024 | Attention params | Theory only | ✓ |
| **Ours** | **PLPI** | **2026** | **All layers progressively** | **✓ (progressive)** | **✓** |
