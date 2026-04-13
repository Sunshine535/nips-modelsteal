---
type: paper
node_id: paper:rawal2025_spsi
title: "S-PSI: Sensitivity-Guided Progressive Suffix Inversion for LLM Weight Recovery"
authors: ["Aayush Rawal"]
year: 2025
venue: unpublished
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: [model-stealing, parameter-inversion, progressive-recovery, sensitivity-matching]
relevance: core
origin_skill: codebase-analysis
created_at: 2026-04-09T01:50:00Z
updated_at: 2026-04-09T01:50:00Z
---

# Progressive recovery of transformer suffix blocks from black-box logit access using sensitivity matching.

## Problem / Gap

Carlini2024 only recovers the embedding layer. Can we go deeper — recovering multiple transformer blocks from the output side inward?

## Method

S-PSI (Sensitivity-Guided Progressive Suffix Inversion):
1. Fix the known embedding (lm_head) layer
2. Progressively recover suffix blocks from output toward input
3. Loss: L = α·L_logit + β·L_sensitivity + γ·L_reg
4. Sensitivity matching: compare how teacher/student outputs change under token perturbations
5. Multiple random initializations; pick best by final loss

## Key Results

- Recovers last 2 suffix blocks with high cosine similarity in oracle regime
- Pure-logits regime is harder but partially works
- Query budget ~500K for Qwen3.5-0.8B

## Assumptions

- Same architecture teacher/student (Qwen3.5-0.8B)
- Full logit access
- lm_head is known (tied embeddings)

## Limitations / Failure Modes

- Random initialization — no principled way to start
- No theory for why some blocks are harder
- Sensitivity matching is heuristic, not derived from any optimality principle
- Symmetries not explicitly handled

## Reusable Ingredients

- Progressive suffix strategy (outer blocks first)
- Sensitivity matching loss term
- Block-by-block recovery with checkpoint/resume
- Teacher caching for query efficiency

## Open Questions

- Can algebraic initialization replace random init?
- Does the Gramian eigenspectrum predict block difficulty?
- Can queries be optimized rather than random?

## Claims

claim:C2 — Progressive suffix inversion can recover multiple transformer blocks from logit access

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

This IS our starting codebase. Transformer Tomography (idea:001) builds on S-PSI by adding observability theory, algebraic init, gauge projection, and Fisher-optimal queries.
