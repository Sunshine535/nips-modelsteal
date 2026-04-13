---
type: paper
node_id: paper:tramer2016_stealing
title: "Stealing Machine Learning Models via Prediction APIs"
authors: ["Florian Tramer", "Fan Zhang", "Ari Juels", "Michael K. Reiter", "Thomas Ristenpart"]
year: 2016
venue: USENIX Security
external_ids:
  arxiv: "1609.02943"
  doi: null
  s2: null
tags: [model-stealing, prediction-API, equation-solving, foundational]
relevance: core
origin_skill: research-lit
created_at: 2026-04-09T01:50:00Z
updated_at: 2026-04-09T01:50:00Z
---

# Foundational work showing ML model parameters can be stolen via prediction API queries using equation-solving attacks.

## Problem / Gap

ML-as-a-service exposes prediction APIs. Can an adversary reconstruct the model?

## Method

For linear models and shallow networks: treat each query-response pair as an equation in the model parameters. Solve the system directly or via optimization.

## Key Results

- Linear models: exact recovery with d+1 queries
- Logistic regression, decision trees, shallow NNs: practical extraction
- Demonstrated on BigML, Amazon ML

## Assumptions

- Full prediction output (class probabilities, not just labels)
- Known model family

## Limitations / Failure Modes

- Does not scale to deep networks
- Requires known architecture
- No theory for transformer-specific symmetries

## Reusable Ingredients

- Equation-solving formulation of model stealing
- Query complexity analysis framework

## Open Questions

- How to extend to deep transformers?
- What is the fundamental query complexity for LLM extraction?

## Claims

claim:C3 — Linear models can be exactly recovered with O(d) queries

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Foundational reference. Our Gramian-based framework is the transformer analog of Tramer's equation-solving approach for linear models.
