---
type: paper
node_id: paper:carlini2024_stealing
title: "Stealing Part of a Production Language Model"
authors: ["Nicholas Carlini", "Daniel Paleka", "Krishnamurthy Dvijotham", "Thomas Steinke", "Jonathan Hayase", "A. Feder Cooper", "Katherine Lee", "Matthew Jagielski", "Milad Nasr", "Arthur Conmy", "Eric Wallace", "David Rolnick", "Florian Tramer"]
year: 2024
venue: arXiv
external_ids:
  arxiv: "2403.06634"
  doi: null
  s2: null
tags: [model-stealing, LLM-security, cryptanalytic-extraction, embedding-recovery]
relevance: core
origin_skill: research-lit
created_at: 2026-04-09T01:50:00Z
updated_at: 2026-04-09T01:50:00Z
---

# Cryptanalytic extraction of embedding/unembedding matrices from production LLMs via carefully designed queries.

## Problem / Gap

Production LLMs expose logit-level outputs; can we extract actual model parameters (not just behavior)?

## Method

Exploit the low-rank structure of the output projection (unembedding) layer. Design queries that isolate individual dimensions of the hidden state, enabling recovery of the full embedding/unembedding matrix up to a linear transformation.

## Key Results

- Recovered the full embedding matrix of OpenAI's production models
- Cost: ~$2000 in API queries
- Confirmed that logit access is sufficient for parameter extraction

## Assumptions

- Full logit access (not just top-k)
- Known vocabulary and tokenizer
- Model uses standard softmax output

## Limitations / Failure Modes

- Only recovers the last layer (embedding/unembedding), not internal transformer blocks
- Assumes the embedding is not rotated or obfuscated
- Does not address intermediate layers at all

## Reusable Ingredients

- Query design strategy for isolating hidden dimensions
- The insight that logit access leaks structural information about parameters

## Open Questions

- Can this approach be extended deeper into the network?
- How does logit quantization affect recovery?

## Claims

claim:C1 — Logit access is sufficient to extract the unembedding matrix of production LLMs

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

Foundation for our work. We extend beyond last-layer extraction to suffix blocks (multiple transformer layers) using progressive inversion with observability theory.
