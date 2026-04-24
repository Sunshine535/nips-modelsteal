#!/usr/bin/env python3
# SAFETY NOTICE: QUARANTINED (alpha-theory prune 2026-04-19)
# This script is NOT cited in the paper. It was part of killed branches:
#   A1 S-PSI, A2 Moments CP, A4 logit-bias, A5 memory probing, A6 active query,
#   A7 algebraic v2/v3/v4, B3 matched-KD.
# Retained in repo for reproducibility of quarantined history; do not use for
# new claims.
#!/usr/bin/env python3
"""
Memory / Selective-Probing Attack — Proof-of-concept.

Goal
----
The residual stream of a decoder LM is empirically low-rank: natural-text
queries only excite ~9 independent directions in the (d=896)-dim hidden
space of Qwen2.5-0.5B, and Carlini's SVD of black-box logits collapses to
rank 9 after a handful of thousands of queries.  That low-rank barrier is
the reason pure-logit SVD recovers only the TIED `lm_head` subspace and
no internal block parameters: there simply isn't enough ACTIVATION
diversity in the last hidden state for outer-product attacks to separate
circuit contributions.

This script attempts to BEAT the low-rank barrier by replacing the
"natural WikiText" probe distribution with FIVE deliberately engineered
probe FAMILIES, each designed to excite a different narrow slice of the
model's activation manifold:

    A  Copy / induction heads      "The cat sat. The cat sat."
    B  Syntactic minimal pairs      "The boy eat|eats apples"
    C  Memorized vs perturbed       verbatim WikiText vs one-word-swapped
    D  Positional variation         same content at different positions
    E  Attention-head isolation     prompts targeting the ~2 KV heads

For each family we run Carlini's SVD restricted to that family to read off
a per-family effective rank, and we run the SVD on the UNION of all
families to read off the combined effective rank.  The headline number is
whether the combined effective rank exceeds ~9 (the passive-text baseline
on this model).

In addition to rank, we compute:
  * Per-family logit entropy (Family C: memorized should be lower)
  * Between-family separation (angle between family mean-subspaces)
  * A modest parameter-recovery attempt via per-family Carlini plus
    Hungarian matching against the last block's W_O / W_down columns

Both outcomes are paper-worthy:
  * If combined rank >> 9 → NOVEL FINDING: the low-rank barrier is an
    artifact of probe distribution, and active probing expands the
    observable subspace.
  * If combined rank ≈ 9 → CONFIRMS the low-rank barrier is intrinsic to
    the architecture and strengthens the paper's robustness claim.

Threat model
------------
Pure black-box: only `model(input_ids)` and the logits it returns.  No
logprobs API tricks, no logit_bias, no hidden state access.

Runtime
-------
Default args (queries_per_family=500, seq_len=32) complete in ~1h on a
single A100-80GB for Qwen2.5-0.5B (bfloat16).  Dry-run mode finishes in
~60 s for CI.

Usage
-----
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 python scripts/attack_memory_inversion.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --queries_per_family 500 \
    --output_dir results/v5_attack_memory \
    --seed 42
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import string
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Make repo root importable (for src/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ── Model architecture constants (Qwen2.5-0.5B) ─────────────────────────────
NUM_LAYERS = 24
LAST_BLOCK_IDX = 23
D_MODEL = 896
D_FF = 4864
VOCAB_SIZE = 151936
NUM_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64

# Passive-text baseline effective rank for Qwen2.5-0.5B last-position logits.
# Measured in prior Carlini reproductions on WikiText; used only as a
# point-of-comparison target.
PASSIVE_BASELINE_EFF_RANK = 9

# Family labels (single letters) and their human-readable names.
FAMILY_LABELS = ["A", "B", "C_mem", "C_perturb", "D", "E"]
FAMILY_DESCRIPTIONS: dict[str, str] = {
    "A":         "induction/copy-head probes",
    "B":         "syntactic minimal pairs",
    "C_mem":     "verbatim WikiText (potentially memorized)",
    "C_perturb": "WikiText with one-word perturbation (control for C_mem)",
    "D":         "same content at varied positions",
    "E":         "attention-head isolation prompts",
}


# ════════════════════════════════════════════════════════════════════════════
#  Setup helpers
# ════════════════════════════════════════════════════════════════════════════


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Memory / Selective-Probing Attack (POC)"
    )
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--queries_per_family", type=int, default=500,
                   help="Probes per family (A/B/C_mem/C_perturb/D/E).")
    p.add_argument("--max_seq_len", type=int, default=32,
                   help="Token sequence length for each probe.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output_dir", type=str,
                   default="results/v5_attack_memory")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eff_rank_threshold", type=float, default=1e-3,
                   help="Relative threshold (singular_i / sigma_0) for counting"
                        " a direction toward effective rank.")
    p.add_argument("--success_eff_rank", type=int, default=50,
                   help="Combined-families eff_rank strictly greater than this"
                        " counts as SUCCESS (vs ~9 for passive WikiText).")
    p.add_argument("--success_param_cos", type=float, default=0.1,
                   help="Aligned-cos threshold for the parameter-recovery probe.")
    p.add_argument("--param_match_rows", type=int, default=256,
                   help="Cap on teacher columns used in Hungarian matching.")
    p.add_argument("--dry_run_small", action="store_true",
                   help="Run with tiny sizes for smoke testing.")
    p.add_argument("--allow_synthetic_wiki", action="store_true", default=True,
                   help="Fall back to a small in-script WikiText-like pool "
                        "when HF datasets are unavailable (HF_HUB_OFFLINE=1).")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def human_bytes(n: float) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def flat_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.double().flatten()
    b_f = b.double().flatten()
    denom = (a_f.norm() * b_f.norm()).clamp(min=1e-30)
    return float((a_f @ b_f) / denom)


# ════════════════════════════════════════════════════════════════════════════
#  Query-family generators
# ════════════════════════════════════════════════════════════════════════════
#
# Each generator returns a list of raw strings.  All strings are later tokenised
# and padded/truncated to `max_seq_len`.  We also enforce the ID rule that
# each family produces exactly `n` distinct strings (with retries) so the
# per-family SVD is well-defined.
# ----------------------------------------------------------------------------


def _fallback_wikitext_corpus() -> list[str]:
    """Small in-script WikiText-like corpus used when HF datasets are offline.

    These are short encyclopaedic-style sentences.  They are not real
    WikiText-103; they are just the only bank available when the loader
    cannot reach the mirror.  For the memorization / perturbation family
    they still provide the "real-string vs one-word-swapped" contrast that
    Phase 4 needs.
    """
    return [
        "The Second World War lasted from 1939 to 1945 and involved most of the world's nations.",
        "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
        "The speed of light in a vacuum is approximately 299,792,458 metres per second.",
        "William Shakespeare wrote 39 plays including Hamlet, Macbeth, and Othello.",
        "The Great Wall of China stretches over 13,000 miles across northern China.",
        "Mount Everest is the highest mountain above sea level at 8,848 metres.",
        "The Pacific Ocean is the largest and deepest of the world's five oceans.",
        "DNA is a double helix composed of four nucleotide bases: adenine, thymine, guanine, and cytosine.",
        "The Renaissance was a cultural movement that began in Italy in the 14th century.",
        "Albert Einstein published his theory of special relativity in 1905.",
        "The Amazon Rainforest covers much of northwestern Brazil and extends into nine countries.",
        "The Industrial Revolution began in Britain in the late 18th century.",
        "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
        "The Eiffel Tower was completed in 1889 and stands 324 metres tall in Paris.",
        "The French Revolution began in 1789 with the storming of the Bastille.",
        "Antarctica is the fifth-largest continent and contains 90 percent of the world's ice.",
        "The Roman Empire reached its greatest territorial extent under the emperor Trajan.",
        "The periodic table organizes chemical elements by increasing atomic number.",
        "Quantum mechanics describes the behaviour of matter and energy at the atomic scale.",
        "Leonardo da Vinci painted the Mona Lisa during the early 16th century.",
        "The Internet originated from a United States military project called ARPANET.",
        "The Nile River is the longest river in the world at over 6,650 kilometres.",
        "The Cold War was a period of geopolitical tension between the United States and the Soviet Union.",
        "Charles Darwin proposed the theory of evolution by natural selection in his 1859 book.",
        "The Milky Way is a barred spiral galaxy containing between 200 and 400 billion stars.",
        "The Berlin Wall was erected in 1961 and torn down in 1989.",
        "Penicillin was discovered by Alexander Fleming in 1928.",
        "The Silk Road was an ancient network of trade routes connecting East and West.",
        "The Roman Colosseum could hold approximately 80,000 spectators at its peak.",
        "The Wright Brothers achieved the first powered flight on December 17, 1903.",
        "The Great Pyramid of Giza was built around 2560 BC as a tomb for pharaoh Khufu.",
        "The Black Death killed between 30 and 60 percent of Europe's population in the 14th century.",
        "The Hubble Space Telescope was launched into orbit by the space shuttle Discovery in 1990.",
        "The Declaration of Independence was adopted by the Second Continental Congress on July 4, 1776.",
        "The Russian Revolution overthrew the Russian Empire in 1917.",
        "The Magna Carta was sealed by King John of England in June 1215.",
        "Penicillium mould produces the antibiotic penicillin as a defense against bacteria.",
        "The Taj Mahal is an ivory-white marble mausoleum built by emperor Shah Jahan.",
        "The Beatles were an English rock band formed in Liverpool in 1960.",
        "The theory of continental drift was proposed by Alfred Wegener in 1912.",
        "The Statue of Liberty was a gift from France to the United States in 1886.",
        "The Apollo 11 mission landed the first humans on the Moon in July 1969.",
        "Napoleon Bonaparte was defeated at the Battle of Waterloo in June 1815.",
        "Coral reefs support about a quarter of all marine species despite covering less than one percent of the ocean.",
        "The Titanic sank on its maiden voyage in April 1912 after colliding with an iceberg.",
        "Vaccines work by training the immune system to recognize specific pathogens.",
        "The Panama Canal connects the Atlantic and Pacific Oceans through Central America.",
        "The Grand Canyon was carved by the Colorado River over millions of years.",
        "The heart pumps approximately 2,000 gallons of blood through the body every day.",
        "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
    ]


def _expand_corpus_with_prefixes(
    base: list[str], n_desired: int, seed: int,
) -> list[str]:
    """Deterministically expand a short base corpus by prepending short,
    semantically-neutral prefixes.

    Each (prefix, base_sentence) pair is a DISTINCT token sequence, so from
    a base of 50 sentences and 20 prefix options we can produce up to 1000
    distinct probes.  This is still ~real-text distribution (prefixes come
    from the "factual encyclopaedic" style) so it stays a reasonable stand-in
    for WikiText when offline.
    """
    prefixes = [
        "",
        "According to recent estimates,",
        "As discussed in scholarly articles,",
        "According to the most reliable sources,",
        "Historical records indicate that",
        "Researchers generally agree that",
        "It is widely accepted that",
        "Scientific literature confirms that",
        "Contemporary accounts note that",
        "Standard reference works state that",
        "Most authorities agree that",
        "It has often been observed that",
        "Primary sources confirm that",
        "Modern analyses suggest that",
        "Established scholarship shows that",
        "According to the encyclopedia,",
        "A common account holds that",
        "Widely available records show that",
        "The consensus view is that",
        "As noted in textbooks,",
    ]
    rng = random.Random(seed + 1777)
    out: list[str] = []
    seen: set[str] = set()
    # First pass: use the base sentences verbatim.
    for b in base:
        if b in seen:
            continue
        seen.add(b)
        out.append(b)
        if len(out) >= n_desired:
            return out[:n_desired]

    # Subsequent passes: cycle (prefix, base) pairs in a shuffled order.
    pairs = [(p, b) for p in prefixes if p for b in base]
    rng.shuffle(pairs)
    for p, b in pairs:
        s = f"{p} {b[:1].lower()}{b[1:]}" if p else b
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= n_desired:
            break
    return out[:n_desired]


def _load_wikitext_strings(
    n_desired: int,
    allow_synthetic: bool,
    seed: int = 42,
) -> list[str]:
    """Prefer real WikiText, fall back to the in-script corpus.

    When the real dataset is unavailable (HF_HUB_OFFLINE=1) we draw from an
    in-script encyclopaedic corpus.  If that corpus has fewer entries than
    requested, we deterministically expand it via prefix prepending so that
    downstream code always sees `n_desired` distinct strings.
    """
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    strings: list[str] = []
    if not offline:
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "wikitext", "wikitext-103-raw-v1", split="validation",
            )
            for ex in ds:
                text = ex.get("text", "").strip()
                # Keep only reasonably sized sentences
                if 40 <= len(text) <= 400 and text.endswith((".", "!", "?")):
                    strings.append(text)
                if len(strings) >= n_desired * 3:
                    break
        except Exception as e:
            logger.warning("wikitext load failed (%s) — using fallback corpus.", e)
    if len(strings) >= n_desired:
        return strings[:n_desired]

    # Either offline, or wikitext was too small — fall back to synthetic.
    if not allow_synthetic:
        raise RuntimeError(
            f"wikitext returned only {len(strings)} strings "
            f"and --allow_synthetic_wiki=False."
        )
    base = strings + _fallback_wikitext_corpus()
    return _expand_corpus_with_prefixes(base, n_desired, seed)


# ---- Family A: induction / copy-head probes -------------------------------

def generate_family_A(n: int, seed: int) -> list[str]:
    """Induction/copy-head probes.

    Induction heads are triggered by REPEATED token patterns: the model,
    on seeing "X Y ... X", tends to predict Y.  We build prompts that
    deliberately include such repetitions so the last hidden state is
    DOMINATED by the copy head's output.
    """
    rng = random.Random(seed)
    # Small vocabulary of simple tokens to build repetition structures.
    nouns = ["cat", "dog", "bird", "tree", "river", "hill", "star", "book",
             "lamp", "train", "cloud", "stone", "plate", "song", "door",
             "flame", "glass", "seed", "rope", "wave", "mountain", "desert",
             "forest", "meadow", "valley", "bridge", "castle", "market",
             "temple", "garden", "harbour", "tower", "island", "village",
             "glacier", "ocean", "canyon", "spring", "creek", "pond"]
    verbs = ["sat", "ran", "flew", "jumped", "sang", "fell", "rose", "glowed",
             "shone", "swam", "danced", "roared", "whispered", "trembled",
             "waited", "slept", "woke", "moved", "stayed", "passed"]
    places = ["on the mat", "in the yard", "by the lake", "under the tree",
              "near the hill", "over the bridge", "across the road",
              "beside the fire", "against the wall", "within the field",
              "through the meadow", "between the pillars"]

    out: list[str] = []
    attempts = 0
    # Four repetition structures; cycle through them.
    while len(out) < n and attempts < n * 20:
        attempts += 1
        choice = attempts % 4
        noun = rng.choice(nouns)
        verb = rng.choice(verbs)
        place = rng.choice(places)
        other_noun = rng.choice(nouns)
        other_verb = rng.choice(verbs)
        if choice == 0:
            # AB AB A→(B)
            s = f"The {noun} {verb} {place}. The {noun} {verb} {place}. The {noun}"
        elif choice == 1:
            # AXYZ.A X Y → Z
            s = (f"The {noun} {verb} {place}, and the {other_noun} "
                 f"{other_verb}. The {noun} {verb} {place}, and the "
                 f"{other_noun}")
        elif choice == 2:
            # token-level copy
            s = (f"{noun} {verb} {noun} {verb} {noun} {verb} {noun}")
        else:
            # list completion
            s = (f"List: {noun}, {other_noun}, {noun}, {other_noun}, {noun},")
        if s in out:
            continue
        out.append(s)
    if len(out) < n:
        raise RuntimeError(
            f"Family A: only generated {len(out)} distinct strings, "
            f"need {n}. Increase vocabulary."
        )
    return out[:n]


# ---- Family B: syntactic minimal pairs -----------------------------------

def generate_family_B(n: int, seed: int) -> list[str]:
    """Syntactic specificity probes.

    Minimal pairs differing only in subject-verb agreement, determiner choice
    or tense.  These probe syntax-selective heads without varying semantic
    content.  We emit BOTH members of each pair so the SVD sees the full
    syntactic manifold; the perturbation that flips agreement is at a
    fixed position so positional encoding is NOT the varying factor.
    """
    rng = random.Random(seed + 1)
    subjects_sing = ["The boy", "My sister", "A teacher", "This dog",
                     "Her friend", "An engineer", "The child", "One runner",
                     "The baker", "His mother", "My uncle", "The farmer",
                     "A student", "The driver", "Her doctor", "A painter"]
    subjects_plur = ["The boys", "My sisters", "Some teachers", "These dogs",
                     "Her friends", "Two engineers", "The children", "Many runners",
                     "The bakers", "His brothers", "My uncles", "The farmers",
                     "Some students", "The drivers", "Her doctors", "Three painters"]
    verb_pairs = [("eats", "eat"), ("runs", "run"), ("writes", "write"),
                  ("walks", "walk"), ("reads", "read"), ("drives", "drive"),
                  ("cooks", "cook"), ("sings", "sing"), ("paints", "paint"),
                  ("builds", "build"), ("studies", "study"), ("watches", "watch")]
    objects = ["the bread", "a letter", "the morning news", "an apple",
               "the garden", "those flowers", "the blue car", "a song",
               "the wooden door", "an old book", "the tall tree",
               "a yellow fruit", "the long road", "their dinner",
               "her grandmother", "the red ball", "a new chair",
               "the small garden"]

    out: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        choice = attempts % 2
        v_sing, v_plur = rng.choice(verb_pairs)
        obj = rng.choice(objects)
        if choice == 0:
            subj = rng.choice(subjects_sing)
            s = f"{subj} {v_sing} {obj} every day, and then"
        else:
            subj = rng.choice(subjects_plur)
            s = f"{subj} {v_plur} {obj} every day, and then"
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    if len(out) < n:
        raise RuntimeError(
            f"Family B: only generated {len(out)} distinct strings, "
            f"need {n}."
        )
    return out[:n]


# ---- Family C: memorized + perturbed pairs --------------------------------

def generate_family_C(
    n_each: int, seed: int, allow_synthetic: bool,
) -> tuple[list[str], list[str]]:
    """Verbatim WikiText (likely memorized) + one-word perturbation.

    We take the first `n_each` qualifying sentences from WikiText and emit
    them as-is for C_mem.  For C_perturb we swap ONE content word in each
    sentence with a random synonym/substitute; this is designed to keep
    the sentence grammatical but LEAVE the memorized sequence.
    """
    rng = random.Random(seed + 2)
    substitutes = {
        "is":       ["was", "became", "remains"],
        "are":      ["were", "remained"],
        "first":    ["final", "earliest", "newest"],
        "largest":  ["smallest", "oldest", "widest"],
        "world":    ["planet", "continent", "country"],
        "war":      ["battle", "campaign", "conflict"],
        "city":     ["town", "village", "capital"],
        "river":    ["stream", "creek", "lake"],
        "century":  ["decade", "millennium", "era"],
        "people":   ["citizens", "residents", "inhabitants"],
        "north":    ["south", "east", "west"],
        "and":      ["plus", "along with"],
        "the":      ["this", "that", "a"],
        "of":       ["from", "within"],
        "in":       ["across", "throughout"],
    }
    strings = _load_wikitext_strings(
        n_desired=n_each, allow_synthetic=allow_synthetic, seed=seed,
    )
    if len(strings) < n_each:
        raise RuntimeError(
            f"Family C: only loaded {len(strings)} sentences, "
            f"need {n_each}."
        )
    strings = strings[:n_each]

    perturbed: list[str] = []
    for s in strings:
        words = s.split(" ")
        candidate_idx = [i for i, w in enumerate(words)
                         if w.lower().strip(string.punctuation) in substitutes]
        if not candidate_idx:
            # no substitutable word → perturb by appending an innocuous
            # prefix; the sentence content is now not verbatim
            perturbed.append("Maybe " + s)
            continue
        idx = rng.choice(candidate_idx)
        raw = words[idx]
        key = raw.lower().strip(string.punctuation)
        new = rng.choice(substitutes[key])
        # Preserve capitalisation + trailing punctuation
        if raw[:1].isupper():
            new = new.capitalize()
        trailing = ""
        stripped = raw.rstrip(string.punctuation)
        if len(stripped) != len(raw):
            trailing = raw[len(stripped):]
        words[idx] = new + trailing
        perturbed.append(" ".join(words))
    return strings, perturbed


# ---- Family D: positional variation ---------------------------------------

def generate_family_D(
    n: int, seed: int, max_seq_len: int,
) -> list[str]:
    """Same content at varied positions.

    We take a pool of CONTENT sentences and prepend variable-length "filler"
    prefixes (numbers, short phrases) so the same content appears at many
    different sequence positions.  This probes positional encoding without
    changing the semantic payload.  We compose (filler1, filler2, content)
    triples so the effective variant count is |fillers|^2 * |contents|.
    """
    rng = random.Random(seed + 3)
    contents = [
        "the key lay on the wooden desk",
        "three red birds flew over the house",
        "an old man walked along the river",
        "the clock struck twelve exactly",
        "a green bottle rolled across the floor",
        "two children ran toward the gate",
        "the window shone in the morning light",
        "a heavy book fell from the shelf",
        "the small dog barked at the stranger",
        "a pale moon rose above the hills",
        "the thick fog covered the valley",
        "a bright flower bloomed in the garden",
        "the captain steered the ship carefully",
        "a quiet song drifted through the hall",
        "the tall tree swayed in the wind",
        "a young rider passed by the inn",
        "the stone archway framed the view",
        "a white horse grazed near the stream",
        "the warm breeze stirred the curtains",
        "a narrow path wound up the hillside",
    ]
    fillers_1 = [
        "", "Note one:", "In brief,", "For example,", "Historically,",
        "Indeed,", "Consider this:", "Observed this morning:",
        "Number eleven on the list:", "Listed below:",
        "One summer evening,", "Long ago in a distant village,",
        "After a long journey through the hills,",
        "During the last meeting of the council,",
        "Following the early reports of the day,",
        "At some point during the afternoon,",
        "As recorded in the ledger,",
        "Per the instructions of the mayor,",
    ]
    fillers_2 = [
        "", "as observed,", "curiously,", "perhaps notably,",
        "by most accounts,", "in the local tradition,", "once again,",
        "for reasons now forgotten,", "without much fanfare,",
        "with the usual care,",
    ]
    out: list[str] = []
    seen: set[str] = set()
    # Attempts allows the dedup loop enough headroom that even with collisions
    # we reach `n`.  With 20 * 18 * 10 = 3600 distinct (filler1, filler2,
    # content) triples, n = 500 is well within capacity.
    attempts = 0
    max_attempts = max(n * 50, 20000)
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        f1 = rng.choice(fillers_1)
        f2 = rng.choice(fillers_2)
        content = rng.choice(contents)
        parts = [p for p in [f1, f2] if p]
        if parts:
            prefix = " ".join(parts)
            s = f"{prefix} {content}."
        else:
            s = content + "."
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    if len(out) < n:
        raise RuntimeError(
            f"Family D: only generated {len(out)} distinct strings "
            f"after {attempts} attempts."
        )
    return out[:n]


# ---- Family E: attention-head-isolation probes ----------------------------

def generate_family_E(n: int, seed: int) -> list[str]:
    """Prompts designed to minimise activation in all but a narrow slice
    of heads.

    Qwen2.5-0.5B uses GQA with 2 KV heads shared across 14 query heads.  We
    cannot truly isolate a single head without white-box access, but we can
    construct prompts whose STRUCTURE is dominated by one of two patterns:

        (e1) Pure lookup: "X: Y\\nX:" → triggers induction-like direct copy.
        (e2) Coreference chain: long subject-name antecedent chains,
             triggering reference-tracking heads.

    Combined, these two family members span a narrow slice of head activity
    that (by construction) differs from Families A-D.
    """
    rng = random.Random(seed + 4)
    names = ["Alice", "Bob", "Carol", "Dave", "Emma", "Frank", "Gina",
             "Henry", "Iris", "Jack", "Kate", "Luis", "Mia", "Noah", "Olga",
             "Paul", "Quinn", "Rhea", "Sam", "Tara", "Uma", "Victor",
             "Wendy", "Xavier"]
    values = ["red", "blue", "green", "yellow", "black", "white", "grey",
              "tall", "short", "heavy", "light", "quick", "slow", "kind",
              "loud", "quiet", "bright", "dim", "wet", "dry"]
    out: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        choice = attempts % 2
        if choice == 0:
            # Lookup pattern
            k = rng.choice(names)
            v = rng.choice(values)
            k2 = rng.choice(names)
            v2 = rng.choice(values)
            s = f"{k}: {v}\n{k2}: {v2}\n{k}:"
        else:
            # Coreference chain
            a = rng.choice(names)
            b = rng.choice(names)
            if a == b:
                continue
            v = rng.choice(values)
            s = (f"{a} met {b} yesterday. {b} was feeling {v}. "
                 f"The next morning, {a} said that {b} was still")
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    if len(out) < n:
        raise RuntimeError(
            f"Family E: only generated {len(out)} distinct strings."
        )
    return out[:n]


# ════════════════════════════════════════════════════════════════════════════
#  Probe-family dispatcher
# ════════════════════════════════════════════════════════════════════════════


def generate_all_families(
    n_per_family: int, seed: int, max_seq_len: int, allow_synthetic: bool,
) -> dict[str, list[str]]:
    """Return {label → list[str]} for every family label."""
    logger.info("Generating %d probes per family (6 families) ...", n_per_family)
    fams: dict[str, list[str]] = {}
    fams["A"]         = generate_family_A(n_per_family, seed)
    fams["B"]         = generate_family_B(n_per_family, seed)
    fams["C_mem"], fams["C_perturb"] = generate_family_C(
        n_per_family, seed, allow_synthetic,
    )
    fams["D"]         = generate_family_D(n_per_family, seed, max_seq_len)
    fams["E"]         = generate_family_E(n_per_family, seed)
    for label, probes in fams.items():
        logger.info("  family %-10s : %4d probes (sample: %r)",
                    label, len(probes),
                    probes[0][:70] + ("..." if len(probes[0]) > 70 else ""))
    return fams


def tokenize_family(
    tokenizer, strings: list[str], seq_len: int,
) -> torch.Tensor:
    """Tokenize a list of strings to a dense (N, seq_len) tensor."""
    toks = tokenizer(
        strings,
        max_length=seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return toks["input_ids"]


# ════════════════════════════════════════════════════════════════════════════
#  Logit collection
# ════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def collect_logits(
    model,
    ids: torch.Tensor,         # (N, T)
    device: torch.device,
    batch_size: int,
    desc: str = "logits",
) -> torch.Tensor:
    """Forward-pass a token tensor and return last-position logits (N, V) on CPU."""
    N = ids.shape[0]
    V = model.config.vocab_size
    Z = torch.zeros(N, V, dtype=torch.float32)
    pbar = tqdm(total=N, desc=desc, unit="q")
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        batch = ids[s:e].to(device)
        out = model(input_ids=batch)
        Z[s:e] = out.logits[:, -1, :].float().cpu()
        pbar.update(e - s)
    pbar.close()
    return Z


# ════════════════════════════════════════════════════════════════════════════
#  SVD + effective-rank analysis
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class SVDReport:
    label: str
    N: int
    singular_values_top20: list[float]
    eff_rank: int                       # with relative threshold
    eff_rank_ratio_to_passive: float    # vs PASSIVE_BASELINE_EFF_RANK
    stable_rank: float                  # ||sigma||^2_2 / ||sigma||^2_inf
    spectral_entropy: float             # H(sigma^2 / sum sigma^2) / log K
    sigma_max: float
    sigma_min_nonzero: float
    cond_number: float
    Vh_top: Optional[torch.Tensor] = None    # (eff_rank, V) — optional for further analysis


def compute_svd_report(
    Z: torch.Tensor,                     # (N, V)
    label: str,
    eff_rank_rel_threshold: float,
    keep_Vh: bool = False,
    max_eff_rank: int = 200,
) -> SVDReport:
    """Centered SVD diagnostics for one logit block."""
    N, V = Z.shape
    Z_c = Z - Z.mean(dim=0, keepdim=True)
    logger.info("[svd:%s] N=%d, V=%d — running SVD ...", label, N, V)
    t0 = time.time()
    _, S, Vh = torch.linalg.svd(Z_c, full_matrices=False)
    logger.info("[svd:%s] SVD done in %.1fs. sigma_0 = %.3e",
                label, time.time() - t0, float(S[0]))

    S_f = S.float()
    sigma_max = float(S_f[0].item())
    # Effective rank with relative threshold
    if sigma_max <= 0:
        eff_rank = 0
    else:
        eff_rank = int((S_f / sigma_max > eff_rank_rel_threshold).sum().item())

    # Stable rank
    stable_rank = float((S_f.pow(2).sum() / (S_f[0].pow(2) + 1e-30)).item()) \
        if sigma_max > 0 else 0.0

    # Spectral entropy (normalised so it maxes at 1)
    p = S_f.pow(2)
    p = p / p.sum().clamp(min=1e-30)
    K = p.numel()
    eps = 1e-30
    H = float(-(p * (p + eps).log()).sum().item())
    H_norm = H / math.log(max(K, 2))

    nonzero = S_f[S_f > 0]
    sigma_min_nz = float(nonzero.min().item()) if nonzero.numel() > 0 else 0.0
    cond = sigma_max / max(sigma_min_nz, 1e-30)

    report = SVDReport(
        label=label,
        N=N,
        singular_values_top20=S_f[:20].tolist(),
        eff_rank=eff_rank,
        eff_rank_ratio_to_passive=(eff_rank / max(PASSIVE_BASELINE_EFF_RANK, 1)),
        stable_rank=stable_rank,
        spectral_entropy=H_norm,
        sigma_max=sigma_max,
        sigma_min_nonzero=sigma_min_nz,
        cond_number=cond,
    )
    if keep_Vh:
        take = min(max_eff_rank, Vh.shape[0])
        report.Vh_top = Vh[:take, :].float().cpu()
    logger.info("[svd:%s] eff_rank=%d (thr=%.0e)  stable_rank=%.2f  "
                "entropy=%.3f  cond=%.2e",
                label, eff_rank, eff_rank_rel_threshold,
                stable_rank, H_norm, cond)
    return report


# ════════════════════════════════════════════════════════════════════════════
#  Entropy diagnostics for Family C (memorization detection)
# ════════════════════════════════════════════════════════════════════════════


def logit_entropy(Z: torch.Tensor) -> torch.Tensor:
    """Softmax entropy per row, nats."""
    p = F.softmax(Z.double(), dim=-1)
    H = -(p * (p + 1e-30).log()).sum(dim=-1)
    return H.float()


# ════════════════════════════════════════════════════════════════════════════
#  Between-family subspace-separation analysis
# ════════════════════════════════════════════════════════════════════════════


def subspace_principal_angles(
    A: torch.Tensor,          # (k1, V)
    B: torch.Tensor,          # (k2, V)
) -> list[float]:
    """Return cos of the principal angles between two row-spaces."""
    # Row-orthonormalise via QR on the TRANSPOSE (so columns of Q span rows of A).
    QA, _ = torch.linalg.qr(A.T)           # (V, k1)
    QB, _ = torch.linalg.qr(B.T)           # (V, k2)
    M = QA.T @ QB                           # (k1, k2)
    s = torch.linalg.svdvals(M)
    return s.clamp(0.0, 1.0).tolist()


# ════════════════════════════════════════════════════════════════════════════
#  Parameter-recovery probe (optional, speculative)
# ════════════════════════════════════════════════════════════════════════════


def hungarian_align_cos(
    factors: torch.Tensor,                 # (d, R) candidate columns
    target: torch.Tensor,                  # (d, K) teacher columns
    max_rows: int,
) -> dict[str, Any]:
    """Hungarian column-assignment of |cos(candidate, teacher)|."""
    from scipy.optimize import linear_sum_assignment

    F_n = F.normalize(factors.float(), dim=0).cpu()
    T_n = F.normalize(target.float(), dim=0).cpu()
    d, R = F_n.shape
    _, K = T_n.shape

    sub_K = min(K, max_rows)
    if sub_K < K:
        idx = torch.linspace(0, K - 1, sub_K, dtype=torch.long)
        T_sub = T_n[:, idx]
    else:
        T_sub = T_n
        idx = torch.arange(K)

    cos_mat = T_sub.T @ F_n                 # (sub_K, R)
    cost = 1.0 - cos_mat.abs().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    aligned = cos_mat.abs().numpy()[row_ind, col_ind]
    aligned_top5 = float(np.mean(np.sort(aligned)[-5:])) if aligned.size >= 1 else 0.0
    return {
        "aligned_cos_mean":  float(np.mean(aligned)),
        "aligned_cos_top5":  aligned_top5,
        "aligned_cos_max":   float(np.max(aligned)),
        "num_matched":       int(len(row_ind)),
        "subsampled_K":      int(sub_K),
        "original_K":        int(K),
    }


def run_parameter_recovery_probe(
    family_reports: dict[str, SVDReport],
    teacher,
    max_rows: int,
) -> dict[str, Any]:
    """For every family's top-right-singular-vectors we form a linear map

        H_candidate = Vh_top @ W_lm^T          (eff_rank, d)

    and interpret its rows as candidate "hidden directions" excited by that
    family.  We then match these candidates against the teacher's last-block
    W_O columns and W_down columns via the Hungarian algorithm.  This is a
    speculative stand-in for true parameter recovery; we do not expect large
    aligned cosines — the value of the experiment is the NEGATIVE result,
    quantified against a random baseline.
    """
    logger.info("-" * 72)
    logger.info("Parameter-recovery probe (speculative)")
    logger.info("-" * 72)

    # Public lm_head / embed table (Qwen2.5-0.5B has tied embeddings)
    W_lm = teacher.lm_head.weight.detach().float().cpu()          # (V, d)
    # Teacher last-block targets (attacker does NOT normally see these; we
    # use them only for evaluation, just like in the other POC scripts).
    t_last = teacher.model.layers[LAST_BLOCK_IDX]
    targets: dict[str, torch.Tensor] = {
        "W_O.cols":    t_last.self_attn.o_proj.weight.detach().float().cpu(),   # (d, d)
        "W_down.cols": t_last.mlp.down_proj.weight.detach().float().cpu(),      # (d, d_ff)
    }

    out: dict[str, Any] = {}
    # Random baseline
    rng = torch.Generator().manual_seed(12345)
    d = W_lm.shape[1]
    rand_candidate = torch.randn(d, max(target.shape[1] for target in targets.values()),
                                 generator=rng)
    random_baseline = {}
    for tname, T_mat in targets.items():
        random_baseline[tname] = hungarian_align_cos(
            rand_candidate[:, :max(16, T_mat.shape[1] // 4)],
            T_mat, max_rows=max_rows,
        )
    out["random_baseline"] = random_baseline

    for label, rep in family_reports.items():
        if rep.Vh_top is None:
            continue
        # Candidate hidden directions: (eff_rank, d)
        Vh = rep.Vh_top                                        # (k, V)
        if Vh.shape[1] != W_lm.shape[0]:
            logger.warning("Skipping %s — Vh width %d != V %d",
                           label, Vh.shape[1], W_lm.shape[0])
            continue
        cand = Vh @ W_lm                                       # (k, d)
        cand = cand.T                                          # (d, k)
        fam_out: dict[str, Any] = {}
        for tname, T_mat in targets.items():
            fam_out[tname] = hungarian_align_cos(
                cand, T_mat, max_rows=max_rows,
            )
        out[label] = fam_out
        # Log a one-line summary
        parts = [f"{tn}: top5={r['aligned_cos_top5']:.4f}"
                 for tn, r in fam_out.items()]
        logger.info("  %-10s → %s", label, "  ".join(parts))

    return out


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    if args.dry_run_small:
        args.queries_per_family = 16
        args.max_seq_len = 16
        args.batch_size = 4

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    device = torch.device(args.device)
    if "cuda" in str(device) and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — falling back to CPU (slow).")
        device = torch.device("cpu")

    logger.info("=" * 72)
    logger.info("MEMORY / SELECTIVE-PROBING ATTACK (POC)")
    logger.info("=" * 72)
    logger.info("Model:               %s", args.model_name)
    logger.info("Queries/family:      %d", args.queries_per_family)
    logger.info("Seq len:             %d", args.max_seq_len)
    logger.info("Output:              %s", out_dir)
    logger.info("Device:              %s", device)

    # ── Load teacher ────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("-" * 72)
    logger.info("STAGE 1: Load teacher + tokenizer")
    logger.info("-" * 72)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    teacher.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = teacher.config.vocab_size
    hidden_size = teacher.config.hidden_size
    assert vocab_size == VOCAB_SIZE, \
        f"Expected V={VOCAB_SIZE}, got {vocab_size}"
    assert hidden_size == D_MODEL, \
        f"Expected d={D_MODEL}, got {hidden_size}"
    logger.info("[setup] teacher loaded: V=%d, d=%d, layers=%d",
                vocab_size, hidden_size, teacher.config.num_hidden_layers)

    # ── Generate families ───────────────────────────────────────────────────
    logger.info("-" * 72)
    logger.info("STAGE 2: Generate probe families")
    logger.info("-" * 72)
    families = generate_all_families(
        n_per_family=args.queries_per_family,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        allow_synthetic=args.allow_synthetic_wiki,
    )
    # Persist probes for reproducibility
    with (out_dir / "probes.json").open("w") as f:
        json.dump({k: v[: min(50, len(v))]
                   for k, v in families.items()}, f, indent=2)

    # ── Tokenize + collect logits per family ────────────────────────────────
    logger.info("-" * 72)
    logger.info("STAGE 3: Collect last-position logits")
    logger.info("-" * 72)
    family_logits: dict[str, torch.Tensor] = {}   # (N, V) per family
    family_ids:    dict[str, torch.Tensor] = {}   # (N, T) per family
    for label, probes in families.items():
        ids = tokenize_family(tokenizer, probes, args.max_seq_len)
        family_ids[label] = ids
        Z = collect_logits(
            teacher, ids, device=device,
            batch_size=args.batch_size,
            desc=f"logits[{label}]",
        )
        family_logits[label] = Z
        logger.info("[family %s] logits shape=%s, ||Z||=%.3e",
                    label, tuple(Z.shape), float(Z.norm()))

    # ── Per-family SVD diagnostics ──────────────────────────────────────────
    logger.info("-" * 72)
    logger.info("STAGE 4: Per-family SVD")
    logger.info("-" * 72)
    family_reports: dict[str, SVDReport] = {}
    for label, Z in family_logits.items():
        family_reports[label] = compute_svd_report(
            Z, label=label,
            eff_rank_rel_threshold=args.eff_rank_threshold,
            keep_Vh=True,
            max_eff_rank=200,
        )

    # ── Combined SVD across all families ────────────────────────────────────
    logger.info("-" * 72)
    logger.info("STAGE 5: Combined SVD across all families")
    logger.info("-" * 72)
    Z_all = torch.cat(list(family_logits.values()), dim=0)
    combined_report = compute_svd_report(
        Z_all, label="COMBINED",
        eff_rank_rel_threshold=args.eff_rank_threshold,
        keep_Vh=False,
    )

    # ── Family-C entropy analysis (memorization detection) ──────────────────
    logger.info("-" * 72)
    logger.info("STAGE 6: Family-C entropy analysis")
    logger.info("-" * 72)
    H_mem = logit_entropy(family_logits["C_mem"])
    H_perturb = logit_entropy(family_logits["C_perturb"])
    mem_entropy_report = {
        "mem_entropy_mean":       float(H_mem.mean()),
        "mem_entropy_std":        float(H_mem.std(unbiased=False)),
        "perturb_entropy_mean":   float(H_perturb.mean()),
        "perturb_entropy_std":    float(H_perturb.std(unbiased=False)),
        "entropy_gap_mean":       float((H_perturb - H_mem).mean()),
        # Positive gap ⇒ memorized is more confident (lower entropy).
        # This is the direction predicted by Carlini et al. 2021
        # "Extracting Training Data from Large Language Models".
    }
    logger.info("[mem] H(mem) = %.4f ± %.4f  |  H(perturb) = %.4f ± %.4f  |  "
                "gap = %+.4f",
                mem_entropy_report["mem_entropy_mean"],
                mem_entropy_report["mem_entropy_std"],
                mem_entropy_report["perturb_entropy_mean"],
                mem_entropy_report["perturb_entropy_std"],
                mem_entropy_report["entropy_gap_mean"])

    # ── Between-family subspace separation ──────────────────────────────────
    logger.info("-" * 72)
    logger.info("STAGE 7: Between-family subspace separation (principal angles)")
    logger.info("-" * 72)
    pair_cos: dict[str, dict[str, float]] = {}
    labels = list(family_reports.keys())
    for i, a in enumerate(labels):
        pair_cos[a] = {}
        rep_a = family_reports[a]
        if rep_a.Vh_top is None:
            continue
        for j, b in enumerate(labels):
            if j == i:
                continue
            rep_b = family_reports[b]
            if rep_b.Vh_top is None:
                continue
            cos_vals = subspace_principal_angles(
                rep_a.Vh_top, rep_b.Vh_top,
            )
            pair_cos[a][b] = {
                "n_angles":     len(cos_vals),
                "mean_cos":     float(np.mean(cos_vals)),
                "min_cos":      float(np.min(cos_vals)) if cos_vals else 1.0,
                "max_cos":      float(np.max(cos_vals)) if cos_vals else 1.0,
            }
            logger.info("  subspace(%s, %s)  mean_cos=%.4f  "
                        "min_cos=%.4f  max_cos=%.4f",
                        a, b, pair_cos[a][b]["mean_cos"],
                        pair_cos[a][b]["min_cos"],
                        pair_cos[a][b]["max_cos"])

    # ── Optional: parameter-recovery probe ──────────────────────────────────
    logger.info("-" * 72)
    logger.info("STAGE 8: Parameter-recovery probe")
    logger.info("-" * 72)
    try:
        param_probe = run_parameter_recovery_probe(
            family_reports=family_reports,
            teacher=teacher,
            max_rows=args.param_match_rows,
        )
    except Exception as e:
        logger.warning("parameter-recovery probe failed: %s", e)
        param_probe = {"error": str(e)}

    # Free teacher / logits before writing the report
    del teacher
    for label in list(family_logits.keys()):
        family_logits[label] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Build the full results payload ──────────────────────────────────────
    def _report_to_dict(r: SVDReport) -> dict[str, Any]:
        d = asdict(r)
        d.pop("Vh_top", None)       # do not JSON-serialise the tensor
        return d

    # Compute best-family eff_rank + best parameter-recovery cos for the verdict
    per_family_eff_rank: dict[str, int] = {
        label: rep.eff_rank for label, rep in family_reports.items()
    }
    best_family_eff_rank = max(per_family_eff_rank.values())
    combined_eff_rank = combined_report.eff_rank

    # Parameter-recovery cos: max top5 aligned cos across all (family, target)
    # pairs, compared against the random baseline.
    param_cos_best = -1.0
    param_cos_best_entry = {"family": "none", "matrix": "none",
                             "real": -1.0, "random": -1.0}
    rand_baseline = param_probe.get("random_baseline", {}) if isinstance(param_probe, dict) else {}
    for label, fam_out in param_probe.items() if isinstance(param_probe, dict) else []:
        if label == "random_baseline" or label == "error":
            continue
        for tname, m in fam_out.items():
            real = float(m.get("aligned_cos_top5", 0.0))
            rnd = float(rand_baseline.get(tname, {}).get("aligned_cos_top5", 0.0))
            if real > param_cos_best:
                param_cos_best = real
                param_cos_best_entry = {
                    "family": label, "matrix": tname,
                    "real": real, "random": rnd,
                }

    base_success = combined_eff_rank > args.success_eff_rank
    # Parameter-recovery extended success: real - random > threshold
    extended_success = (
        (param_cos_best - float(param_cos_best_entry["random"])) > args.success_param_cos
    )

    verdict = {
        "base_success":            bool(base_success),
        "extended_success":        bool(extended_success),
        "combined_eff_rank":       int(combined_eff_rank),
        "passive_baseline_eff_rank": int(PASSIVE_BASELINE_EFF_RANK),
        "best_family_eff_rank":    int(best_family_eff_rank),
        "combined_eff_rank_gain":  int(combined_eff_rank - PASSIVE_BASELINE_EFF_RANK),
        "success_eff_rank_threshold": int(args.success_eff_rank),
        "per_family_eff_rank":     per_family_eff_rank,
        "param_cos_best":          param_cos_best,
        "param_cos_best_entry":    param_cos_best_entry,
        "success_param_cos_threshold": args.success_param_cos,
    }

    results: dict[str, Any] = {
        "args": vars(args),
        "family_descriptions": FAMILY_DESCRIPTIONS,
        "per_family_svd":      {l: _report_to_dict(r)
                                for l, r in family_reports.items()},
        "combined_svd":        _report_to_dict(combined_report),
        "memorization_entropy": mem_entropy_report,
        "pair_subspace_cosines": pair_cos,
        "parameter_recovery_probe": param_probe,
        "verdict": verdict,
    }
    results_path = out_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Human-readable summary ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  MEMORY / SELECTIVE-PROBING ATTACK — SUMMARY")
    print("=" * 72)
    print(f"  Model:                      {args.model_name}")
    print(f"  Queries/family:             {args.queries_per_family}  "
          f"(seq_len={args.max_seq_len})")
    print(f"  Total queries collected:    {args.queries_per_family * len(families)}")
    print(f"  Output dir:                 {out_dir}")
    print("-" * 72)
    print("  Per-family SVD diagnostics:")
    print(f"    {'family':12s} {'N':>5s} {'eff_rank':>10s} "
          f"{'stable_rank':>12s} {'entropy':>8s} {'cond':>10s}")
    for label in labels:
        r = family_reports[label]
        print(f"    {label:12s} {r.N:>5d} {r.eff_rank:>10d} "
              f"{r.stable_rank:>12.2f} {r.spectral_entropy:>8.3f} "
              f"{r.cond_number:>10.2e}")
    print("-" * 72)
    print("  Combined SVD (union of all families):")
    print(f"    N = {combined_report.N}")
    print(f"    eff_rank         = {combined_report.eff_rank}   "
          f"(passive-text baseline ~ {PASSIVE_BASELINE_EFF_RANK})")
    print(f"    stable_rank      = {combined_report.stable_rank:.2f}")
    print(f"    spectral_entropy = {combined_report.spectral_entropy:.3f}")
    print(f"    sigma_0          = {combined_report.sigma_max:.3e}")
    print(f"    top 10 singular values:")
    for i, v in enumerate(combined_report.singular_values_top20[:10]):
        print(f"      [{i:2d}]  {v:.3e}")
    print("-" * 72)
    print("  Family-C memorization entropy:")
    print(f"    H(mem)     = {mem_entropy_report['mem_entropy_mean']:.4f} "
          f"± {mem_entropy_report['mem_entropy_std']:.4f}")
    print(f"    H(perturb) = {mem_entropy_report['perturb_entropy_mean']:.4f} "
          f"± {mem_entropy_report['perturb_entropy_std']:.4f}")
    gap = mem_entropy_report["entropy_gap_mean"]
    interp = ("mem more CONFIDENT (consistent with memorisation)"
              if gap > 0
              else "mem LESS confident (no memorisation signal)")
    print(f"    gap        = {gap:+.4f}  ({interp})")
    print("-" * 72)
    print("  Between-family subspace separation (mean cos of principal angles):")
    hdr = "           " + " ".join(f"{l:>10s}" for l in labels)
    print(hdr)
    for a in labels:
        row = f"  {a:8s}"
        for b in labels:
            if a == b:
                row += f"  {'1.0000':>10s}"
            else:
                val = pair_cos.get(a, {}).get(b, {}).get("mean_cos", float('nan'))
                row += f"  {val:10.4f}"
        print(row)
    print("-" * 72)
    print("  Parameter-recovery probe (speculative):")
    if isinstance(param_probe, dict) and "error" not in param_probe:
        print(f"    {'family':12s} {'matrix':14s} {'real top5':>10s} "
              f"{'rand top5':>10s} {'real - rand':>12s}")
        rand_base = param_probe.get("random_baseline", {})
        for label in labels:
            fam_out = param_probe.get(label, {})
            if not isinstance(fam_out, dict):
                continue
            for tname, m in fam_out.items():
                real = float(m.get("aligned_cos_top5", 0.0))
                rnd = float(rand_base.get(tname, {}).get("aligned_cos_top5", 0.0))
                print(f"    {label:12s} {tname:14s} {real:>10.4f} "
                      f"{rnd:>10.4f} {real - rnd:>+12.4f}")
        print(f"    best: family={param_cos_best_entry['family']} "
              f"matrix={param_cos_best_entry['matrix']} "
              f"real={param_cos_best_entry['real']:.4f} "
              f"rand={param_cos_best_entry['random']:.4f}")
    else:
        print(f"    (probe error: {param_probe.get('error', 'unknown')})")
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    gain = combined_eff_rank - PASSIVE_BASELINE_EFF_RANK
    print(f"  Combined eff_rank:  {combined_eff_rank}  "
          f"(passive baseline {PASSIVE_BASELINE_EFF_RANK};  "
          f"{'+' if gain >= 0 else ''}{gain})")
    print(f"  Best-family eff_rank: {best_family_eff_rank}")
    print(f"  Success threshold (combined eff_rank > {args.success_eff_rank}):  "
          f"{'MET' if verdict['base_success'] else 'NOT MET'}")
    print(f"  Parameter-recovery best signal-above-random: "
          f"{param_cos_best - float(param_cos_best_entry['random']):+.4f}  "
          f"(threshold {args.success_param_cos:.2f})")

    if verdict["base_success"] and verdict["extended_success"]:
        print("\n  SUCCESS [BASE + EXTENDED]")
        print("    * Selective probing expands the observable subspace "
              f"from {PASSIVE_BASELINE_EFF_RANK} to {combined_eff_rank}.")
        print("    * A parameter-recovery signal is detectable above chance "
              f"(best cos > rand by {param_cos_best - float(param_cos_best_entry['random']):.4f}).")
        print("    * This is a NOVEL finding: the residual-stream low-rank barrier is")
        print("      a property of the probe distribution, NOT the architecture.")
    elif verdict["base_success"]:
        print("\n  SUCCESS [BASE]")
        print(f"    * Selective probing expands the observable subspace "
              f"from {PASSIVE_BASELINE_EFF_RANK} to {combined_eff_rank}.")
        print("    * The residual-stream low-rank barrier is softer than previously")
        print("      believed; targeted probe families excite additional dimensions.")
        print("    * Parameter recovery has not succeeded, so the barrier to block-")
        print("      level parameter extraction remains.")
    else:
        print("\n  FAILED (but informative)")
        print(f"    * Combined eff_rank {combined_eff_rank} "
              f"did NOT exceed threshold {args.success_eff_rank}.")
        if combined_eff_rank <= PASSIVE_BASELINE_EFF_RANK + 3:
            print("    * The observable subspace barely grows over passive WikiText.")
            print("    * This CONFIRMS the residual-stream low-rank barrier is")
            print("      an intrinsic property of the decoder architecture,")
            print("      not an artifact of the query distribution.")
        else:
            print(f"    * Selective probing DOES raise eff_rank by "
                  f"{combined_eff_rank - PASSIVE_BASELINE_EFF_RANK} over passive,")
            print("      but not enough to clear the success threshold.")
            print("      Increase queries_per_family, add more diverse probe")
            print("      families, or lower the eff_rank_threshold to quantify")
            print("      the weaker signal.")
        print("    * Root-cause candidates for persisting low-rank:")
        print("      - RMSNorm at the end of the residual stream projects")
        print("        onto a unit sphere, suppressing magnitude information.")
        print("      - Tied embeddings mean the output subspace is locked to")
        print("        the input embedding columns regardless of how we probe.")
        print("      - Most circuits contribute through a few dominant heads")
        print("        whose outputs ALREADY saturate the ~d_hidden/100 low-rank")
        print("        subspace.")

    print("=" * 72)
    print(f"  Results JSON:  {results_path}")
    print(f"  Probes JSON:   {out_dir / 'probes.json'}")
    print("=" * 72 + "\n")

    # Terminal one-word status line (consumed by shell scripts)
    if verdict["base_success"]:
        print("SUCCESS")
    else:
        print("FAILED")


if __name__ == "__main__":
    main()
