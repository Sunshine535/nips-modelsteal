"""Basic tests for the nips-modelsteal project."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parameter_inverter import (
    BlackBoxTeacher,
    SPSIConfig,
    TeacherCache,
    BlockResult,
    SPSIResult,
    get_num_blocks,
    get_lm_head_param_names,
    invert_block,
)
from src.permutation_alignment import (
    align_attention_heads,
    align_ffn_neurons,
    remove_rmsnorm_scale,
    compute_aligned_cosine,
)


# ── Tiny model for fast tests ────────────────────────────────────────

class TinyTransformerBlock(nn.Module):
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(d)
        self.self_attn = nn.ModuleDict({
            "q_proj": nn.Linear(d, d, bias=False),
            "k_proj": nn.Linear(d, d, bias=False),
            "v_proj": nn.Linear(d, d, bias=False),
            "o_proj": nn.Linear(d, d, bias=False),
        })
        self.post_attention_layernorm = nn.LayerNorm(d)
        self.mlp = nn.ModuleDict({
            "gate_proj": nn.Linear(d, d * 4, bias=False),
            "up_proj": nn.Linear(d, d * 4, bias=False),
            "down_proj": nn.Linear(d * 4, d, bias=False),
        })

    def forward(self, x, **kwargs):
        h = self.input_layernorm(x)
        q = self.self_attn["q_proj"](h)
        k = self.self_attn["k_proj"](h)
        v = self.self_attn["v_proj"](h)
        attn_out = self.self_attn["o_proj"](v)
        x = x + attn_out
        h2 = self.post_attention_layernorm(x)
        gate = torch.sigmoid(self.mlp["gate_proj"](h2))
        up = self.mlp["up_proj"](h2)
        x = x + self.mlp["down_proj"](gate * up)
        return (x,)


class TinyLM(nn.Module):
    """Minimal causal LM with transformer blocks for testing."""

    def __init__(self, vocab=128, d=64, n_layers=2, heads=4):
        super().__init__()

        class _Config:
            vocab_size = vocab
            num_attention_heads = heads
            hidden_size = d

        self.config = _Config()
        self.model = nn.ModuleDict({
            "embed_tokens": nn.Embedding(vocab, d),
            "layers": nn.ModuleList([TinyTransformerBlock(d, heads) for _ in range(n_layers)]),
        })
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids, output_hidden_states=False, return_dict=False, **kwargs):
        x = self.model["embed_tokens"](input_ids)
        hidden_states = [x] if output_hidden_states else None
        for layer in self.model["layers"]:
            x = layer(x)[0]
            if output_hidden_states:
                hidden_states.append(x)

        logits = self.lm_head(x)

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        if output_hidden_states:
            out.hidden_states = hidden_states
        return out


# ── Config Tests ─────────────────────────────────────────────────────

class TestSPSIConfig:
    def test_defaults(self):
        cfg = SPSIConfig()
        assert cfg.query_budget == 500_000
        assert cfg.batch_size == 16
        assert cfg.alpha == 1.0
        assert cfg.beta == 0.1
        assert cfg.gamma == 1e-5

    def test_custom(self):
        cfg = SPSIConfig(query_budget=1000, batch_size=4, alpha=2.0)
        assert cfg.query_budget == 1000
        assert cfg.batch_size == 4
        assert cfg.alpha == 2.0


# ── Data Structure Tests ─────────────────────────────────────────────

class TestBlockResult:
    def test_aliases(self):
        br = BlockResult(block_name="block_0", mean_cosine=0.95, num_queries=100)
        assert br.cosine_similarity == 0.95
        assert br.layer_name == "block_0"
        assert br.num_queries_used == 100

    def test_defaults(self):
        br = BlockResult(block_name="test")
        assert br.mean_cosine == -1.0
        assert br.l2_distance == -1.0


class TestSPSIResult:
    def test_layer_results_alias(self):
        r = SPSIResult(block_results=[BlockResult(block_name="b0")])
        assert len(r.layer_results) == 1
        assert r.recovered_state_dict is None


# ── BlackBoxTeacher Tests ────────────────────────────────────────────

class TestBlackBoxTeacher:
    def test_query_counts(self):
        model = TinyLM()
        teacher = BlackBoxTeacher(model, device="cpu")
        assert teacher.query_count == 0

        ids = torch.randint(0, 128, (2, 8))
        logits = teacher.query(ids)
        assert teacher.query_count == 2
        assert logits.shape == (2, 8, 128)

    def test_defense_fn(self):
        model = TinyLM()

        def round_defense(logits):
            return torch.round(logits)

        teacher = BlackBoxTeacher(model, device="cpu", defense_fn=round_defense)
        ids = torch.randint(0, 128, (1, 4))
        logits = teacher.query(ids)
        assert torch.equal(logits, torch.round(logits))

    def test_boundary_state(self):
        model = TinyLM(n_layers=2)
        teacher = BlackBoxTeacher(model, device="cpu")
        ids = torch.randint(0, 128, (2, 8))
        h = teacher.get_boundary_state(ids, layer_idx=0)
        assert h.shape == (2, 8, 64)


# ── TeacherCache Tests ───────────────────────────────────────────────

class TestTeacherCache:
    def test_build_and_get_batch(self):
        model = TinyLM()
        teacher = BlackBoxTeacher(model, device="cpu")
        cfg = SPSIConfig(num_perturbation_positions=2, num_replacement_tokens=1)
        cache = TeacherCache(device="cpu")

        ids = torch.randint(0, 128, (8, 16))
        cache.build(teacher, ids, cfg)

        assert cache.clean_logits is not None
        assert cache.clean_logits.shape[0] == 8

        indices = torch.tensor([0, 1])
        input_ids, cl, pi, pl = cache.get_batch(indices, cfg)
        assert input_ids.shape[0] == 2
        assert cl.shape[0] == 2


# ── Model Utility Tests ──────────────────────────────────────────────

class TestModelUtils:
    def test_get_num_blocks(self):
        model = TinyLM(n_layers=3)
        assert get_num_blocks(model) >= 2

    def test_get_lm_head_params(self):
        model = TinyLM()
        names = get_lm_head_param_names(model)
        assert len(names) >= 1
        assert any("lm_head" in n for n in names)


# ── Permutation Alignment Tests ──────────────────────────────────────

class TestPermutationAlignment:
    @pytest.fixture
    def paired_models(self):
        torch.manual_seed(0)
        teacher = TinyLM(vocab=128, d=64, n_layers=2, heads=4)
        torch.manual_seed(1)
        student = TinyLM(vocab=128, d=64, n_layers=2, heads=4)
        t_sd = {n: p.data.clone() for n, p in teacher.named_parameters()}
        s_sd = {n: p.data.clone() for n, p in student.named_parameters()}
        return t_sd, s_sd

    def test_align_attention_heads_runs(self, paired_models):
        t_sd, s_sd = paired_models
        prefix = "model.layers.0"
        result = align_attention_heads(s_sd, t_sd, prefix, num_heads=4, head_dim=16)
        assert isinstance(result, dict)
        assert len(result) >= len(s_sd)

    def test_align_ffn_neurons_runs(self, paired_models):
        t_sd, s_sd = paired_models
        prefix = "model.layers.0"
        result = align_ffn_neurons(s_sd, t_sd, prefix)
        assert isinstance(result, dict)

    def test_remove_rmsnorm_scale(self, paired_models):
        t_sd, _ = paired_models
        prefix = "model.layers.0"
        result = remove_rmsnorm_scale(t_sd, prefix)
        assert isinstance(result, dict)

    def test_compute_aligned_cosine(self, paired_models):
        t_sd, s_sd = paired_models
        prefix = "model.layers.0"
        unaligned, aligned = compute_aligned_cosine(s_sd, t_sd, prefix, 4, 16)
        assert isinstance(unaligned, dict)
        assert isinstance(aligned, dict)

    def test_identity_alignment_perfect(self):
        """Aligning identical params should give cosine ~1.0."""
        torch.manual_seed(42)
        model = TinyLM(vocab=128, d=64, n_layers=1, heads=4)
        sd = {n: p.data.clone() for n, p in model.named_parameters()}
        prefix = "model.layers.0"
        unaligned, aligned = compute_aligned_cosine(sd, sd, prefix, 4, 16)
        for v in aligned.values():
            assert v > 0.99, f"Self-alignment cosine should be ~1.0, got {v}"


# ── Smoke Test: invert_block ─────────────────────────────────────────

class TestInvertBlockSmoke:
    def test_runs_few_steps(self):
        """Verify invert_block runs without crashing (tiny model, 10 steps)."""
        torch.manual_seed(0)
        model = TinyLM(vocab=128, d=64, n_layers=1, heads=4)
        teacher = BlackBoxTeacher(model, device="cpu")

        torch.manual_seed(1)
        student = TinyLM(vocab=128, d=64, n_layers=1, heads=4)
        gt = {n: p.data.clone() for n, p in model.named_parameters()}

        cfg = SPSIConfig(
            batch_size=2,
            max_steps_per_block=10,
            num_perturbation_positions=1,
            num_replacement_tokens=1,
            max_seq_len=8,
            log_every=5,
            patience=100,
        )
        cache = TeacherCache(device="cpu")
        ids = torch.randint(0, 128, (4, 8))
        cache.build(teacher, ids, cfg)

        lm_names = get_lm_head_param_names(student)
        result = invert_block(
            student, cache, cfg, lm_names,
            ground_truth=gt, is_lm_head=True,
        )
        assert isinstance(result, BlockResult)
        assert result.num_steps > 0
        assert result.num_queries > 0
