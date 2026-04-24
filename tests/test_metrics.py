"""Metric sanity tests."""
import torch
import torch.nn.functional as F
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_self_kl_is_zero():
    """KL(p || p) = 0 for any distribution."""
    logits = torch.randn(4, 10, 100)
    lp = F.log_softmax(logits, dim=-1)
    kl = (lp.exp() * (lp - lp)).sum(-1).mean()
    assert abs(kl.item()) < 1e-5, f"Self-KL should be 0, got {kl.item()}"


def test_kl_is_positive_for_different_distributions():
    """KL is non-negative and positive for distinct distributions."""
    torch.manual_seed(0)
    t = torch.randn(4, 10, 100)
    s = torch.randn(4, 10, 100)
    tlp = F.log_softmax(t, dim=-1)
    slp = F.log_softmax(s, dim=-1)
    kl = (tlp.exp() * (tlp - slp)).sum(-1).mean()
    assert kl.item() > 0, f"KL between distinct dists should be > 0, got {kl.item()}"


def test_top1_agreement_zero_for_identical_argmax():
    """Top-1 agreement is 1.0 when argmax matches."""
    t_logits = torch.randn(2, 5, 100)
    s_logits = t_logits.clone()
    agree = (t_logits.argmax(-1) == s_logits.argmax(-1)).float().mean().item()
    assert agree == 1.0


def test_perplexity_is_exp_of_ce():
    import math
    loss = 2.5
    ppl = math.exp(loss)
    assert abs(ppl - 12.1825) < 0.01


def test_topk_mask_shape():
    """topk returns K values per position."""
    logits = torch.randn(3, 4, 50)
    K = 5
    vals, idx = logits.topk(K, dim=-1)
    assert vals.shape == (3, 4, 5)
    assert idx.shape == (3, 4, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
