"""Tests for result manifest."""
import json, tempfile, os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.result_manifest import save_manifest, get_git_hash


def test_manifest_saves_required_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_manifest(
            tmpdir,
            command="test_command",
            config={"lr": 1e-4},
            seed=42,
            model_name="test-model",
            dataset="wikitext",
            split_info={"train": "train", "eval": "validation"},
            query_budget={"topk": 100, "probe": 2000},
            metrics={"ppl": 24.0, "kl": 0.1},
        )
        assert path.exists()
        with open(path) as f:
            m = json.load(f)
        assert "timestamp" in m
        assert "git_hash" in m
        assert m["seed"] == 42
        assert m["model_name"] == "test-model"
        assert m["config"]["lr"] == 1e-4
        assert m["query_budget"]["probe"] == 2000


def test_git_hash_returns_string():
    h = get_git_hash()
    assert isinstance(h, str)
    assert len(h) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
