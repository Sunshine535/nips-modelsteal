"""Data split sanity tests — manifest must record split info."""
import json, tempfile, os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.result_manifest import save_manifest


def test_split_info_recorded_in_manifest():
    """Manifest must save train/eval split names explicitly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_manifest(
            tmpdir,
            split_info={
                "train": "train",
                "eval": "validation",
                "train_samples": 50000,
                "eval_samples": 2000,
            },
        )
        with open(path) as f:
            m = json.load(f)
        assert "split_info" in m
        assert m["split_info"]["train"] == "train"
        assert m["split_info"]["eval"] == "validation"
        assert m["split_info"]["train"] != m["split_info"]["eval"], \
            "Train and eval split names MUST be different"


def test_manifest_captures_query_budget():
    """Query budget must be in manifest for auditable cost tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_manifest(
            tmpdir,
            query_budget={"topk_queries": 10000, "probe_queries": 500000,
                          "total_queries": 510000, "max_probe_budget": -1},
        )
        with open(path) as f:
            m = json.load(f)
        assert "query_budget" in m
        assert m["query_budget"]["topk_queries"] == 10000
        assert m["query_budget"]["probe_queries"] == 500000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
