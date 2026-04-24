"""Result manifest: every experiment run saves a complete provenance record."""
from __future__ import annotations
import json, subprocess, time, sys, os
from pathlib import Path


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def save_manifest(output_dir: str | Path, *,
                  command: str = "",
                  config: dict = None,
                  seed: int = None,
                  model_name: str = "",
                  dataset: str = "",
                  split_info: dict = None,
                  query_budget: dict = None,
                  metrics: dict = None,
                  extra: dict = None):
    """Save manifest.json alongside results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_hash": get_git_hash(),
        "python_version": sys.version.split()[0],
        "command": command or " ".join(sys.argv),
        "config": config or {},
        "seed": seed,
        "model_name": model_name,
        "dataset": dataset,
        "split_info": split_info or {},
        "query_budget": query_budget or {},
        "metrics": metrics or {},
    }
    if extra:
        manifest.update(extra)

    path = output_dir / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
