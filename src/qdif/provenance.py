"""Run provenance: enough metadata in every run directory to reconstruct what produced it.

Written next to the metrics, not in a lab notebook, so a `runs/` directory found in
six months is self-describing: package versions, git commit and dirtiness, config,
dataset revision, seeds, model path and hardware.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "mlx", "mlx-lm", "mlx-metal", "unsloth", "unsloth_zoo", "transformers", "torch",
    "numpy", "datasets", "huggingface-hub", "tokenizers", "safetensors", "pyyaml",
)

#: Pinned so a reproduction downloads the same bytes.
DATASET_REVISIONS = {
    "Salesforce/wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
}


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def package_versions() -> dict[str, str | None]:
    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    try:
        info["processor"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        mem = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        info["unified_memory_gb"] = round(int(mem) / 1e9, 1) if mem else None
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        import mlx.core as mx

        info["mlx_device"] = str(mx.default_device())
        info["metal_available"] = bool(mx.metal.is_available())
    except ImportError:
        pass
    return info


def run_metadata(cfg, extra: dict | None = None) -> dict:
    dataset = getattr(cfg.data, "hf_dataset", None) or ""
    repo = dataset.partition(":")[0]
    return {
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
            "describe": _git("describe", "--always", "--dirty"),
        },
        "packages": package_versions(),
        "hardware": hardware(),
        "config": cfg.to_dict(),
        "seeds": {"training": cfg.training.seed, "eval": 777},
        "dataset": {
            "id": dataset or None,
            "revision": DATASET_REVISIONS.get(repo),
            "train_split": getattr(cfg.data, "hf_split", None),
            "eval_split": getattr(cfg.data, "eval_split", None),
        },
        "model": {"id": cfg.model.id, "local_path": cfg.model.local_path},
        **(extra or {}),
    }


def write_run_metadata(run_dir: str | Path, cfg, extra: dict | None = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_metadata.json"
    path.write_text(json.dumps(run_metadata(cfg, extra), indent=2, default=str))
    return path
