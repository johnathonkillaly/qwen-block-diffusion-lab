"""Shared fixtures.

Tests split into two groups:

  * pure-math tests (corruption, schedules, objective, masks, LoRA mechanics) run
    with no checkpoint and no network -- these are the ones that must always pass.
  * `@pytest.mark.model` tests need a real Qwen3.5 checkpoint on disk and are
    skipped automatically when none is present.

Run only the fast set with:  pytest -m "not model"
"""

from __future__ import annotations

import pytest
import torch

from qdif.config import ExperimentConfig, load_config

SMOKE_CONFIG = "configs/smoke_16.yaml"


@pytest.fixture(scope="session")
def cfg() -> ExperimentConfig:
    try:
        return load_config(SMOKE_CONFIG)
    except FileNotFoundError:
        return ExperimentConfig()


@pytest.fixture(scope="session")
def model_path(cfg):
    """Resolved local checkpoint, or skip the test."""
    from qdif.models.registry import resolve_model_path

    try:
        return resolve_model_path(
            cfg.model.id, cfg.model.local_path, local_files_only=True
        )
    except FileNotFoundError as exc:
        pytest.skip(f"no local Qwen3.5 checkpoint: {exc}")


@pytest.fixture(scope="session")
def tokenizer(model_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path))


@pytest.fixture(scope="session")
def diffusion_model(cfg, model_path):
    """A DiffusionQwen with LoRA attached. Session-scoped: loading is the slow part."""
    from qdif.models.qwen35_adapter import load_qwen35_text, resolve_device, resolve_dtype
    from qdif.models.qwen35_diffusion import DiffusionQwen
    from qdif.training.lora import freeze_base, inject_lora

    device = resolve_device(cfg.training.device)
    base, tok = load_qwen35_text(
        model_path,
        dtype=resolve_dtype(cfg.training.dtype),
        device=device,
        attn_implementation=cfg.model.attn_implementation,
    )
    inject_lora(
        base,
        rank=cfg.lora.rank,
        alpha=cfg.lora.alpha,
        target_preset=cfg.lora.target_preset,
    )
    freeze_base(base)
    model = DiffusionQwen(base, cfg.diffusion).to(device)
    for module in model.auxiliary_modules().values():
        for p in module.parameters():
            p.requires_grad_(True)
    return model, tok


@pytest.fixture
def gen() -> torch.Generator:
    return torch.Generator().manual_seed(0)
