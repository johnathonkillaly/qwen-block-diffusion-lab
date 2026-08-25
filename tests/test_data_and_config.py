"""Config loading, dataset windowing, collation, and the auxiliary modules."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from qdif.config import ExperimentConfig, load_config, save_config
from qdif.diffusion.self_conditioning import (
    SelfConditioner,
    TimestepConditioner,
    sinusoidal_features,
)


def test_defaults_are_self_consistent():
    cfg = ExperimentConfig()
    assert cfg.diffusion.canvas_length > 0
    assert cfg.training.mode in ("lora", "full")
    assert cfg.diffusion.self_conditioning is False, "milestone 1 default"


def test_yaml_round_trip(tmp_path: Path):
    cfg = ExperimentConfig()
    cfg.lora.rank = 64
    cfg.diffusion.canvas_length = 128
    save_config(cfg, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml").lora.rank == 64
    assert load_config(tmp_path / "c.yaml").diffusion.canvas_length == 128


def test_unknown_key_fails_loudly(tmp_path: Path):
    (tmp_path / "bad.yaml").write_text("lora:\n  rnak: 16\n")
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(tmp_path / "bad.yaml")


def test_nested_sections_are_typed(tmp_path: Path):
    (tmp_path / "c.yaml").write_text("diffusion:\n  canvas_length: 64\n  loss_on: corrupted\n")
    cfg = load_config(tmp_path / "c.yaml")
    assert cfg.diffusion.canvas_length == 64
    assert cfg.diffusion.loss_on == "corrupted"
    assert cfg.training.backend == "torch"  # untouched sections keep defaults


@pytest.mark.parametrize(
    "path",
    sorted(str(p) for p in Path("configs").glob("*.yaml")) if Path("configs").exists() else [],
)
def test_shipped_configs_all_load(path):
    load_config(path)


# ------------------------------------------------------------------- conditioners


def test_sinusoidal_features_shape_and_range():
    f = sinusoidal_features(torch.linspace(0, 1, 8), 64)
    assert f.shape == (8, 64)
    assert torch.isfinite(f).all()
    assert float(f.abs().max()) <= 1.0


def test_sinusoidal_features_distinguish_timesteps():
    f = sinusoidal_features(torch.tensor([0.0, 0.5, 1.0]), 64)
    assert not torch.allclose(f[0], f[1])
    assert not torch.allclose(f[1], f[2])


def test_timestep_conditioner_starts_as_a_no_op():
    """Zero-initialised output keeps the diffusion path identical to AR at init."""
    cond = TimestepConditioner(hidden_size=16, feature_dim=32)
    out = cond(torch.tensor([0.0, 0.7]), torch.float32)
    assert out.shape == (2, 1, 16)
    assert float(out.abs().max()) == 0.0


def test_timestep_conditioner_becomes_active_after_training():
    cond = TimestepConditioner(hidden_size=16, feature_dim=32)
    with torch.no_grad():
        cond.mlp[-1].weight.normal_(std=0.1)
    assert float(cond(torch.tensor([0.3]), torch.float32).abs().max()) > 0


def test_self_conditioner_starts_as_a_no_op():
    sc = SelfConditioner(hidden_size=16, top_k=4)
    logits = torch.randn(2, 5, 32)
    embedding = torch.randn(32, 16)
    out = sc(logits, embedding)
    assert out.shape == (2, 5, 16)
    assert float(out.abs().max()) == 0.0


def test_expected_embedding_is_a_convex_combination():
    sc = SelfConditioner(hidden_size=4, top_k=0)  # exact full-vocab expectation
    embedding = torch.eye(4)
    logits = torch.full((1, 1, 4), 0.0)  # uniform over 4 tokens
    e = sc.expected_embedding(logits, embedding)
    assert torch.allclose(e, torch.full((1, 1, 4), 0.25), atol=1e-6)


def test_topk_expectation_approximates_the_full_one():
    torch.manual_seed(0)
    embedding = torch.randn(64, 8)
    logits = torch.randn(1, 3, 64) * 4  # peaked, so top-k captures most of the mass
    exact = SelfConditioner(8, top_k=0).expected_embedding(logits, embedding)
    approx = SelfConditioner(8, top_k=32).expected_embedding(logits, embedding)
    assert torch.allclose(exact, approx, atol=0.05)


# ----------------------------------------------------------------------- datasets


class DummyTokenizer:
    """Whitespace tokenizer, enough to exercise windowing without a checkpoint."""

    vocab_size = 100
    eos_token_id = 99
    mask_token_id = None

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [(abs(hash(w)) % 90) + 1 for w in text.split()]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(i)) for i in ids)


def test_dataset_windows_have_the_requested_shape():
    from qdif.data.datasets import CanvasDataset

    texts = [" ".join(f"w{i}" for i in range(200))]
    ds = CanvasDataset(texts, DummyTokenizer(), prefix_length=8, canvas_length=16)
    assert len(ds) > 0
    assert ds[0].prefix_ids.shape == (8,)
    assert ds[0].canvas_ids.shape == (16,)


def test_dataset_rejects_texts_shorter_than_the_window():
    from qdif.data.datasets import CanvasDataset

    with pytest.raises(ValueError, match="shorter than"):
        CanvasDataset(["too short"], DummyTokenizer(), prefix_length=8, canvas_length=16)


def test_dataset_windows_do_not_overlap_by_default():
    from qdif.data.datasets import CanvasDataset

    texts = [" ".join(f"w{i}" for i in range(96))]
    ds = CanvasDataset(texts, DummyTokenizer(), prefix_length=8, canvas_length=16)
    joined = torch.cat([ds[0].prefix_ids, ds[0].canvas_ids])
    joined_next = torch.cat([ds[1].prefix_ids, ds[1].canvas_ids])
    assert not torch.equal(joined, joined_next)


def test_collator_produces_a_consistent_batch():
    from qdif.config import DiffusionConfig
    from qdif.data.collator import DiffusionCollator
    from qdif.data.datasets import CanvasDataset

    texts = [" ".join(f"w{i}" for i in range(200))]
    ds = CanvasDataset(texts, DummyTokenizer(), prefix_length=8, canvas_length=16)
    coll = DiffusionCollator(
        DiffusionConfig(canvas_length=16), vocab_size=100, generator=torch.Generator().manual_seed(0)
    )
    batch = coll([ds[0], ds[1]])
    assert batch.prefix_ids.shape == (2, 8)
    assert batch.canvas_x0.shape == (2, 16)
    assert batch.canvas_xt.shape == (2, 16)
    assert batch.t.shape == (2,)
    # xt differs from x0 exactly where the process targeted a position (up to
    # coincidental identical replacements).
    assert bool(((batch.canvas_xt != batch.canvas_x0) <= batch.corrupted_mask).all())


def test_collator_honours_a_fixed_timestep():
    from qdif.config import DiffusionConfig
    from qdif.data.collator import DiffusionCollator
    from qdif.data.datasets import CanvasDataset

    texts = [" ".join(f"w{i}" for i in range(200))]
    ds = CanvasDataset(texts, DummyTokenizer(), prefix_length=8, canvas_length=16)
    coll = DiffusionCollator(DiffusionConfig(), vocab_size=100, generator=torch.Generator().manual_seed(0))
    batch = coll([ds[0]], t=0.0)
    assert float(batch.t[0]) == 0.0
    assert torch.equal(batch.canvas_xt, batch.canvas_x0)


def test_same_example_gets_different_noise_across_calls():
    """Corruption must be dynamic -- no pre-generated corrupted dataset."""
    from qdif.config import DiffusionConfig
    from qdif.data.collator import DiffusionCollator
    from qdif.data.datasets import CanvasDataset

    texts = [" ".join(f"w{i}" for i in range(200))]
    ds = CanvasDataset(texts, DummyTokenizer(), prefix_length=8, canvas_length=16)
    coll = DiffusionCollator(DiffusionConfig(), vocab_size=100, generator=torch.Generator().manual_seed(0))
    a = coll([ds[0]], t=0.5)
    b = coll([ds[0]], t=0.5)
    assert torch.equal(a.canvas_x0, b.canvas_x0)
    assert not torch.equal(a.canvas_xt, b.canvas_xt)
