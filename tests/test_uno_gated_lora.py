"""Gated LoRA: the adapter must be a no-op wherever Uno says the backbone runs bare."""

from __future__ import annotations

import mlx.core as mx
import pytest

from qdif.uno.gated_lora import (
    GatedLoRALinear,
    gated_lora_modules,
    lora_disabled,
    lora_token_mask,
    routing_state,
)
from qdif.uno.model import fingerprint_backbone
from qdif.uno.tiny import build_tiny_model, random_windows


@pytest.fixture()
def model():
    m = build_tiny_model(seed=1)
    m.attach_adapter(rank=4, full_attention=True, mlp=True)
    return m


def _perturb(model, scale: float = 0.05) -> None:
    """Give the adapter a non-zero effect, as training would."""
    for index, module in enumerate(gated_lora_modules(model.base)):
        module.lora_b = mx.random.normal(module.lora_b.shape, scale=scale).astype(
            mx.float32
        )
    mx.eval(model.base.parameters())


def test_zero_init_adapter_is_a_bit_exact_noop(model):
    ids = random_windows(2, 10, model.vocab_size, seed=3)
    bare = model.ar_logits(ids)
    with_adapter = model.draft_logits(ids, mx.ones(ids.shape, dtype=mx.float32))
    assert mx.array_equal(bare, with_adapter)


def test_disabled_adapter_reproduces_backbone_after_training(model):
    _perturb(model)
    ids = random_windows(2, 10, model.vocab_size, seed=4)
    always_on = model.draft_logits(ids, None)
    bare = model.ar_logits(ids)
    # A trained adapter must actually change something, or the test proves nothing.
    assert not mx.array_equal(always_on, bare)
    with lora_disabled():
        again = model.base(ids)
    assert mx.array_equal(bare, again)


def test_token_mask_gates_exactly_the_selected_rows(model):
    """Rows with mask 0 must match the bare backbone; rows with 1 must not.

    Causality makes this a one-sided check per row: masking row j can only change
    positions >= j. So we gate a suffix and assert the untouched prefix is bit-exact.
    """
    _perturb(model)
    ids = random_windows(1, 12, model.vocab_size, seed=5)
    split = 7
    mask = mx.concatenate(
        [mx.zeros((1, split), dtype=mx.float32), mx.ones((1, 12 - split), dtype=mx.float32)],
        axis=1,
    )
    bare = model.ar_logits(ids)
    gated = model.draft_logits(ids, mask)
    assert mx.array_equal(bare[:, :split], gated[:, :split])
    assert not mx.array_equal(bare[:, split:], gated[:, split:])


def test_all_zero_mask_is_identical_to_disabled(model):
    _perturb(model)
    ids = random_windows(2, 9, model.vocab_size, seed=6)
    zeros = model.draft_logits(ids, mx.zeros(ids.shape, dtype=mx.float32))
    assert mx.array_equal(zeros, model.ar_logits(ids))


def test_mask_shape_mismatch_is_a_hard_error(model):
    _perturb(model)
    ids = random_windows(1, 8, model.vocab_size, seed=7)
    with pytest.raises(ValueError, match="does not match activation"):
        model.draft_logits(ids, mx.ones((1, 5), dtype=mx.float32))


def test_routing_state_is_restored_after_each_context(model):
    ids = random_windows(1, 6, model.vocab_size, seed=8)
    before = routing_state()
    with lora_token_mask(mx.ones(ids.shape, dtype=mx.float32)):
        pass
    with lora_disabled():
        pass
    assert routing_state() == before


def test_routing_state_is_restored_even_when_the_forward_raises(model):
    before = routing_state()
    with pytest.raises(ValueError):
        with lora_token_mask(mx.ones((1, 3), dtype=mx.float32)):
            raise ValueError("boom")
    assert routing_state() == before


def test_only_lora_tensors_are_trainable(model):
    from mlx.utils import tree_flatten

    names = [name for name, _ in tree_flatten(model.base.trainable_parameters())]
    assert names
    assert all(name.endswith(("lora_a", "lora_b")) for name in names)


def test_adapter_targets_match_the_hybrid_architecture(model):
    """Only the 1 full-attention layer has q/k/v/o; all 4 layers have an MLP."""
    injected = model.lora_report.injected
    attention = [name for name in injected if "self_attn" in name]
    mlp = [name for name in injected if ".mlp." in name]
    assert len(attention) == 4  # q, k, v, o on the single full-attention layer
    assert len(mlp) == 12  # gate, up, down on 4 layers
    assert all(name.startswith("layers.3.") for name in attention)


def test_fingerprint_ignores_adapter_and_catches_backbone_edits(model):
    before = fingerprint_backbone(model.base)
    _perturb(model)
    after_adapter_change = fingerprint_backbone(model.base)
    assert after_adapter_change.digest == before.digest, "adapters are not backbone"

    layer = model.base.layers[3].self_attn.q_proj.base
    layer.weight = layer.weight + mx.ones_like(layer.weight) * 1e-3
    mx.eval(model.base.parameters())
    tampered = fingerprint_backbone(model.base)
    assert tampered.digest != before.digest
    assert any("q_proj" in name for name in before.diff(tampered))


def test_gated_lora_rejects_nonpositive_rank():
    import mlx.nn as nn

    with pytest.raises(ValueError, match="rank must be positive"):
        GatedLoRALinear(nn.Linear(4, 4), rank=0, alpha=1.0)


def test_fingerprint_is_invariant_to_adapter_attachment():
    """A wrapped and an unwrapped model must hash identically.

    Attaching the adapter renames `q_proj.weight` to `q_proj.base.weight`. If the raw
    name went into the digest, an adapted model would hash differently from the same
    weights unadapted — indistinguishable, at a glance, from the backbone actually
    having changed. One digest has to be comparable across every run in the project.
    """
    bare = build_tiny_model(seed=61)
    before = fingerprint_backbone(bare.base)

    adapted = build_tiny_model(seed=61)
    adapted.attach_adapter(rank=4, full_attention=True, mlp=True)
    after = fingerprint_backbone(adapted.base)

    assert after.digest == before.digest
    assert after.num_tensors == before.num_tensors
    assert after.num_params == before.num_params
    assert before.diff(after) == []
