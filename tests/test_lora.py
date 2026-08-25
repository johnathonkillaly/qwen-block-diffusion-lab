"""LoRA mechanics on a stand-in module tree -- no checkpoint required."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from qdif.training.lora import (
    TARGET_PRESETS,
    LoRALinear,
    freeze_base,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
    set_adapters_enabled,
)


class FakeAttention(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)


class FakeMLP(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.gate_proj = nn.Linear(h, h, bias=False)
        self.up_proj = nn.Linear(h, h, bias=False)
        self.down_proj = nn.Linear(h, h, bias=False)


class FakeLayer(nn.Module):
    """Mirrors Qwen3.5's dispatch: full-attention layers own `self_attn`,
    linear-attention layers own `linear_attn`."""

    def __init__(self, block_type: str, h=32):
        super().__init__()
        self.block_type = block_type
        if block_type == "full_attention":
            self.self_attn = FakeAttention(h)
        else:
            self.linear_attn = nn.Module()
            self.linear_attn.in_proj_qkv = nn.Linear(h, 3 * h, bias=False)
            self.linear_attn.out_proj = nn.Linear(h, h, bias=False)
        self.mlp = FakeMLP(h)


class FakeModel(nn.Module):
    """8 layers in Qwen3.5's 3:1 pattern -> full attention at indices 3 and 7."""

    def __init__(self, h=32):
        super().__init__()
        types = ["linear_attention"] * 8
        for i in (3, 7):
            types[i] = "full_attention"
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(FakeLayer(t, h) for t in types)
        self.lm_head = nn.Linear(h, 100, bias=False)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return FakeModel()


def test_injection_hits_only_full_attention_layers_with_the_default_preset(model):
    report = inject_lora(model, rank=4, alpha=8, target_preset="full_attn")
    assert report.num_injected == 8  # 2 layers x q/k/v/o
    assert all("self_attn" in name for name in report.injected)
    assert all(".layers.3." in n or ".layers.7." in n for n in report.injected)


def test_layer_index_filter_restricts_injection(model):
    report = inject_lora(model, rank=4, target_preset="full_attn", layer_indices=[3])
    assert report.num_injected == 4
    assert all(".layers.3." in n for n in report.injected)


def test_untrained_adapter_is_an_exact_no_op(model):
    x = torch.randn(2, 5, 32)
    before = model.model.layers[3].self_attn.q_proj(x)
    inject_lora(model, rank=4, target_preset="full_attn")
    after = model.model.layers[3].self_attn.q_proj(x)
    assert torch.allclose(before, after, atol=1e-6), "lora_B is zero-init: output must not move"


def test_adapter_changes_output_once_b_is_nonzero(model):
    inject_lora(model, rank=4, target_preset="full_attn")
    wrapper = model.model.layers[3].self_attn.q_proj
    x = torch.randn(2, 5, 32)
    before = wrapper(x)
    with torch.no_grad():
        wrapper.lora_B.normal_(std=0.1)
    assert not torch.allclose(before, wrapper(x), atol=1e-6)


def test_freeze_base_leaves_only_lora_trainable(model):
    inject_lora(model, rank=4, target_preset="full_attn")
    freeze_base(model)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all(".lora_A" in n or ".lora_B" in n for n in trainable)


def test_gradients_reach_lora_and_not_the_base(model):
    inject_lora(model, rank=4, target_preset="full_attn")
    freeze_base(model)
    wrapper = model.model.layers[3].self_attn.q_proj
    with torch.no_grad():
        wrapper.lora_B.normal_(std=0.1)  # otherwise dL/dA is exactly zero at init

    wrapper(torch.randn(2, 5, 32)).sum().backward()
    assert wrapper.lora_A.grad is not None and float(wrapper.lora_A.grad.abs().sum()) > 0
    assert wrapper.lora_B.grad is not None and float(wrapper.lora_B.grad.abs().sum()) > 0
    assert wrapper.base.weight.grad is None


def test_lora_a_gradient_is_zero_at_initialisation(model):
    """Documented consequence of zero-initialising B -- not a bug."""
    inject_lora(model, rank=4, target_preset="full_attn")
    freeze_base(model)
    wrapper = model.model.layers[3].self_attn.q_proj
    wrapper(torch.randn(2, 5, 32)).sum().backward()
    assert float(wrapper.lora_A.grad.abs().sum()) == 0.0
    assert float(wrapper.lora_B.grad.abs().sum()) > 0.0


def test_base_weights_are_unchanged_by_an_optimizer_step(model):
    inject_lora(model, rank=4, target_preset="full_attn")
    freeze_base(model)
    wrapper = model.model.layers[3].self_attn.q_proj
    snapshot = wrapper.base.weight.detach().clone()
    with torch.no_grad():
        wrapper.lora_B.normal_(std=0.1)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.1)
    wrapper(torch.randn(2, 5, 32)).sum().backward()
    opt.step()
    assert torch.equal(wrapper.base.weight, snapshot)


def test_disabling_adapters_restores_base_behaviour(model):
    inject_lora(model, rank=4, target_preset="full_attn")
    wrapper = model.model.layers[3].self_attn.q_proj
    x = torch.randn(2, 5, 32)
    baseline = wrapper.base(x)
    with torch.no_grad():
        wrapper.lora_B.normal_(std=0.5)

    assert not torch.allclose(wrapper(x), baseline, atol=1e-5)
    n = set_adapters_enabled(model, False)
    assert n == 8
    assert torch.allclose(wrapper(x), baseline, atol=1e-6)
    set_adapters_enabled(model, True)
    assert not torch.allclose(wrapper(x), baseline, atol=1e-5)


def test_state_dict_round_trip(model):
    inject_lora(model, rank=4, target_preset="full_attn")
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(std=0.1)
    saved = lora_state_dict(model)
    assert saved and all("lora_" in k for k in saved)

    fresh = FakeModel()
    inject_lora(fresh, rank=4, target_preset="full_attn")
    assert load_lora_state_dict(fresh, saved) == len(saved)
    x = torch.randn(2, 5, 32)
    # Base weights differ between the two models, so compare the adapter delta only.
    a = model.model.layers[3].self_attn.q_proj
    b = fresh.model.layers[3].self_attn.q_proj
    assert torch.allclose(a(x) - a.base(x), b(x) - b.base(x), atol=1e-6)


def test_state_dict_contains_no_base_weights(model):
    inject_lora(model, rank=4, target_preset="full_attn")
    assert not any("base.weight" in k for k in lora_state_dict(model))


def test_scaling_follows_alpha_over_rank(model):
    inject_lora(model, rank=8, alpha=32, target_preset="full_attn")
    assert model.model.layers[3].self_attn.q_proj.scaling == pytest.approx(4.0)


@pytest.mark.parametrize("preset", sorted(TARGET_PRESETS))
def test_every_preset_matches_something(preset, model):
    report = inject_lora(FakeModel(), rank=2, target_preset=preset)
    assert report.num_injected > 0


def test_unknown_preset_rejected(model):
    with pytest.raises(ValueError, match="unknown lora.target_preset"):
        inject_lora(model, target_preset="nope")


def test_custom_preset_requires_targets(model):
    with pytest.raises(ValueError, match="target_modules"):
        inject_lora(model, target_preset="custom")


def test_no_match_raises_with_a_useful_message(model):
    with pytest.raises(RuntimeError, match="arch-report"):
        inject_lora(model, target_preset="custom", target_modules=["does.not.exist"])
