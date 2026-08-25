"""LoRA for the MLX backend.

Written against `mlx.nn.Linear` for the same reasons the torch version was written
by hand: we need targeting by *layer index* (to adapt only full-attention layers, or
only DeltaNet layers), and a runtime enable/disable switch so the AR control runs on
identical weights.

`lora_b` is zero-initialised, so an untrained adapter is an exact no-op and the
wrapped model reproduces the base model bit-for-bit until the first optimizer step.

IMPORTANT for v0.2: a DeltaNet adapter is used by BOTH the forward and the reverse
pass, because both call the same `base` submodules. That is the intended semantics --
"the same pretrained recurrence, evaluated twice" includes the adaptation. A
separate reverse-direction adapter is a different (larger) hypothesis and is gated
behind `bidirectional_deltanet.train_reverse_lora`, which is not implemented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

#: Module attribute names, grouped by family. Qwen3.5 uses identical names at every
#: scale, so these resolve on 0.8B, 4B and 27B alike.
FULL_ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
DELTANET_TARGETS = ("in_proj_qkv", "out_proj")
DELTANET_GATE_TARGETS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B(A(x)), base frozen, B zero-initialised."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0,
                 init_scale: float = 0.01):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        out_features, in_features = base.weight.shape
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.enabled = True
        # Adapters in fp32: they are tiny and low-rank updates accumulate badly in
        # bf16 under Adam.
        self.lora_a = mx.random.normal((in_features, rank), scale=init_scale).astype(mx.float32)
        self.lora_b = mx.zeros((rank, out_features), dtype=mx.float32)
        self.dropout = dropout

    def __call__(self, x: mx.array) -> mx.array:
        out = self.base(x)
        if not self.enabled:
            return out
        h = x.astype(mx.float32)
        if self.dropout > 0 and self.training:
            h = nn.Dropout(self.dropout)(h)
        update = (h @ self.lora_a) @ self.lora_b
        return out + (update * self.scaling).astype(out.dtype)


@dataclass
class LoraReport:
    injected: list[str]
    lora_params: int
    rank: int
    alpha: int
    families: list[str]

    @property
    def num_injected(self) -> int:
        return len(self.injected)


def _wrap(parent, attr: str, rank: int, alpha: int, dropout: float, init_scale: float):
    base = getattr(parent, attr, None)
    if not isinstance(base, nn.Linear) or isinstance(base, LoRALinear):
        return 0
    layer = LoRALinear(base, rank, alpha, dropout, init_scale)
    setattr(parent, attr, layer)
    return layer.lora_a.size + layer.lora_b.size


def inject_lora(
    model,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    init_scale: float = 0.01,
    full_attention: bool = True,
    deltanet: bool = False,
    deltanet_gates: bool = False,
    mlp: bool = False,
    layer_indices: list[int] | None = None,
) -> LoraReport:
    """Attach LoRA to the selected module families, in place.

    Returns a report naming every adapted module -- v0.2 explicitly requires knowing
    exactly which modules are trainable rather than adapting every linear layer.
    """
    selected = set(layer_indices) if layer_indices else None
    injected: list[str] = []
    total = 0
    families: list[str] = []

    for i, layer in enumerate(model.layers):
        if selected is not None and i not in selected:
            continue
        is_linear = getattr(layer, "is_linear", False)

        if not is_linear and full_attention and hasattr(layer, "self_attn"):
            for attr in FULL_ATTENTION_TARGETS:
                n = _wrap(layer.self_attn, attr, rank, alpha, dropout, init_scale)
                if n:
                    injected.append(f"layers.{i}.self_attn.{attr}")
                    total += n

        if is_linear and (deltanet or deltanet_gates):
            # The bidirectional wrapper keeps the original module at `.base`.
            attn = layer.linear_attn
            target_parent = getattr(attn, "base", attn)
            attrs = DELTANET_GATE_TARGETS if deltanet_gates else DELTANET_TARGETS
            for attr in attrs:
                n = _wrap(target_parent, attr, rank, alpha, dropout, init_scale)
                if n:
                    injected.append(f"layers.{i}.linear_attn.{attr}")
                    total += n

        if mlp and hasattr(layer, "mlp"):
            for attr in MLP_TARGETS:
                n = _wrap(layer.mlp, attr, rank, alpha, dropout, init_scale)
                if n:
                    injected.append(f"layers.{i}.mlp.{attr}")
                    total += n

    for name, flag in (("full_attention", full_attention), ("deltanet", deltanet),
                       ("deltanet_gates", deltanet_gates), ("mlp", mlp)):
        if flag:
            families.append(name)

    if not injected:
        raise RuntimeError(
            f"LoRA injection matched no modules (families={families}, "
            f"layer_indices={sorted(selected) if selected else 'all'}). "
            f"Run `qdif arch-report --backend mlx` to list what this model exposes."
        )
    return LoraReport(injected=injected, lora_params=total, rank=rank, alpha=alpha,
                      families=families)


def lora_modules(model) -> list[LoRALinear]:
    out = []

    def walk(m):
        if isinstance(m, LoRALinear):
            out.append(m)
        for child in getattr(m, "children", lambda: {})().values():
            if isinstance(child, nn.Module):
                walk(child)
            elif isinstance(child, (list, tuple)):
                for c in child:
                    if isinstance(c, nn.Module):
                        walk(c)

    walk(model)
    return out


def set_adapters_enabled(model, enabled: bool) -> int:
    mods = lora_modules(model)
    for m in mods:
        m.enabled = enabled
    return len(mods)


def configure_trainable(model, train_fusion: bool = True, train_conditioner: bool = True) -> dict:
    """Freeze the pretrained backbone; unfreeze only adapters and new modules.

    MLX expresses trainability by freezing modules, and `nn.value_and_grad` then
    differentiates only w.r.t. `model.trainable_parameters()`. Freezing everything
    first and unfreezing a named allowlist means a frozen base weight cannot receive
    a gradient even in principle -- there is no tensor for one to land on.

    Returns a report of what was unfrozen.
    """
    from .deltanet import BidirectionalGatedDeltaNet

    model.freeze(recurse=True)

    unfrozen: list[str] = []
    lora_params = 0
    fusion_params = 0
    cond_params = 0

    for m in lora_modules(model):
        m.unfreeze(recurse=False, keys=["lora_a", "lora_b"])
        lora_params += m.lora_a.size + m.lora_b.size
    if lora_modules(model):
        unfrozen.append(f"lora x{len(lora_modules(model))}")

    if train_fusion:
        n = 0
        for _, mod in _bidi_modules(model, BidirectionalGatedDeltaNet):
            p = mod.fusion.num_parameters
            if p:
                mod.fusion.unfreeze(recurse=True)
                fusion_params += p
                n += 1
        if n:
            unfrozen.append(f"fusion x{n}")

    if train_conditioner and getattr(model, "timestep_conditioner", None) is not None:
        model.timestep_conditioner.unfreeze(recurse=True)
        cond_params = count_parameters(model.timestep_conditioner.parameters())
        unfrozen.append("timestep_conditioner")

    return {
        "unfrozen": unfrozen,
        "lora_params": lora_params,
        "fusion_params": fusion_params,
        "conditioner_params": cond_params,
        "trainable_params": count_parameters(model.trainable_parameters()),
    }


def _bidi_modules(model, cls):
    layers = model.layers if hasattr(model, "layers") else model.base_model.layers
    out = []
    for i, layer in enumerate(layers):
        mod = getattr(layer, "linear_attn", None)
        if isinstance(mod, cls):
            out.append((i, mod))
    return out


def count_parameters(tree) -> int:
    return sum(v.size for _, v in tree_flatten(tree))


def rslora_scaling(rank: int) -> float:
    return 1.0 / math.sqrt(rank)
