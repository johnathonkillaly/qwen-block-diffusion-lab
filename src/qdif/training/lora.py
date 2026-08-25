"""Minimal LoRA for Qwen3.5.

Written by hand rather than pulled from `peft` for three reasons specific to this
experiment: we need to target modules by *layer index* (to test "adapt only the
full-attention layers"), we need a runtime enable/disable switch to run the AR
control on identical weights, and we need adapters to survive on a hybrid
architecture whose module names peft does not know about.

y = W x + (alpha / r) * B(A(x)),  with A ~ N(0, init_scale), B = 0.

B is zero-initialised so an untrained adapter is an exact no-op: the wrapped model
reproduces the base model bit-for-bit until the first optimizer step.

TARGET PRESETS
--------------
Qwen3.5's decoder exposes these Linear families (0.8B, 24 layers):

  full-attention layers (6 of 24, indices 3/7/11/15/19/23)
      self_attn.q_proj  self_attn.k_proj  self_attn.v_proj  self_attn.o_proj
  Gated DeltaNet layers (18 of 24)
      linear_attn.in_proj_qkv   (q, k, v fused)
      linear_attn.in_proj_z     (output gate)
      linear_attn.in_proj_b     (beta / write strength)
      linear_attn.in_proj_a     (decay / forget gate)
      linear_attn.out_proj
  every layer
      mlp.gate_proj  mlp.up_proj  mlp.down_proj

`full_attn` is the deliberate conservative default: the full-attention layers are
the only place where bidirectional canvas information exists at all (see
`qdif.models.masks`), so they are the first hypothesis to test. The other presets
exist to answer research question 2 ("which layer families require adaptation?").
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
import torch.nn as nn

TARGET_PRESETS: dict[str, list[str]] = {
    # Only the periodic full-attention layers' projections.
    "full_attn": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
    # Full attention + the MLP that follows it.
    "full_attn_mlp": [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
    # DeltaNet input/output projections only (tests whether the recurrent path can
    # be reshaped without touching full attention).
    "deltanet": ["linear_attn.in_proj_qkv", "linear_attn.out_proj"],
    # DeltaNet including its gates (beta/decay control what the recurrent state keeps).
    "deltanet_gates": [
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_a",
        "linear_attn.out_proj",
    ],
    # Both token mixers, no MLP.
    "all_attn": [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "linear_attn.in_proj_qkv",
        "linear_attn.out_proj",
    ],
    # Everything linear in the decoder. Included for completeness; not recommended
    # as a first experiment.
    "all_linear": [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_a",
        "linear_attn.out_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
}

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` with a trainable low-rank update."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
        init_scale: float = 0.01,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.enabled = True

        # Adapters are held in fp32 even when the backbone is bf16: they are tiny,
        # and low-rank updates accumulate badly in bf16 under Adam.
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=torch.float32))
        nn.init.normal_(self.lora_A, std=init_scale)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        h = self.dropout(x).to(self.lora_A.dtype)
        update = torch.nn.functional.linear(torch.nn.functional.linear(h, self.lora_A), self.lora_B)
        return out + (update * self.scaling).to(out.dtype)

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}"


@dataclass
class LoraInjectionReport:
    target_suffixes: list[str]
    layer_indices: list[int] | None
    injected: list[str]
    skipped_no_match: int
    lora_params: int
    rank: int
    alpha: int

    @property
    def num_injected(self) -> int:
        return len(self.injected)


def _resolve_targets(preset: str, custom: list[str]) -> list[str]:
    if preset == "custom":
        if not custom:
            raise ValueError("lora.target_preset='custom' requires lora.target_modules")
        return list(custom)
    if preset not in TARGET_PRESETS:
        raise ValueError(
            f"unknown lora.target_preset {preset!r}; known: {sorted(TARGET_PRESETS)} + 'custom'"
        )
    return list(TARGET_PRESETS[preset])


def inject_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    target_preset: str = "full_attn",
    target_modules: list[str] | None = None,
    layer_indices: list[int] | None = None,
    init_scale: float = 0.01,
) -> LoraInjectionReport:
    """Replace matching `nn.Linear` modules in-place with `LoRALinear`.

    Args:
        layer_indices: restrict to these decoder layer indices. None/empty = all layers.
                       Modules outside any `.layers.<i>.` path are only adapted when
                       this is empty.
    """
    targets = _resolve_targets(target_preset, target_modules or [])
    layer_filter = set(layer_indices) if layer_indices else None

    to_replace: list[tuple[str, nn.Linear]] = []
    skipped = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in targets):
            continue
        if layer_filter is not None:
            m = _LAYER_RE.search("." + name)
            if m is None or int(m.group(1)) not in layer_filter:
                skipped += 1
                continue
        to_replace.append((name, module))

    injected: list[str] = []
    lora_params = 0
    for name, module in to_replace:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapper = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout, init_scale=init_scale)
        setattr(parent, attr, wrapper)
        injected.append(name)
        lora_params += wrapper.lora_A.numel() + wrapper.lora_B.numel()

    if not injected:
        raise RuntimeError(
            f"LoRA injection matched no modules. targets={targets} "
            f"layer_indices={sorted(layer_filter) if layer_filter else 'all'}. "
            f"Run `qdif arch-report` to list the module names this checkpoint exposes."
        )

    return LoraInjectionReport(
        target_suffixes=targets,
        layer_indices=sorted(layer_filter) if layer_filter else None,
        injected=injected,
        skipped_no_match=skipped,
        lora_params=lora_params,
        rank=rank,
        alpha=alpha,
    )


def freeze_base(model: nn.Module) -> int:
    """Freeze every parameter that is not a LoRA factor. Returns #frozen params."""
    frozen = 0
    for name, p in model.named_parameters():
        if ".lora_A" in name or ".lora_B" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
            frozen += p.numel()
    return frozen


def set_adapters_enabled(model: nn.Module, enabled: bool) -> int:
    """Flip every LoRALinear on/off. Returns the number of adapters touched."""
    n = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled
            n += 1
    return n


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter tensors -- never base weights."""
    return {
        name: p.detach().cpu()
        for name, p in model.named_parameters()
        if ".lora_A" in name or ".lora_B" in name
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> int:
    own = dict(model.named_parameters())
    missing = [k for k in state if k not in own]
    if missing:
        raise KeyError(f"adapter checkpoint has keys absent from the model: {missing[:5]}")
    loaded = 0
    with torch.no_grad():
        for name, tensor in state.items():
            own[name].copy_(tensor.to(own[name].device, own[name].dtype))
            loaded += 1
    return loaded


def estimate_lora_params(
    hidden_size: int, targets: list[str], shapes: dict[str, list[int]], rank: int, count: int
) -> int:
    """Rough parameter count for a target set, for reports before injection."""
    total = 0
    for suffix in targets:
        io = shapes.get(suffix)
        if io is None:
            continue
        total += rank * (io[0] + io[1]) * count
    return total or rank * 2 * hidden_size * count * len(targets)


def sqrt_scaling(rank: int) -> float:
    """rsLoRA-style scaling, offered for rank-sweep experiments (research question 5)."""
    return 1.0 / math.sqrt(rank)
