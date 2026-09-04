"""Act IV-U — Uno-style diffusion distillation on a frozen autoregressive backbone.

This package is a from-source reproduction of IFM's Uno
(`github.com/ifm-ai/uno`, released 2026-09-03), adapted to Qwen3.5-4B-Base on MLX.
`docs/act4u_uno_source_notes.md` tags every design decision as confirmed-from-IFM,
inferred, or our own approximation. Read it before changing anything here.

It is deliberately independent of `qdif.mlx_backend` (Acts I-IV-N): the backbone here
is never wrapped, never made bidirectional, and never given a custom mask. The
autoregressive path is exercised exactly as the pretrained model defines it.
"""

from __future__ import annotations

__all__ = [
    "GatedLoRALinear",
    "inject_gated_lora",
    "lora_disabled",
    "lora_token_mask",
    "UnoModel",
]


def __getattr__(name):  # lazy: importing mlx is expensive
    if name in ("GatedLoRALinear", "inject_gated_lora", "lora_disabled", "lora_token_mask"):
        from . import gated_lora

        return getattr(gated_lora, name)
    if name == "UnoModel":
        from .model import UnoModel

        return UnoModel
    raise AttributeError(name)
