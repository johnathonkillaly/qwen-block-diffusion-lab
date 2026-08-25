"""`DiffusionQwen` -- a block-discrete-diffusion head on a Qwen3.5 text decoder.

Design constraints this class exists to satisfy:

1. The base checkpoint is used unmodified. No monkeypatching of transformers, no
   rewriting of attention modules. The only hook is the documented mask-dict
   argument of `Qwen3_5TextModel.forward`.
2. The autoregressive path must survive intact, so that AR-vs-diffusion comparison
   is a comparison of the *same weights*. `ar_forward()` and `no_diffusion()` give
   that, and at initialisation the diffusion path is numerically identical to the
   AR path (auxiliary modules are zero-initialised).
3. Diffusion math lives in `qdif.diffusion`, not here. This file only handles
   sequence layout, embedding injection, masks, and readout.

SEQUENCE LAYOUT
    [ prefix (clean, causal) | canvas (noisy, bidirectional within itself) ]
      0 .. P-1                 P .. P+C-1

READOUT
    logits[:, i] predicts x0[i] for canvas position i -- no shift. See
    `qdif.diffusion.objective` for why, and for the diagnostics that make the
    difference from AR next-token prediction visible.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import DiffusionConfig
from ..diffusion.self_conditioning import SelfConditioner, TimestepConditioner
from .masks import build_diffusion_masks


@dataclass
class DiffusionForwardOutput:
    logits: torch.Tensor  # [B, C, V] over canvas positions only
    hidden_states: torch.Tensor  # [B, C, H] canvas hidden states
    canvas_start: int
    total_length: int


class DiffusionQwen(nn.Module):
    """Wraps a `Qwen3_5ForCausalLM` with a diffusion forward path."""

    def __init__(self, base, cfg: DiffusionConfig):
        super().__init__()
        self.base = base
        self.cfg = cfg
        hidden = base.config.hidden_size
        param_dtype = next(base.parameters()).dtype

        self.timestep_conditioner: TimestepConditioner | None = None
        if cfg.timestep_conditioning:
            self.timestep_conditioner = TimestepConditioner(hidden).to(param_dtype)

        self.self_conditioner: SelfConditioner | None = None
        if cfg.self_conditioning:
            self.self_conditioner = SelfConditioner(hidden).to(param_dtype)

        self._adapters_enabled = True

    # ------------------------------------------------------------------ helpers

    @property
    def config(self):
        return self.base.config

    @property
    def device(self) -> torch.device:
        return next(self.base.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.base.parameters()).dtype

    @property
    def embedding_matrix(self) -> torch.Tensor:
        return self.base.get_input_embeddings().weight

    def auxiliary_modules(self) -> dict[str, nn.Module]:
        """Diffusion-specific modules that are *not* part of the base checkpoint."""
        mods: dict[str, nn.Module] = {}
        if self.timestep_conditioner is not None:
            mods["timestep_conditioner"] = self.timestep_conditioner
        if self.self_conditioner is not None:
            mods["self_conditioner"] = self.self_conditioner
        return mods

    @contextlib.contextmanager
    def no_diffusion(self):
        """Temporarily disable LoRA adapters (if any), exposing the pristine AR model.

        Used by the AR control: `with model.no_diffusion(): model.ar_forward(...)`.
        """
        from ..training.lora import set_adapters_enabled

        previous = self._adapters_enabled
        set_adapters_enabled(self.base, False)
        self._adapters_enabled = False
        try:
            yield self
        finally:
            set_adapters_enabled(self.base, previous)
            self._adapters_enabled = previous

    # ------------------------------------------------------------------ forwards

    def ar_forward(self, input_ids: torch.Tensor, **kwargs):
        """The untouched autoregressive path (causal mask, next-token logits)."""
        return self.base(input_ids=input_ids, use_cache=False, **kwargs)

    def forward(
        self,
        canvas_ids: torch.Tensor,
        t: torch.Tensor,
        prefix_ids: torch.Tensor | None = None,
        self_cond_logits: torch.Tensor | None = None,
        bidirectional: bool | None = None,
        committed_spans: list[tuple[int, int]] | None = None,
    ) -> DiffusionForwardOutput:
        """Denoise a canvas.

        Args:
            canvas_ids: [B, C] noisy token ids to be denoised.
            t: [B] noise levels in [0, 1] used for timestep conditioning.
            prefix_ids: [B, P] clean conditioning tokens, or None for an unconditional
                canvas.
            self_cond_logits: [B, C, V] logits from the previous denoising iteration.
                Ignored unless self-conditioning is enabled.
            bidirectional: override `cfg.bidirectional_canvas` (used by the causal
                control run).
            committed_spans: canvas-relative [start, end) spans already finalised.
                They stay inside the canvas block but are otherwise treated normally;
                supplied for future block-level mask refinements.

        Returns:
            DiffusionForwardOutput with logits over canvas positions only.
        """
        del committed_spans  # accepted for interface stability; not used yet
        if canvas_ids.ndim != 2:
            raise ValueError(f"canvas_ids must be [B, C], got {tuple(canvas_ids.shape)}")

        device, dtype = self.device, self.dtype
        canvas_ids = canvas_ids.to(device)
        B, C = canvas_ids.shape

        if prefix_ids is not None and prefix_ids.numel() > 0:
            prefix_ids = prefix_ids.to(device)
            if prefix_ids.shape[0] != B:
                raise ValueError("prefix_ids and canvas_ids disagree on batch size")
            input_ids = torch.cat([prefix_ids, canvas_ids], dim=1)
            P = prefix_ids.shape[1]
        else:
            input_ids = canvas_ids
            P = 0
        T = input_ids.shape[1]

        inputs_embeds = self.base.get_input_embeddings()(input_ids)

        # --- inject conditioning at canvas positions only -----------------------
        if self.timestep_conditioner is not None:
            bias = self.timestep_conditioner(t.to(device).reshape(-1), dtype)  # [B,1,H]
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[:, P:] = inputs_embeds[:, P:] + bias

        if self.self_conditioner is not None and self_cond_logits is not None:
            delta = self.self_conditioner(
                self_cond_logits.to(device), self.embedding_matrix
            ).to(dtype)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[:, P:] = inputs_embeds[:, P:] + delta

        # --- masks --------------------------------------------------------------
        bidi = self.cfg.bidirectional_canvas if bidirectional is None else bidirectional
        masks = build_diffusion_masks(
            total_length=T,
            canvas_spans=[(P, T)],
            dtype=dtype,
            batch_size=B,
            bidirectional=bidi,
            device=device,
        )

        outputs = self.base.model(
            inputs_embeds=inputs_embeds,
            attention_mask=masks,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state[:, P:]  # [B, C, H]
        logits = self.base.lm_head(hidden)  # [B, C, V]

        return DiffusionForwardOutput(
            logits=logits, hidden_states=hidden, canvas_start=P, total_length=T
        )


def parameter_report(model: DiffusionQwen) -> dict:
    """Total / trainable / LoRA / auxiliary parameter counts and dtype byte sizes."""
    total = 0
    trainable = 0
    lora = 0
    aux = 0
    bytes_total = 0
    aux_names = {f"{k}." for k in model.auxiliary_modules()}

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        bytes_total += n * p.element_size()
        if p.requires_grad:
            trainable += n
        if ".lora_A" in name or ".lora_B" in name:
            lora += n
        elif any(name.startswith(a) for a in aux_names):
            aux += n

    return {
        "total_params": total,
        "trainable_params": trainable,
        "lora_params": lora,
        "auxiliary_params": aux,
        "frozen_params": total - trainable,
        "percent_trainable": 100.0 * trainable / total if total else 0.0,
        "param_bytes": bytes_total,
        "param_gb": bytes_total / 1e9,
    }
