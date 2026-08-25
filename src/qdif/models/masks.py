"""Attention masks for the block-diffusion canvas.

WHAT BIDIRECTIONALITY ACTUALLY MEANS IN QWEN3.5
------------------------------------------------
Qwen3.5's text decoder is a hybrid stack. In the 24-layer 0.8B config, 18 layers are
`linear_attention` (Gated DeltaNet) and 6 are `full_attention`, in a repeating
3:1 pattern with full attention at layers 3, 7, 11, 15, 19, 23.

Only the full-attention layers take an attention mask that we can relax. The Gated
DeltaNet layers are causal *by construction*, not by masking:

  * `Qwen3_5GatedDeltaNet` runs a causal depthwise `conv1d` over the sequence, and
  * `torch_chunk_gated_delta_rule` enforces order with `torch.tril` / `torch.triu`
    inside the chunked recurrence.

Their `attention_mask` argument is a *padding* mask (`create_recurrent_attention_mask`)
whose only job is to zero out padded positions. Passing a bidirectional mask there
does nothing. See docs/QWEN35_NOTES.md.

So `build_diffusion_masks` produces exactly the mask dict that
`Qwen3_5TextModel.forward` accepts:

    {"full_attention": <4-D additive float mask>, "linear_attention": <padding mask|None>}

and the honest description of the resulting model is:

    causal DeltaNet + bidirectional full-attention canvas

Every canvas position can be influenced by every other canvas position, but only
through the 6 full-attention layers. Information cannot flow backwards through the
18 DeltaNet layers. `qdif probe-bidir` measures this empirically.
"""

from __future__ import annotations

import torch


def build_canvas_allow_matrix(
    total_length: int,
    canvas_spans: list[tuple[int, int]],
    bidirectional: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Boolean [T, T] matrix; entry [q, k] is True if query q may attend to key k.

    Starts from the standard lower-triangular causal matrix and then opens up each
    canvas span into a fully-connected block. Prefix positions keep strict causality,
    so a committed prefix is unaffected by whatever noise currently sits in the
    canvas -- which is what makes block-autoregressive generation coherent.

    Args:
        total_length: T, prefix + canvas(es).
        canvas_spans: list of [start, end) index pairs marking diffusion blocks.
        bidirectional: if False, returns a plain causal matrix (the control condition).
    """
    allow = torch.tril(torch.ones(total_length, total_length, dtype=torch.bool, device=device))
    if bidirectional:
        for start, end in canvas_spans:
            if not (0 <= start < end <= total_length):
                raise ValueError(
                    f"canvas span [{start}, {end}) out of range for length {total_length}"
                )
            allow[start:end, start:end] = True
    return allow


def allow_to_additive(
    allow: torch.Tensor,
    dtype: torch.dtype,
    batch_size: int = 1,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert a boolean allow-matrix into the additive [B, 1, T, T] mask that
    eager/SDPA attention expects (0.0 where allowed, dtype-min where blocked).

    Args:
        allow: [T, T] bool.
        padding_mask: [B, T] bool, True = real token. Blocked as keys for every query.
    """
    T = allow.shape[-1]
    neg = torch.finfo(dtype).min
    a = allow[None, None].expand(batch_size, 1, T, T)
    if padding_mask is not None:
        a = a & padding_mask[:, None, None, :].to(allow.device).bool()
    return torch.zeros(a.shape, dtype=dtype, device=allow.device).masked_fill(~a, neg)


def build_diffusion_masks(
    total_length: int,
    canvas_spans: list[tuple[int, int]],
    dtype: torch.dtype,
    batch_size: int = 1,
    bidirectional: bool = True,
    padding_mask: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor | None]:
    """Build the per-layer-type mask dict consumed by `Qwen3_5TextModel.forward`.

    The `linear_attention` entry is intentionally a *padding* mask (or None when
    there is no padding): the DeltaNet layers cannot be made bidirectional by a
    mask, and pretending otherwise would silently produce a causal model dressed up
    as a diffusion model.
    """
    allow = build_canvas_allow_matrix(
        total_length, canvas_spans, bidirectional=bidirectional, device=device
    )
    full = allow_to_additive(allow, dtype, batch_size=batch_size, padding_mask=padding_mask)
    linear = None if padding_mask is None else padding_mask.to(device)
    return {"full_attention": full, "linear_attention": linear}
