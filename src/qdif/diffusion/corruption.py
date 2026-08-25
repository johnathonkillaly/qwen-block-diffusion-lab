"""Forward (noising) process for discrete token diffusion.

Given a clean canvas `x0` of token ids and a noise level `t`, produce `xt` by
independently corrupting each position:

    u_i ~ Uniform(0, 1)
    xt_i = replacement_i   if u_i < p(t)
           x0_i            otherwise

Two corruption distributions are supported:

  uniform  -- replacement_i ~ Uniform{0, ..., vocab_size-1}. This is the default
              for this project: the canvas always holds real tokens, so the model
              never sees an out-of-distribution symbol, and t=1 is a genuine
              "no information left" condition.
  mask     -- replacement_i = mask_token_id (absorbing state). Offered as a
              separate experimental mode only; it changes the input distribution
              in a way the pretrained model has never seen.

Determinism: every function takes an optional `torch.Generator`. With a seeded
generator the exact set of corrupted positions and the exact replacement tokens
are reproducible, which is what `tests/test_corruption.py` pins down.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .schedule import corruption_probability

CORRUPTION_MODES = ("uniform", "mask")


@dataclass
class CorruptionResult:
    """Everything the loss and the diagnostics need about one noising draw."""

    xt: torch.Tensor  # [B, C] corrupted canvas
    x0: torch.Tensor  # [B, C] clean canvas (unchanged, carried for convenience)
    corrupted_mask: torch.Tensor  # [B, C] bool: positions the process *targeted*
    t: torch.Tensor  # [B] float32 noise level actually used

    @property
    def targeted_fraction(self) -> torch.Tensor:
        """Fraction of positions the process chose to replace, per example. [B]"""
        return self.corrupted_mask.float().mean(dim=-1)

    @property
    def changed_fraction(self) -> torch.Tensor:
        """Fraction of positions whose token id actually differs from x0. [B]

        Strictly <= targeted_fraction under uniform corruption, because a random
        replacement can coincidentally equal the original token. Reported
        separately so noise-level diagnostics are not silently biased.
        """
        return (self.xt != self.x0).float().mean(dim=-1)


def corrupt_canvas(
    x0: torch.Tensor,
    t: torch.Tensor,
    vocab_size: int,
    mode: str = "uniform",
    schedule: str = "uniform",
    mask_token_id: int | None = None,
    generator: torch.Generator | None = None,
    protected_mask: torch.Tensor | None = None,
) -> CorruptionResult:
    """Corrupt a clean canvas.

    Args:
        x0: [B, C] int64 clean token ids.
        t: [B] float noise levels in [0, 1], or a 0-dim tensor broadcast to all.
        vocab_size: size of the sampling range for `uniform` replacement.
        mode: "uniform" | "mask".
        schedule: passed to `corruption_probability` to turn t into a probability.
        mask_token_id: required when mode == "mask".
        generator: optional seeded RNG (kept on CPU so results are device-independent).
        protected_mask: [B, C] bool; True positions are never corrupted (e.g. padding
            or a committed block during block-autoregressive sampling).

    Returns:
        CorruptionResult
    """
    if mode not in CORRUPTION_MODES:
        raise ValueError(f"unknown corruption mode {mode!r}, expected one of {CORRUPTION_MODES}")
    if x0.ndim != 2:
        raise ValueError(f"x0 must be [B, C], got shape {tuple(x0.shape)}")
    if x0.dtype not in (torch.int64, torch.int32):
        raise ValueError(f"x0 must hold integer token ids, got dtype {x0.dtype}")

    B, C = x0.shape
    t = t.reshape(-1).float()
    if t.numel() == 1:
        t = t.expand(B).clone()
    if t.numel() != B:
        raise ValueError(f"t must have {B} entries (or 1), got {t.numel()}")
    if bool((t < 0).any() or (t > 1).any()):
        raise ValueError("timesteps must lie in [0, 1]")

    p = corruption_probability(t.cpu(), schedule).view(B, 1)  # [B, 1]

    # RNG is done on CPU so that a given seed reproduces bit-identically on
    # CPU / MPS / CUDA. The canvas is tiny; this is not a bottleneck.
    u = torch.rand(B, C, generator=generator, dtype=torch.float32)
    corrupted = u < p  # [B, C] bool

    if protected_mask is not None:
        corrupted = corrupted & ~protected_mask.cpu().bool()

    if mode == "uniform":
        replacement = torch.randint(
            low=0, high=vocab_size, size=(B, C), generator=generator, dtype=torch.int64
        )
    else:
        if mask_token_id is None:
            raise ValueError("mode='mask' requires mask_token_id")
        replacement = torch.full((B, C), int(mask_token_id), dtype=torch.int64)

    corrupted = corrupted.to(x0.device)
    replacement = replacement.to(x0.device)
    xt = torch.where(corrupted, replacement, x0)

    return CorruptionResult(xt=xt, x0=x0, corrupted_mask=corrupted, t=t.to(x0.device))


def random_canvas(
    batch_size: int,
    canvas_length: int,
    vocab_size: int,
    mode: str = "uniform",
    mask_token_id: int | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Draw a fully-noised canvas, i.e. the t = 1 state the sampler starts from."""
    if mode == "mask":
        if mask_token_id is None:
            raise ValueError("mode='mask' requires mask_token_id")
        out = torch.full((batch_size, canvas_length), int(mask_token_id), dtype=torch.int64)
    else:
        out = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, canvas_length),
            generator=generator,
            dtype=torch.int64,
        )
    return out.to(device)
