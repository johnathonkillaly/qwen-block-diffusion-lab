"""Diffusion timestep schedules.

`t` is a continuous noise level in [0, 1]:
    t = 0  -> canvas is exactly x0 (clean)
    t = 1  -> every canvas position is resampled from the corruption distribution

We keep `t` and "corruption probability" identical under the uniform schedule so
that the diagnostics stay interpretable; `cosine` remaps t before it is used as a
probability, and that remapping is the only thing that differs.
"""

from __future__ import annotations

import math

import torch

SCHEDULES = ("uniform", "cosine")


def sample_timesteps(
    batch_size: int,
    schedule: str = "uniform",
    t_min: float = 0.0,
    t_max: float = 1.0,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Draw one timestep per batch element, shape [B], dtype float32, in [t_min, t_max]."""
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}, expected one of {SCHEDULES}")
    if not 0.0 <= t_min <= t_max <= 1.0:
        raise ValueError(f"need 0 <= t_min <= t_max <= 1, got {t_min=} {t_max=}")
    u = torch.rand(batch_size, generator=generator, device="cpu", dtype=torch.float32)
    t = t_min + u * (t_max - t_min)
    return t.to(device)


def corruption_probability(t: torch.Tensor, schedule: str = "uniform") -> torch.Tensor:
    """Map a timestep to the per-position probability of replacing a token.

    uniform: p = t (identity — the interpretable default)
    cosine:  p = 1 - cos(pi/2 * t)^2 = sin(pi/2 * t)^2, which spends more steps at
             low noise. Monotone, p(0)=0, p(1)=1.
    """
    if schedule == "uniform":
        return t.clamp(0.0, 1.0)
    if schedule == "cosine":
        return torch.sin(t.clamp(0.0, 1.0) * (math.pi / 2)) ** 2
    raise ValueError(f"unknown schedule {schedule!r}")


def noise_bucket(t: float, num_buckets: int = 5) -> str:
    """Human-readable bucket label used to group reconstruction accuracy in logs."""
    idx = min(int(t * num_buckets), num_buckets - 1)
    lo = idx / num_buckets
    hi = (idx + 1) / num_buckets
    return f"t[{lo:.1f},{hi:.1f})"
