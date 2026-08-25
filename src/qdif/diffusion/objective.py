"""The x0-prediction training objective.

READOUT ALIGNMENT — the single most important convention in this repository.

An autoregressive LM is trained so that the logits at position i predict the token
at position i+1. A discrete-diffusion denoiser is trained so that the logits at
position i predict the *clean token at position i*, given the noisy token that was
actually fed in at position i.

    AR        : logits[:, i] -> x[i + 1]        (shift by one)
    diffusion : logits[:, i] -> x0[i]           (no shift)

We deliberately do NOT shift. That mismatch is precisely the thing LoRA has to
learn to undo, and it has a concrete, checkable consequence:

    An *unadapted* Qwen3.5 evaluated under this objective at t = 0 will score near
    0% identity accuracy and high next-token accuracy, because it is still doing
    its old job. That is the expected starting point, not a bug. See
    `diffusion_metrics(...)`, which reports both numbers so the distinction is
    always visible in the logs.

Loss: plain token-level cross entropy over canvas positions.
`loss_on` selects which positions contribute:
    "all"       -- every canvas position (default; matches DiffusionGemma-style x0
                   prediction and gives gradient signal even at t = 0)
    "corrupted" -- only positions the forward process actually changed (a BERT-like
                   variant; zero valid positions at t = 0, so guarded)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

LOSS_REDUCTIONS = ("all", "corrupted")


@dataclass
class DiffusionLossOutput:
    loss: torch.Tensor  # scalar, differentiable
    per_position_loss: torch.Tensor  # [B, C] detached
    metrics: dict[str, float] = field(default_factory=dict)


def diffusion_loss(
    logits: torch.Tensor,
    x0: torch.Tensor,
    corrupted_mask: torch.Tensor | None = None,
    loss_on: str = "all",
    valid_mask: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> DiffusionLossOutput:
    """Cross entropy between canvas logits and the clean tokens.

    Args:
        logits: [B, C, V] logits aligned to canvas positions (no shift, see module docstring).
        x0: [B, C] clean token ids.
        corrupted_mask: [B, C] bool, positions the forward process targeted.
        loss_on: "all" | "corrupted".
        valid_mask: [B, C] bool, False marks padding that must not contribute.
        label_smoothing: passed through to cross_entropy.

    Returns:
        DiffusionLossOutput with a scalar `loss` suitable for `.backward()`.
    """
    if loss_on not in LOSS_REDUCTIONS:
        raise ValueError(f"loss_on must be one of {LOSS_REDUCTIONS}, got {loss_on!r}")
    if logits.ndim != 3:
        raise ValueError(f"logits must be [B, C, V], got {tuple(logits.shape)}")
    if logits.shape[:2] != x0.shape:
        raise ValueError(
            f"logits {tuple(logits.shape[:2])} and x0 {tuple(x0.shape)} disagree on [B, C]"
        )

    B, C, V = logits.shape
    # Cross entropy in fp32 regardless of the model's compute dtype: bf16 logsumexp
    # over a 248k vocabulary loses enough precision to distort the reported loss.
    flat = logits.reshape(B * C, V).float()
    per_position = F.cross_entropy(
        flat, x0.reshape(B * C), reduction="none", label_smoothing=label_smoothing
    ).view(B, C)

    weight = torch.ones_like(per_position, dtype=torch.float32)
    if valid_mask is not None:
        weight = weight * valid_mask.to(per_position.device).float()
    if loss_on == "corrupted":
        if corrupted_mask is None:
            raise ValueError("loss_on='corrupted' requires corrupted_mask")
        weight = weight * corrupted_mask.to(per_position.device).float()

    denom = weight.sum()
    if denom == 0:
        # Happens legitimately at t = 0 with loss_on="corrupted". Return a zero that
        # is still connected to the graph so optimizer.step() stays well-defined.
        loss = (per_position * weight).sum()
    else:
        loss = (per_position * weight).sum() / denom

    return DiffusionLossOutput(
        loss=loss,
        per_position_loss=per_position.detach(),
        metrics={"loss_positions": float(denom.item())},
    )


@torch.no_grad()
def diffusion_metrics(
    logits: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    corrupted_mask: torch.Tensor,
) -> dict[str, float]:
    """Diagnostics that distinguish 'the model learned to denoise' from 'the model
    is still an AR next-token predictor' and from 'the model just copies its input'.

    Returns:
        identity_accuracy      -- P(argmax logits[i] == x0[i]) over all canvas positions.
                                  This is the number that must go up.
        corrupted_accuracy     -- same, restricted to positions that were corrupted.
                                  The genuinely hard subset.
        clean_accuracy         -- same, restricted to positions that were left alone.
                                  Reaching ~100% here only proves copying.
        copy_rate              -- P(argmax logits[i] == xt[i]). A model that has
                                  collapsed to "echo my input" shows copy_rate ~ 1
                                  with corrupted_accuracy ~ 0.
        next_token_accuracy    -- P(argmax logits[i] == x0[i+1]). High for an
                                  unadapted AR model; should fall as adaptation
                                  takes hold.
        mean_top1_prob         -- mean confidence, used by the adaptive sampler.
        mean_entropy           -- mean predictive entropy in nats.
    """
    pred = logits.argmax(dim=-1)  # [B, C]
    correct = (pred == x0).float()

    corrupted = corrupted_mask.to(pred.device).bool()
    clean = ~corrupted

    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> float:
        n = m.sum()
        return float((x * m.float()).sum() / n) if n > 0 else float("nan")

    probs = logits.float().softmax(dim=-1)
    top1 = probs.max(dim=-1).values
    entropy = -(probs.clamp_min(1e-9).log() * probs).sum(dim=-1)

    out = {
        "identity_accuracy": float(correct.mean()),
        "corrupted_accuracy": _masked_mean(correct, corrupted),
        "clean_accuracy": _masked_mean(correct, clean),
        "copy_rate": float((pred == xt).float().mean()),
        "mean_top1_prob": float(top1.mean()),
        "mean_entropy": float(entropy.mean()),
    }

    if x0.shape[1] > 1:
        out["next_token_accuracy"] = float((pred[:, :-1] == x0[:, 1:]).float().mean())
    else:
        out["next_token_accuracy"] = float("nan")
    return out
