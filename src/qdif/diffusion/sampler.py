"""Iterative denoising sampler.

Deliberately simple and readable before anything clever. One step is:

    1. model predicts clean-token logits for every canvas position
    2. score each position's confidence (top-1 probability)
    3. commit the most confident uncommitted positions (schedule below)
    4. re-corrupt the still-uncommitted positions at the next, lower noise level
    5. repeat

Commit schedule: by step k of K, ceil(C * (k+1)/K) positions are frozen, so the
last step commits everything. With K=1 the sampler degenerates to a single
argmax pass over the whole canvas -- the cheapest possible "diffusion" decode and
a useful lower bound.

The metric worth watching is `tokens_per_forward` = canvas_length / K, the whole
reason to be interested in diffusion decoding at all. It is recorded on every
`SamplerResult`.

`strategy="all"` rewrites every position each step (no commitment) as a control:
if that performs as well as confidence-based commitment, the commitment machinery
is not earning its keep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .corruption import corrupt_canvas, random_canvas

STRATEGIES = ("confidence", "all")


@dataclass
class SamplerStepRecord:
    step: int
    t: float
    num_committed: int
    mean_top1_prob: float
    mean_entropy: float
    num_changed: float


@dataclass
class SamplerResult:
    tokens: torch.Tensor  # [B, C] final canvas
    logits: torch.Tensor  # [B, C, V] final-step logits
    steps_run: int
    forwards: int
    canvas_length: int
    history: list[SamplerStepRecord] = field(default_factory=list)
    stopped_early: bool = False

    @property
    def tokens_per_forward(self) -> float:
        return self.canvas_length / max(self.forwards, 1)


@torch.no_grad()
def denoise(
    model,
    canvas_length: int,
    prefix_ids: torch.Tensor | None = None,
    steps: int = 8,
    batch_size: int = 1,
    strategy: str = "confidence",
    temperature: float = 0.0,
    adaptive: bool = False,
    confidence_threshold: float = 0.95,
    corruption: str = "uniform",
    schedule: str = "uniform",
    mask_token_id: int | None = None,
    generator: torch.Generator | None = None,
    init_canvas: torch.Tensor | None = None,
    self_conditioning: bool = False,
    record_history: bool = True,
) -> SamplerResult:
    """Denoise one block from pure noise (or from `init_canvas`).

    Args:
        model: a `DiffusionQwen`.
        steps: number of denoising iterations K (>= 1).
        strategy: "confidence" (progressive commitment) or "all" (rewrite everything).
        temperature: 0 -> argmax; > 0 -> sample from the softened distribution.
        adaptive: stop once mean top-1 probability exceeds `confidence_threshold`.
        init_canvas: [B, C] starting tokens; defaults to a t=1 random canvas.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")

    device = model.device
    vocab_size = model.config.vocab_size

    if init_canvas is None:
        x = random_canvas(
            batch_size,
            canvas_length,
            vocab_size,
            mode=corruption,
            mask_token_id=mask_token_id,
            generator=generator,
            device=device,
        )
    else:
        x = init_canvas.to(device)
        batch_size, canvas_length = x.shape

    if prefix_ids is not None and prefix_ids.shape[0] != batch_size:
        prefix_ids = prefix_ids.expand(batch_size, -1)

    committed = torch.zeros(batch_size, canvas_length, dtype=torch.bool, device=device)
    history: list[SamplerStepRecord] = []
    prev_logits: torch.Tensor | None = None
    logits = None
    forwards = 0
    stopped_early = False
    step = 0

    for step in range(steps):
        # Noise level fed to the timestep conditioner: 1 at the first step, 0 at the last.
        t_now = 1.0 - step / steps
        t_vec = torch.full((batch_size,), t_now, device=device, dtype=torch.float32)

        out = model(
            canvas_ids=x,
            t=t_vec,
            prefix_ids=prefix_ids,
            self_cond_logits=prev_logits if self_conditioning else None,
        )
        forwards += 1
        logits = out.logits
        if self_conditioning:
            prev_logits = logits.detach()

        probs = logits.float().softmax(dim=-1)
        top1, pred = probs.max(dim=-1)  # [B, C]
        if temperature > 0:
            # Sampling uses torch's global RNG: `generator` is a CPU generator kept
            # for reproducible *corruption*, and multinomial needs it on `device`.
            pred = torch.multinomial(
                (logits.float() / temperature).softmax(dim=-1).reshape(-1, vocab_size),
                num_samples=1,
            ).view(batch_size, canvas_length)
        entropy = -(probs.clamp_min(1e-9).log() * probs).sum(dim=-1)

        # Apply the prediction everywhere that is not already committed.
        new_x = torch.where(committed, x, pred)
        num_changed = float((new_x != x).float().sum(dim=-1).mean())
        x = new_x

        if strategy == "confidence":
            target_committed = math.ceil(canvas_length * (step + 1) / steps)
            # Rank by confidence; already-committed positions keep top priority.
            score = top1.masked_fill(committed, float("inf"))
            order = score.argsort(dim=-1, descending=True)
            keep = order[:, :target_committed]
            committed = torch.zeros_like(committed).scatter_(1, keep, True)
        elif step == steps - 1:
            # "all": nothing is frozen until the final pass, which commits everything.
            committed = torch.ones_like(committed)

        if record_history:
            history.append(
                SamplerStepRecord(
                    step=step,
                    t=t_now,
                    num_committed=int(committed[0].sum()),
                    mean_top1_prob=float(top1.mean()),
                    mean_entropy=float(entropy.mean()),
                    num_changed=num_changed,
                )
            )

        if adaptive and float(top1.mean()) >= confidence_threshold:
            stopped_early = True
            break

        # Re-corrupt the uncommitted remainder at the next (lower) noise level, so the
        # next forward pass sees a canvas that is genuinely noisier where the model is
        # unsure. On the last step there is nothing left to renoise.
        t_next = 1.0 - (step + 1) / steps
        if step < steps - 1 and t_next > 0:
            renoised = corrupt_canvas(
                x.cpu(),
                torch.full((batch_size,), t_next),
                vocab_size=vocab_size,
                mode=corruption,
                schedule=schedule,
                mask_token_id=mask_token_id,
                generator=generator,
                protected_mask=committed.cpu(),
            )
            x = renoised.xt.to(device)

    return SamplerResult(
        tokens=x,
        logits=logits,
        steps_run=step + 1,
        forwards=forwards,
        canvas_length=canvas_length,
        history=history,
        stopped_early=stopped_early,
    )


@torch.no_grad()
def block_generate(
    model,
    prompt_ids: torch.Tensor,
    canvas_length: int,
    num_blocks: int = 1,
    **sampler_kwargs,
) -> tuple[torch.Tensor, list[SamplerResult]]:
    """Block-autoregressive generation: denoise a block, commit it, extend the prefix.

    NOTE ON CACHING. Each block currently re-runs the full prefix. A KV / DeltaNet
    state cache would be valid here -- the committed prefix is causal and frozen --
    but the recurrent state has to be rolled back whenever a canvas is rewritten, and
    getting that wrong silently corrupts results. The harness therefore recomputes.
    Cost is O(blocks * prefix), which is irrelevant at the canvas sizes used for
    correctness work and is the first thing to optimise afterwards. See
    docs/ARCHITECTURE.md.
    """
    prefix = prompt_ids.to(model.device)
    if prefix.ndim == 1:
        prefix = prefix[None]
    results: list[SamplerResult] = []
    for _ in range(num_blocks):
        res = denoise(model, canvas_length=canvas_length, prefix_ids=prefix, **sampler_kwargs)
        results.append(res)
        prefix = torch.cat([prefix, res.tokens], dim=1)
    return prefix, results
