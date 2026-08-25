"""Iterative block generation for absorbing-state (mask) diffusion.

Different from the Act I/II sampler, which re-corrupted a canvas of *real* tokens.
Under absorbing-state diffusion the canvas starts fully masked and positions are
progressively **unmasked** — once a position is committed it is never re-masked, so
the process is monotone and every step strictly reduces uncertainty:

    x_1 (all [MASK])  ->  x_{t1}  ->  x_{t2}  ->  ...  ->  x_0 (no [MASK])

One step:
    1. forward the current canvas (masked positions visible to the model as [MASK])
    2. read x0 logits at every still-masked position
    3. score confidence (top-1 probability)
    4. unmask the most confident positions, up to the schedule's budget for this step
    5. repeat

Schedule: by step k of K, `round(C * (k+1)/K)` positions are committed, so the final
step commits everything. This is the standard confidence-ordered unmasking used by
masked diffusion LMs; it is deliberately simple and readable rather than tuned.

`generate_trace` retains every intermediate canvas so the user can literally watch a
block resolve. That is a primary educational feature of this repository, not a
debugging aid.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np


@dataclass
class TraceStep:
    step: int
    t: float
    num_masked_before: int
    num_committed_this_step: int
    mean_top1_prob: float
    mean_entropy: float
    canvas_ids: list[int]
    newly_committed: list[int]


@dataclass
class TraceResult:
    prompt: str
    final_ids: list[int]
    steps: list[TraceStep] = field(default_factory=list)
    forwards: int = 0
    seconds: float = 0.0
    canvas_length: int = 0
    peak_unified_gb: float = 0.0

    @property
    def tokens_per_forward(self) -> float:
        return self.canvas_length / max(self.forwards, 1)


def render(tokenizer, ids, mask_token_id: int, mask_str: str = "█") -> str:
    """Render a partially-masked canvas. Masked positions become a block glyph."""
    out = []
    for i in ids:
        i = int(i)
        out.append(mask_str if i == mask_token_id else tokenizer.decode([i]))
    return "".join(out).replace("\n", "\\n")


def denoise_block(
    model,
    prefix_ids: mx.array,
    canvas_length: int,
    mask_token_id: int,
    steps: int = 16,
    temperature: float = 0.0,
    seed: int = 0,
    keep_trace: bool = True,
) -> TraceResult:
    """Denoise one fully-masked block, committing confident positions progressively."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    model.eval()
    B = prefix_ids.shape[0]
    if B != 1:
        raise ValueError("generate-trace renders a single sequence; use batch size 1")

    canvas = np.full((1, canvas_length), mask_token_id, dtype=np.int64)
    committed = np.zeros((1, canvas_length), dtype=bool)
    result = TraceResult(prompt="", final_ids=[], canvas_length=canvas_length)
    t0 = time.time()

    for k in range(steps):
        frac_masked = float((~committed).mean())
        t_now = frac_masked  # noise level fed to the timestep conditioner
        out = model(
            canvas_ids=mx.array(canvas),
            t=mx.array(np.array([t_now], dtype=np.float32)),
            prefix_ids=prefix_ids,
        )
        logits = out.logits.astype(mx.float32)
        # The absorbing token is never a valid prediction.
        logits[:, :, mask_token_id] = -1e9
        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)
        top1 = np.asarray(mx.max(probs, axis=-1))[0]
        pred = np.asarray(mx.argmax(probs, axis=-1).astype(mx.int32))[0]
        ent = np.asarray(
            (-(mx.log(mx.clip(probs, 1e-9, 1.0)) * probs).sum(axis=-1))
        )[0]

        if temperature > 0:
            rng = np.random.default_rng(seed + k)
            p = np.asarray(mx.softmax(logits / temperature, axis=-1))[0]
            pred = np.array([rng.choice(p.shape[-1], p=p[i] / p[i].sum())
                             for i in range(canvas_length)], dtype=np.int64)

        target_committed = int(math.ceil(canvas_length * (k + 1) / steps))
        score = np.where(committed[0], np.inf, top1)  # already-committed keep priority
        order = np.argsort(-score)
        keep = order[:target_committed]

        before = int((~committed).sum())
        new_committed = np.zeros_like(committed)
        new_committed[0, keep] = True
        newly = [int(i) for i in keep if not committed[0, i]]

        canvas[0, newly] = pred[newly]
        committed = new_committed
        canvas[0, ~committed[0]] = mask_token_id  # everything else stays absorbed

        if keep_trace:
            result.steps.append(
                TraceStep(
                    step=k,
                    t=t_now,
                    num_masked_before=before,
                    num_committed_this_step=len(newly),
                    mean_top1_prob=float(top1.mean()),
                    mean_entropy=float(ent.mean()),
                    canvas_ids=[int(v) for v in canvas[0]],
                    newly_committed=newly,
                )
            )
        result.forwards += 1

    result.final_ids = [int(v) for v in canvas[0]]
    result.seconds = time.time() - t0
    result.peak_unified_gb = mx.get_peak_memory() / 1e9
    return result


def ar_generate(model, prefix_ids: mx.array, max_new_tokens: int, mask_token_id: int) -> dict:
    """Greedy AR decode through the same weights, adapters active.

    Deliberately unoptimised: no KV cache, recomputed each step. It exists to show what
    the *same checkpoint* does in its autoregressive mode, NOT as a speed benchmark.
    Never compare these tokens/sec against GGUF/llama.cpp inference.
    """
    model.eval()
    ids = prefix_ids
    t0 = time.time()
    for _ in range(max_new_tokens):
        logits = model.ar_forward(ids).astype(mx.float32)
        logits[:, :, mask_token_id] = -1e9
        nxt = mx.argmax(logits[:, -1], axis=-1)[:, None]
        ids = mx.concatenate([ids, nxt.astype(ids.dtype)], axis=1)
        mx.eval(ids)
    return {
        "ids": [int(v) for v in np.asarray(ids)[0]],
        "new_ids": [int(v) for v in np.asarray(ids)[0][prefix_ids.shape[1]:]],
        "seconds": time.time() - t0,
        "backend": "research MLX python loop, no KV cache -- not a speed benchmark",
    }
