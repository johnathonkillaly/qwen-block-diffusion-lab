"""Act III: FLARE-inspired AR + diffusion transfer.

This is an independent implementation of the *methodological ideas* described in

    FLARE: Diffusion for Hybrid Language Model
    Zhu, Shi, Ge, Tan, Xu, Zhu, Kuen, Goswami, Jain, Chen, Tao, Gu
    arXiv:2606.01774v2   https://arxiv.org/abs/2606.01774

written from the paper's mathematical specification against this repository's own
harness. No source was copied from the FLARE reference implementation (which is
PolyForm Noncommercial 1.0.0). Deviations are documented in docs/ACT3_OBJECTIVE.md
and THIRD_PARTY.md.

WHY ACT III EXISTS
------------------
Acts I and II trained a *pure* diffusion objective under uniform random-token
corruption. Phase 3 of Act II ended in a degenerate regime: held-out masked-position
accuracy was ~6-7% and, damningly, **almost independent of the noise level**, with
copy rate ~3.5%. That is the signature of a model that has stopped reading the canvas
and emits a generic prefix-conditioned guess.

Act III changes three things at once, deliberately, because FLARE's recipe is a
package and testing it piecemeal is what Act II already did:

  1. a combined objective  L_total = L_AR + lambda * L_diff,
  2. absorbing-state MASK corruption instead of uniform random-token replacement, so
     corrupted positions are *identifiable* rather than having to be detected, and
  3. complementary mask views, so every canvas token contributes exactly one AR
     signal and one diffusion signal per step.

THE OBJECTIVE
-------------
FLARE Eq. (4)-(5), specialised to our single-block canvas layout:

    L_AR   = mean_{l} -log p(x_l | x_<l)                over the clean causal stream
    L_diff = mean over the block partition of
               -log p(x_l | x~_M,   x_<b)   for l in M
               -log p(x_l | x~_Mc,  x_<b)   for l in M^c
    L_total = L_AR + lambda_diff * L_diff

`M` and `M^c` partition the canvas, so the union of the two noisy views supervises
every canvas position exactly once -- FLARE's "every token contributes one AR signal
and one diffusion signal at unit weight".

Normalisation note: FLARE writes unweighted *sums*. Because they pack L tokens and
partition every block, their AR and diffusion terms cover the same L tokens, i.e. a
1:1 token balance. Our single-block layout has P+C-1 AR targets and C diffusion
targets, so we take a per-token **mean** of each term, which restores exactly that
1:1 balance. `lambda_diff` is then the explicit knob, default 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .ops import cross_entropy


# --------------------------------------------------------------- mask corruption


@dataclass
class MaskedCanvas:
    """One absorbing-state corruption draw, plus the bookkeeping the logs need."""

    xt: np.ndarray  # [B, C] canvas with M replaced by the mask token
    xt_complement: np.ndarray  # [B, C] canvas with M^c replaced instead
    x0: np.ndarray  # [B, C] clean canvas
    mask_set: np.ndarray  # [B, C] bool, True = position is in M
    t: np.ndarray  # [B] requested mask fraction

    @property
    def masked_fraction(self) -> float:
        return float(self.mask_set.mean())

    @property
    def num_masked(self) -> float:
        return float(self.mask_set.sum(axis=-1).mean())

    @property
    def num_visible(self) -> float:
        return float((~self.mask_set).sum(axis=-1).mean())


def mask_corrupt(
    x0: np.ndarray,
    t: np.ndarray,
    mask_token_id: int,
    rng: np.random.Generator,
    exact_count: bool = True,
) -> MaskedCanvas:
    """Absorbing-state corruption: selected positions become the mask token.

    Unlike uniform random-token corruption (Acts I-II), the model can *see* which
    positions need repair. Act II experiment 001 showed uniform corruption makes the
    task ill-posed -- the model must detect and repair simultaneously, and copying
    scores (1-t) for free.

    Args:
        x0: [B, C] clean canvas.
        t: [B] requested mask fraction in [0, 1].
        exact_count: if True, mask exactly round(t*C) positions per example (lower
            variance, and it makes `masked_fraction` match `t` closely enough that
            the noise-bucket diagnostics stay interpretable). If False, sample each
            position independently as a Bernoulli(t).
    """
    x0 = np.asarray(x0)
    if x0.ndim != 2:
        raise ValueError(f"x0 must be [B, C], got {x0.shape}")
    B, C = x0.shape
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    if t.size == 1:
        t = np.repeat(t, B)
    if (t < 0).any() or (t > 1).any():
        raise ValueError("mask fraction must lie in [0, 1]")

    mask_set = np.zeros((B, C), dtype=bool)
    if exact_count:
        for i in range(B):
            k = int(round(float(t[i]) * C))
            if k > 0:
                mask_set[i, rng.choice(C, size=k, replace=False)] = True
    else:
        mask_set = rng.random((B, C)).astype(np.float32) < t[:, None]

    xt = np.where(mask_set, mask_token_id, x0)
    xt_c = np.where(~mask_set, mask_token_id, x0)
    return MaskedCanvas(xt=xt, xt_complement=xt_c, x0=x0, mask_set=mask_set, t=t)


def resolve_mask_token_id(model_vocab_size: int, tokenizer_vocab_size: int,
                          configured: int | None = None) -> int:
    """Pick an absorbing-state token that can never appear in real text.

    Qwen3.5's model vocabulary (248,320) is larger than the tokenizer's (248,044).
    The trailing embedding rows are untrained padding -- on Qwen3.5-4B-Base every row
    from ~248,070 upward shares the same norm (0.445587), the signature of rows that
    were initialised and never updated. Using the last row gives an absorbing token
    that (a) the tokenizer can never emit, so it cannot collide with real data, and
    (b) carries no pretrained semantics the model must unlearn.
    """
    if configured is not None:
        if not (0 <= configured < model_vocab_size):
            raise ValueError(f"mask_token_id {configured} outside vocab {model_vocab_size}")
        return configured
    if model_vocab_size <= tokenizer_vocab_size:
        raise ValueError(
            "model vocab is not larger than the tokenizer vocab, so there is no spare "
            "row for an absorbing mask token. Set diffusion.mask_token_id explicitly."
        )
    return model_vocab_size - 1


# ------------------------------------------------------------------- the objective


@dataclass
class Act3LossOutput:
    total: mx.array
    loss_ar: mx.array
    loss_diff: mx.array
    metrics: dict = field(default_factory=dict)


def ar_loss(model, ids: mx.array) -> mx.array:
    """Clean causal stream: standard next-token prediction over the whole sequence.

    This is the term that keeps Qwen doing its pretrained job while the diffusion
    term teaches same-position reconstruction. Acts I-II omitted it, and the AR path
    degraded catastrophically (reference perplexity 8.06 -> 115.87).
    """
    logits = model.ar_forward(ids)
    return cross_entropy(logits[:, :-1], ids[:, 1:])


def masked_positions_ce(logits: mx.array, x0: mx.array, sel: mx.array) -> mx.array:
    """Cross entropy restricted to selected canvas positions.

    Args:
        logits: [B, C, V] canvas logits (unshifted readout: logits[i] -> x0[i]).
        x0: [B, C] clean targets.
        sel: [B, C] float 0/1 selector.
    """
    logits = logits.astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    picked = mx.take_along_axis(logprobs, x0[..., None].astype(mx.int32), axis=-1).squeeze(-1)
    denom = mx.maximum(sel.sum(), 1.0)
    return -(picked * sel).sum() / denom


def act3_loss(
    model,
    prefix_ids: mx.array,
    canvas: MaskedCanvas,
    lambda_diff: float = 1.0,
    complementary_views: bool = True,
    ar_weight: float = 1.0,
) -> Act3LossOutput:
    """L_total = ar_weight * L_AR + lambda_diff * L_diff.

    Forward passes per step:
        1  clean causal   -> L_AR
        2  noisy view M   -> L_diff on M
        3  noisy view M^c -> L_diff on M^c        (when complementary_views)

    With complementary views every canvas position is supervised exactly once per
    step. Disabling them halves the diffusion cost and supervises only M; over many
    steps that is statistically similar, but it is not FLARE's construction.
    """
    x0 = mx.array(canvas.x0)
    ids = mx.concatenate([prefix_ids, x0], axis=1)
    t = mx.array(canvas.t)

    l_ar = ar_loss(model, ids)

    sel_m = mx.array(canvas.mask_set.astype(np.float32))
    out_m = model(canvas_ids=mx.array(canvas.xt), t=t, prefix_ids=prefix_ids)
    ce_m = masked_positions_ce(out_m.logits, x0, sel_m)

    if complementary_views:
        sel_c = 1.0 - sel_m
        out_c = model(canvas_ids=mx.array(canvas.xt_complement), t=t, prefix_ids=prefix_ids)
        ce_c = masked_positions_ce(out_c.logits, x0, sel_c)
        n_m = float(canvas.mask_set.sum())
        n_c = float((~canvas.mask_set).sum())
        # Weight the two views by how many positions each supervises, so the result
        # is a per-canvas-token mean over the partition rather than an average of
        # two differently-sized means.
        total_n = max(n_m + n_c, 1.0)
        l_diff = (ce_m * n_m + ce_c * n_c) / total_n
    else:
        l_diff = ce_m

    total = ar_weight * l_ar + lambda_diff * l_diff
    return Act3LossOutput(
        total=total,
        loss_ar=l_ar,
        loss_diff=l_diff,
        metrics={
            "requested_mask_fraction": float(canvas.t.mean()),
            "actual_masked_fraction": canvas.masked_fraction,
            "num_masked": canvas.num_masked,
            "num_visible": canvas.num_visible,
        },
    )


# -------------------------------------------------------------- health metrics


def health_metrics(
    logits: mx.array, canvas: MaskedCanvas, mask_token_id: int
) -> dict[str, float]:
    """The Act III health gate.

    The Act II Phase 3 pathology was masked-position accuracy ~6-7% *flat across all
    noise levels*, with the model ignoring the canvas. These metrics are chosen so
    that pathology cannot be mistaken for progress:

        masked_accuracy      accuracy at positions the model must reconstruct
        visible_preservation accuracy at positions it can simply read off. A healthy
                             denoiser copies these nearly perfectly; Act II scored
                             ~7% here, i.e. it was destroying known-correct tokens.
        identity_accuracy    overall
        copy_rate            P(prediction == input token). Under MASK corruption a
                             high copy rate at masked positions means predicting the
                             mask token, which is always wrong.
        predicts_mask_rate   how often the model emits the absorbing token. Should
                             go to ~0; it is never a valid target.
    """
    pred = np.asarray(mx.argmax(logits, axis=-1).astype(mx.int32))
    x0, xt, m = canvas.x0, canvas.xt, canvas.mask_set
    correct = pred == x0
    vis = ~m

    def mean_where(a, sel):
        return float(a[sel].mean()) if sel.any() else float("nan")

    return {
        "masked_accuracy": mean_where(correct, m),
        "visible_preservation": mean_where(correct, vis),
        "identity_accuracy": float(correct.mean()),
        "copy_rate": float((pred == xt).mean()),
        "predicts_mask_rate": float((pred == mask_token_id).mean()),
        "exact_reconstruction": float(correct.all(axis=-1).mean()),
    }


def next_token_accuracy(model, ids: mx.array) -> float:
    """AR health: does the clean causal stream still predict the next token?"""
    logits = model.ar_forward(ids)
    pred = np.asarray(mx.argmax(logits[:, :-1], axis=-1).astype(mx.int32))
    return float((pred == np.asarray(ids[:, 1:])).mean())


def ar_reference_perplexity(model, ids: mx.array) -> float:
    """Teacher-forced perplexity on a FIXED reference set.

    Act II established that self-perplexity is degenerate when the adapted model's
    output degenerates, so AR health is always measured on fixed text.
    """
    logits = model.ar_forward(ids)
    return float(mx.exp(cross_entropy(logits[:, :-1], ids[:, 1:])))


# ------------------------------------------------------ canvas-conditioning probe


def canvas_conditioning(
    model,
    prefix_ids: mx.array,
    canvas: MaskedCanvas,
    rng: np.random.Generator,
    vocab_size: int,
) -> dict[str, float]:
    """Does the model actually read `xt`? A first-class metric, not a nicety.

    Method: hold the prefix, the clean target and the mask set FIXED, then replace
    the *visible* (unmasked) canvas tokens with random tokens. A model that uses the
    canvas must change its predictions at the masked positions; a model that has
    collapsed to a generic prefix-conditioned guess will not move at all.

    Returns mean L1 distance and Jensen-Shannon divergence between the two predictive
    distributions, averaged over masked positions. Both are 0 when the model ignores
    the canvas entirely.
    """
    m = canvas.mask_set
    alt = rng.integers(0, vocab_size, size=canvas.x0.shape, dtype=np.int64)
    xt_a = canvas.xt
    xt_b = np.where(m, canvas.xt, alt)  # masked positions identical, visible scrambled

    t = mx.array(canvas.t)
    pa = mx.softmax(
        model(canvas_ids=mx.array(xt_a), t=t, prefix_ids=prefix_ids).logits.astype(mx.float32),
        axis=-1,
    )
    pb = mx.softmax(
        model(canvas_ids=mx.array(xt_b), t=t, prefix_ids=prefix_ids).logits.astype(mx.float32),
        axis=-1,
    )
    mx.eval(pa, pb)

    l1 = mx.abs(pa - pb).sum(axis=-1)  # [B, C], in [0, 2]
    mid = 0.5 * (pa + pb)

    def kl(p, q):
        return (p * (mx.log(mx.clip(p, 1e-9, 1.0)) - mx.log(mx.clip(q, 1e-9, 1.0)))).sum(axis=-1)

    js = 0.5 * kl(pa, mid) + 0.5 * kl(pb, mid)
    sel = mx.array(m.astype(np.float32))
    n = mx.maximum(sel.sum(), 1.0)
    return {
        "canvas_l1": float((l1 * sel).sum() / n),
        "canvas_js": float((js * sel).sum() / n),
    }
