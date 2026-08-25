"""Act III: absorbing-state masking, the dual objective, and the health metrics.

Pure-math tests run without a checkpoint. The `model` marked ones need Qwen3.5-4B.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
import numpy as np  # noqa: E402

from qdif.mlx_backend.act3 import (  # noqa: E402
    canvas_conditioning,
    health_metrics,
    mask_corrupt,
    masked_positions_ce,
    resolve_mask_token_id,
)

pytestmark = pytest.mark.mlx

MASK = 999
VOCAB = 1000


def _x0(b=4, c=16):
    return (np.arange(b * c).reshape(b, c) % 500).astype(np.int64)


# ---------------------------------------------------------------- mask corruption


def test_exact_mask_count_matches_requested_fraction():
    rng = np.random.default_rng(0)
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        c = mask_corrupt(_x0(), np.full(4, t, np.float32), MASK, rng, exact_count=True)
        assert c.masked_fraction == pytest.approx(t, abs=1e-6)


def test_m_and_mc_partition_the_block():
    """FLARE's construction: every canvas token is supervised exactly once."""
    c = mask_corrupt(_x0(), np.full(4, 0.5, np.float32), MASK, np.random.default_rng(1))
    a = c.xt == MASK
    b = c.xt_complement == MASK
    assert (a ^ b).all(), "a position must be masked in exactly one of the two views"
    assert not (a & b).any()
    assert (a | b).all()


def test_visible_positions_hold_the_true_token():
    c = mask_corrupt(_x0(), np.full(4, 0.5, np.float32), MASK, np.random.default_rng(2))
    assert (c.xt[~c.mask_set] == c.x0[~c.mask_set]).all()
    assert (c.xt[c.mask_set] == MASK).all()


def test_t_zero_masks_nothing_and_t_one_masks_everything():
    rng = np.random.default_rng(3)
    lo = mask_corrupt(_x0(), np.zeros(4, np.float32), MASK, rng)
    hi = mask_corrupt(_x0(), np.ones(4, np.float32), MASK, rng)
    assert (lo.xt == lo.x0).all()
    assert (hi.xt == MASK).all()
    assert hi.num_visible == 0


def test_bernoulli_mode_is_approximately_right():
    c = mask_corrupt(
        np.zeros((32, 256), dtype=np.int64), np.full(32, 0.3, np.float32), MASK,
        np.random.default_rng(4), exact_count=False,
    )
    assert abs(c.masked_fraction - 0.3) < 0.02


def test_counts_are_reported():
    c = mask_corrupt(_x0(4, 16), np.full(4, 0.25, np.float32), MASK, np.random.default_rng(5))
    assert c.num_masked == 4.0
    assert c.num_visible == 12.0


def test_rejects_out_of_range_fraction():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        mask_corrupt(_x0(), np.full(4, 1.5, np.float32), MASK, np.random.default_rng(0))


# ------------------------------------------------------------------ mask token id


def test_mask_token_is_outside_the_tokenizer_vocabulary():
    """It must be unreachable by the tokenizer, so it cannot collide with real text."""
    mid = resolve_mask_token_id(248320, 248044)
    assert mid == 248319
    assert mid >= 248044


def test_explicit_mask_token_is_honoured_and_range_checked():
    assert resolve_mask_token_id(1000, 900, configured=950) == 950
    with pytest.raises(ValueError, match="outside vocab"):
        resolve_mask_token_id(1000, 900, configured=1000)


def test_refuses_when_there_is_no_spare_row():
    with pytest.raises(ValueError, match="no spare row"):
        resolve_mask_token_id(1000, 1000)


# --------------------------------------------------------------------- objective


def _logits_for(target, b=2, c=8, v=32, magnitude=20.0):
    base = mx.zeros((b, c, v))
    return base + mx.array(np.eye(v, dtype=np.float32)[target] * magnitude)


def test_masked_ce_ignores_unselected_positions():
    x0 = np.random.default_rng(0).integers(0, 32, size=(2, 8))
    good = _logits_for(x0)
    bad = mx.zeros((2, 8, 32))
    sel = mx.array(np.tile([1, 0, 1, 0, 1, 0, 1, 0], (2, 1)).astype(np.float32))
    # correct on selected, garbage elsewhere -> loss must still be ~0
    mixed = mx.where(mx.array(np.tile([1, 0, 1, 0, 1, 0, 1, 0], (2, 1)).astype(bool))[..., None],
                     good, bad)
    assert float(masked_positions_ce(mixed, mx.array(x0), sel)) < 1e-3


def test_masked_ce_with_no_selection_is_finite():
    x0 = np.random.default_rng(0).integers(0, 32, size=(2, 8))
    out = masked_positions_ce(mx.random.normal((2, 8, 32)), mx.array(x0), mx.zeros((2, 8)))
    assert np.isfinite(float(out))


# ----------------------------------------------------------------- health metrics


def test_perfect_denoiser_scores_one_on_masked_positions():
    # x0 must never contain the mask id -- that invariant holds in reality because
    # the absorbing token is outside the tokenizer's vocabulary.
    x0 = np.random.default_rng(0).integers(0, 31, size=(2, 8))
    c = mask_corrupt(x0, np.full(2, 0.5, np.float32), 31, np.random.default_rng(1))
    m = health_metrics(_logits_for(c.x0), c, 31)
    assert m["masked_accuracy"] == 1.0
    assert m["visible_preservation"] == 1.0
    assert m["predicts_mask_rate"] == 0.0


def test_metrics_expose_a_model_that_only_echoes_visible_tokens():
    """The Act III analogue of copy collapse: reproduce the input verbatim, which at a
    masked position means emitting the mask token -- always wrong."""
    x0 = np.random.default_rng(0).integers(0, 31, size=(2, 8))
    c = mask_corrupt(x0, np.full(2, 0.5, np.float32), 31, np.random.default_rng(1))
    m = health_metrics(_logits_for(c.xt), c, 31)
    assert m["copy_rate"] == 1.0
    assert m["masked_accuracy"] == 0.0
    assert m["visible_preservation"] == 1.0
    assert m["predicts_mask_rate"] > 0.0


def test_visible_preservation_is_separate_from_masked_accuracy():
    """These must not be conflated: Act III passes one and fails the other."""
    x0 = np.random.default_rng(0).integers(0, 31, size=(2, 8))
    c = mask_corrupt(x0, np.full(2, 0.5, np.float32), 31, np.random.default_rng(1))
    # correct at masked positions, wrong at visible ones
    pred = np.where(c.mask_set, c.x0, (c.x0 + 1) % 31)
    m = health_metrics(_logits_for(pred), c, 31)
    assert m["masked_accuracy"] == 1.0
    assert m["visible_preservation"] == 0.0


# ------------------------------------------------------ canvas-conditioning probe


@pytest.mark.model
def test_canvas_conditioning_detects_a_model_that_ignores_the_canvas(monkeypatch):
    """A model whose output does not depend on xt must score ~0."""

    class Frozen:
        """Returns the same logits regardless of the canvas."""

        def __init__(self, shape):
            self.shape = shape

        def __call__(self, canvas_ids, t, prefix_ids):
            class Out:
                logits = mx.zeros(self.shape)

            return Out()

    x0 = np.random.default_rng(0).integers(0, 32, size=(2, 8))
    c = mask_corrupt(x0, np.full(2, 0.5, np.float32), 31, np.random.default_rng(1))
    m = Frozen((2, 8, 32))
    out = canvas_conditioning(m, mx.zeros((2, 4), dtype=mx.int32), c,
                              np.random.default_rng(2), 32)
    assert out["canvas_l1"] == pytest.approx(0.0, abs=1e-6)
    assert out["canvas_js"] == pytest.approx(0.0, abs=1e-6)
