"""Forward noising process: rates, determinism, bounds, and mode behaviour."""

from __future__ import annotations

import pytest
import torch

from qdif.diffusion.corruption import corrupt_canvas, random_canvas
from qdif.diffusion.schedule import corruption_probability, noise_bucket, sample_timesteps

VOCAB = 1000


def _x0(b=4, c=16):
    return torch.arange(b * c).reshape(b, c) % VOCAB


def test_t_zero_leaves_canvas_untouched():
    x0 = _x0()
    r = corrupt_canvas(x0, torch.zeros(4), VOCAB, generator=torch.Generator().manual_seed(0))
    assert torch.equal(r.xt, x0)
    assert not r.corrupted_mask.any()
    assert float(r.changed_fraction.mean()) == 0.0


def test_t_one_targets_every_position():
    x0 = _x0()
    r = corrupt_canvas(x0, torch.ones(4), VOCAB, generator=torch.Generator().manual_seed(0))
    assert r.corrupted_mask.all()
    # Every position was targeted, but a uniform replacement can coincide with the
    # original, so changed_fraction is high rather than exactly 1.
    assert float(r.changed_fraction.mean()) > 0.95


def test_corruption_rate_matches_t_statistically():
    """Over many positions the targeted fraction converges on p(t)."""
    x0 = torch.zeros(64, 256, dtype=torch.long)
    gen = torch.Generator().manual_seed(7)
    for t in (0.1, 0.25, 0.5, 0.9):
        r = corrupt_canvas(x0, torch.full((64,), t), VOCAB, generator=gen)
        observed = float(r.corrupted_mask.float().mean())
        assert abs(observed - t) < 0.02, f"t={t} observed={observed}"


def test_seeded_corruption_is_exactly_reproducible():
    x0 = _x0()
    t = torch.full((4,), 0.5)
    a = corrupt_canvas(x0, t, VOCAB, generator=torch.Generator().manual_seed(1234))
    b = corrupt_canvas(x0, t, VOCAB, generator=torch.Generator().manual_seed(1234))
    assert torch.equal(a.xt, b.xt)
    assert torch.equal(a.corrupted_mask, b.corrupted_mask)


def test_different_seeds_give_different_noise():
    x0 = _x0()
    t = torch.full((4,), 0.5)
    a = corrupt_canvas(x0, t, VOCAB, generator=torch.Generator().manual_seed(1))
    b = corrupt_canvas(x0, t, VOCAB, generator=torch.Generator().manual_seed(2))
    assert not torch.equal(a.xt, b.xt)


def test_replacement_tokens_stay_in_vocabulary():
    x0 = _x0()
    r = corrupt_canvas(x0, torch.ones(4), VOCAB, generator=torch.Generator().manual_seed(3))
    assert int(r.xt.min()) >= 0
    assert int(r.xt.max()) < VOCAB


def test_per_example_timesteps_are_independent():
    x0 = torch.zeros(2, 512, dtype=torch.long)
    r = corrupt_canvas(
        x0, torch.tensor([0.0, 1.0]), VOCAB, generator=torch.Generator().manual_seed(5)
    )
    assert not r.corrupted_mask[0].any()
    assert r.corrupted_mask[1].all()


def test_protected_positions_are_never_corrupted():
    x0 = _x0()
    protected = torch.zeros(4, 16, dtype=torch.bool)
    protected[:, :8] = True
    r = corrupt_canvas(
        x0, torch.ones(4), VOCAB, generator=torch.Generator().manual_seed(0), protected_mask=protected
    )
    assert torch.equal(r.xt[:, :8], x0[:, :8])
    assert r.corrupted_mask[:, 8:].all()


def test_mask_mode_uses_the_mask_token():
    x0 = _x0()
    r = corrupt_canvas(
        x0, torch.ones(4), VOCAB, mode="mask", mask_token_id=999,
        generator=torch.Generator().manual_seed(0),
    )
    assert (r.xt == 999).all()


def test_mask_mode_requires_a_token_id():
    with pytest.raises(ValueError, match="mask_token_id"):
        corrupt_canvas(_x0(), torch.ones(4), VOCAB, mode="mask")


@pytest.mark.parametrize("bad_t", [-0.01, 1.01, 5.0])
def test_timesteps_outside_unit_interval_are_rejected(bad_t):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        corrupt_canvas(_x0(), torch.full((4,), bad_t), VOCAB)


def test_rejects_non_2d_canvas():
    with pytest.raises(ValueError, match=r"\[B, C\]"):
        corrupt_canvas(torch.zeros(16, dtype=torch.long), torch.zeros(1), VOCAB)


def test_rejects_float_canvas():
    with pytest.raises(ValueError, match="integer token ids"):
        corrupt_canvas(torch.zeros(2, 8), torch.zeros(2), VOCAB)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown corruption mode"):
        corrupt_canvas(_x0(), torch.zeros(4), VOCAB, mode="gaussian")


# --------------------------------------------------------------------- schedules


def test_sampled_timesteps_respect_bounds():
    t = sample_timesteps(1000, t_min=0.2, t_max=0.8, generator=torch.Generator().manual_seed(0))
    assert t.shape == (1000,)
    assert float(t.min()) >= 0.2
    assert float(t.max()) <= 0.8


def test_uniform_schedule_is_the_identity():
    t = torch.linspace(0, 1, 11)
    assert torch.allclose(corruption_probability(t, "uniform"), t)


def test_cosine_schedule_is_monotone_and_pinned_at_the_ends():
    t = torch.linspace(0, 1, 21)
    p = corruption_probability(t, "cosine")
    assert pytest.approx(float(p[0]), abs=1e-6) == 0.0
    assert pytest.approx(float(p[-1]), abs=1e-6) == 1.0
    assert bool((p[1:] >= p[:-1]).all())


def test_invalid_schedule_bounds_rejected():
    with pytest.raises(ValueError, match="t_min"):
        sample_timesteps(4, t_min=0.8, t_max=0.2)


def test_noise_buckets_partition_the_unit_interval():
    assert noise_bucket(0.0, 5) == "t[0.0,0.2)"
    assert noise_bucket(0.55, 5) == "t[0.4,0.6)"
    # t = 1.0 lands in the top bucket rather than falling off the end.
    assert noise_bucket(1.0, 5) == "t[0.8,1.0)"


def test_random_canvas_shape_and_range():
    c = random_canvas(3, 16, VOCAB, generator=torch.Generator().manual_seed(0))
    assert c.shape == (3, 16)
    assert int(c.min()) >= 0 and int(c.max()) < VOCAB
