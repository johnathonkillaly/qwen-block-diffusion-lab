"""Act IV-U5: making block size the *only* difference between two training arms.

U5 compares adapters trained at K=4, K=6 and K=8 and asks whether the longer training
horizon produces a better K=4 decoder. That comparison is only worth making if the arms
differ in their horizon and in nothing else. Two things had to change for that to be
true, and each is pinned here:

1. **Corruption alignment.** The draft rows for block size `L` end at absolute position
   `W-2` for every `L`, so arms overlap on their rightmost positions. Drawing noise of
   shape `(B, L-1)` gives each arm a different realisation even under one key. Aligned
   drawing makes the shared positions bit-identical.
2. **Curriculum staging on a resumed run.** Stages must divide the *resumed span*, not
   the absolute step axis, or a curriculum arm resumed near the end of its schedule
   trains entirely in its final stage and is not a curriculum at all.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from qdif.uno.rng import TRAIN_NOISE, stream_key
from qdif.uno.teacher import build_uno_batch
from qdif.uno.tiny import random_windows
from qdif.uno.trainer import UnoTrainConfig

ALIGN = 8


def _batch(block_size, step=0, align=ALIGN, corruption="uniform", width=32):
    windows = random_windows(4, width, 400, seed=101)
    return build_uno_batch(
        windows, block_size, 480, 512, corruption=corruption,
        key=stream_key(4242, TRAIN_NOISE, step), noise_align_width=align,
    )


# ------------------------------------------------------- corruption alignment


@pytest.mark.parametrize("corruption", ["uniform", "full"])
def test_arms_share_the_noise_on_the_positions_they_share(corruption):
    """K=4's three draft rows must hold exactly K=8's last three."""
    small = _batch(4, corruption=corruption)
    large = _batch(8, corruption=corruption)

    # both blocks end at the same absolute position, so compare from the right
    shared = 3  # K=4 has 3 draft rows
    assert mx.array_equal(
        small.student_ids[:, -shared:], large.student_ids[:, -shared:]
    ), "shared draft rows differ between arms — the horizon is not the only variable"
    assert mx.array_equal(small.corrupted[:, -shared:], large.corrupted[:, -shared:])


def test_alignment_is_what_makes_them_share():
    """Non-vacuity: without alignment the same key gives different realisations."""
    windows = random_windows(4, 32, 400, seed=101)
    key = stream_key(4242, TRAIN_NOISE, 0)
    small = build_uno_batch(windows, 4, 480, 512, corruption="uniform", key=key)
    large = build_uno_batch(windows, 8, 480, 512, corruption="uniform", key=key)
    assert not mx.array_equal(small.student_ids[:, -3:], large.student_ids[:, -3:])


def test_all_three_u5_horizons_agree_on_their_common_suffix():
    four, six, eight = _batch(4), _batch(6), _batch(8)
    assert mx.array_equal(four.student_ids[:, -3:], six.student_ids[:, -3:])
    assert mx.array_equal(four.student_ids[:, -3:], eight.student_ids[:, -3:])
    # ...and K=6 shares five rows with K=8
    assert mx.array_equal(six.student_ids[:, -5:], eight.student_ids[:, -5:])


def test_corruption_rate_is_already_block_size_independent():
    """The per-sequence rate has shape (B, 1), so it needs no alignment."""
    assert mx.array_equal(_batch(4).corruption_rate, _batch(8).corruption_rate)


def test_alignment_preserves_the_prefix_and_seed_invariants():
    batch = _batch(4)
    cut = batch.block_start + 1
    assert mx.array_equal(batch.student_ids[:, :cut], batch.teacher_ids[:, :cut])
    assert float(mx.sum(batch.lora_mask[:, :cut]).item()) == 0.0
    assert batch.student_ids.shape == batch.teacher_ids.shape


def test_alignment_still_corrupts_the_right_number_of_rows():
    for block_size in (2, 4, 6, 8):
        batch = _batch(block_size, corruption="full")
        assert batch.corrupted.shape[1] == block_size
        assert float(mx.mean(batch.corrupted[:, 1:]).item()) == 1.0


def test_default_alignment_reproduces_the_u3_u4_behaviour():
    """`None` must be bit-identical to the pre-U5 code path, or U4 stops reproducing."""
    windows = random_windows(4, 32, 400, seed=77)
    key = stream_key(1, TRAIN_NOISE, 5)
    plain = build_uno_batch(windows, 4, 480, 512, corruption="uniform", key=key)
    explicit = build_uno_batch(
        windows, 4, 480, 512, corruption="uniform", key=key, noise_align_width=4
    )
    assert mx.array_equal(plain.student_ids, explicit.student_ids)
    assert mx.array_equal(plain.corrupted, explicit.corrupted)


def test_alignment_narrower_than_the_block_is_an_error():
    windows = random_windows(2, 32, 400, seed=78)
    with pytest.raises(ValueError, match="smaller than block_size"):
        build_uno_batch(windows, 8, 480, 512, noise_align_width=4)


def test_alignment_does_not_leak_the_gold_future():
    """The invariant that matters most must survive the change."""
    batch = _batch(8, corruption="full")
    start = batch.block_start + 1
    matches = int(mx.sum(
        (batch.student_ids[:, start:] == batch.teacher_ids[:, start:]).astype(mx.int32)
    ).item())
    assert matches <= 2, f"{matches} draft rows leaked the gold token"


def test_alignment_consumes_no_global_rng():
    # Windows are built up front: `random_windows` reseeds the global stream itself,
    # so building them inside the measured region would test the fixture, not the
    # batch construction.
    windows = random_windows(4, 32, 400, seed=101)

    mx.random.seed(11)
    reference = mx.random.uniform(shape=(4,))
    mx.random.seed(11)
    for step in range(5):
        build_uno_batch(
            windows, 8, 480, 512, corruption="uniform",
            key=stream_key(4242, TRAIN_NOISE, step), noise_align_width=ALIGN,
        )
    assert mx.array_equal(reference, mx.random.uniform(shape=(4,)))

    # non-vacuity: unkeyed construction *does* advance the global stream
    mx.random.seed(11)
    for _ in range(5):
        build_uno_batch(windows, 8, 480, 512, corruption="uniform")
    assert not mx.array_equal(reference, mx.random.uniform(shape=(4,)))


# --------------------------------------------------- curriculum on a resume


def _curriculum(**kwargs):
    base = dict(block_curriculum=[4, 6, 8], steps=16000)
    base.update(kwargs)
    return UnoTrainConfig(**base)


def test_curriculum_stages_divide_the_resumed_span():
    config = _curriculum()
    start = 12800  # 3200 steps of run, three equal stages of ~1066
    assert config.block_size_at(12800, start) == 4
    assert config.block_size_at(13865, start) == 4
    assert config.block_size_at(13867, start) == 6
    assert config.block_size_at(14932, start) == 6
    assert config.block_size_at(14934, start) == 8
    assert config.block_size_at(15999, start) == 8


def test_curriculum_without_the_offset_would_be_a_single_stage():
    """The bug this fix exists for: every step of a late resume lands in stage 3."""
    config = _curriculum()
    assert {config.block_size_at(s) for s in range(12800, 16000, 100)} == {8}


def test_curriculum_from_zero_is_unchanged():
    config = _curriculum(steps=12)
    assert config.block_size_at(0) == 4
    assert config.block_size_at(5) == 6
    assert config.block_size_at(11) == 8


def test_every_stage_gets_the_same_number_of_steps():
    config = _curriculum()
    start = 12800
    counts: dict[int, int] = {}
    for step in range(start, config.steps):
        block = config.block_size_at(step, start)
        counts[block] = counts.get(block, 0) + 1
    assert set(counts) == {4, 6, 8}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_fixed_block_size_ignores_the_offset():
    config = UnoTrainConfig(block_size=4, steps=16000)
    assert config.block_size_at(15000, 12800) == 4
    assert config.block_size_at(0) == 4
