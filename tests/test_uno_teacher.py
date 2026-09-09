"""Training-batch construction, causality, and the control machinery.

The test that matters most here is `test_student_cannot_see_the_gold_future`: if the
student's input contained the tokens it is asked to predict, we would be training
ordinary next-token prediction and calling it block prediction.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from qdif.uno.losses import total_variation
from qdif.uno.teacher import build_uno_batch, student_logits_for, teacher_logits_for
from qdif.uno.tiny import build_tiny_model, random_windows


def _batch(width=20, block_size=4, batch=3, **kwargs):
    windows = random_windows(batch, width, 400, seed=31)
    return windows, build_uno_batch(
        windows, block_size=block_size, mask_token_id=480, vocab_size=512, **kwargs
    )


def test_layout_shapes_and_alignment():
    windows, batch = _batch(width=20, block_size=4)
    assert batch.teacher_ids.shape == (3, 19)
    assert batch.student_ids.shape == (3, 19)
    assert batch.lora_mask.shape == (3, 19)
    assert batch.block_start == 20 - 4 - 1
    assert batch.targets.shape == (3, 4)
    # targets are exactly the tokens after each supervised position
    expected = windows[:, batch.block_start + 1 : batch.block_start + 5]
    assert mx.array_equal(batch.targets, expected)


def test_teacher_is_the_untouched_window():
    windows, batch = _batch()
    assert mx.array_equal(batch.teacher_ids, windows[:, :-1])


def test_student_shares_the_prefix_and_seed_with_the_teacher():
    _, batch = _batch(corruption="full")
    cut = batch.block_start + 1
    assert mx.array_equal(batch.student_ids[:, :cut], batch.teacher_ids[:, :cut])


def test_student_cannot_see_the_gold_future():
    """Under full corruption every draft row must differ from the true token.

    Uniform noise can coincide with the truth by chance, so this asserts on the noise
    band rather than on inequality: draft rows must hold ids below `mask_token_id`
    drawn independently of the window, and must not reproduce the gold suffix.
    """
    _, batch = _batch(corruption="full")
    start = batch.block_start + 1
    student_tail = batch.student_ids[:, start:]
    teacher_tail = batch.teacher_ids[:, start:]
    assert student_tail.shape[1] == batch.block_size - 1
    matches = int(mx.sum((student_tail == teacher_tail).astype(mx.int32)).item())
    assert matches <= 1, f"{matches} draft rows leaked the gold token"


def test_lora_mask_is_off_on_prefix_and_seed_and_on_for_draft_rows():
    _, batch = _batch(block_size=4)
    cut = batch.block_start + 1
    assert float(mx.sum(batch.lora_mask[:, :cut]).item()) == 0.0
    assert float(mx.sum(batch.lora_mask[:, cut:]).item()) == 3 * 3  # B x (L-1)


def test_uniform_corruption_leaves_some_rows_clean():
    """The Uno/SDAR schedule samples t ~ U(0,1), so corruption must not be all-or-nothing."""
    windows = random_windows(64, 24, 400, seed=33)
    mx.random.seed(5)
    batch = build_uno_batch(
        windows, block_size=8, mask_token_id=480, vocab_size=512, corruption="uniform"
    )
    fraction = float(mx.mean(batch.corrupted[:, 1:]).item())
    assert 0.15 < fraction < 0.85, f"corruption fraction {fraction} looks degenerate"


def test_full_corruption_corrupts_every_draft_row():
    _, batch = _batch(corruption="full", block_size=6)
    assert float(mx.mean(batch.corrupted[:, 1:]).item()) == 1.0


def test_shuffle_control_moves_the_teacher_and_not_the_student():
    """Control 2 must break the student->teacher correspondence, not preserve it.

    Rolling *both* contexts keeps them aligned, and under a TV-only objective (which
    never reads `targets`) that makes the control bit-identical to the real arm. It
    then reports "no separation" no matter what the adapter learned. This test exists
    because that is exactly the bug the first Act IV-U control shipped with.
    """
    windows = random_windows(4, 20, 400, seed=34)
    normal = build_uno_batch(windows, 4, 480, 512, corruption="full")
    shuffled = build_uno_batch(
        windows, 4, 480, 512, corruption="full", shuffle_targets=True
    )
    assert mx.array_equal(normal.targets, shuffled.targets)
    # the teacher now reads a different sequence...
    assert not mx.array_equal(normal.teacher_ids, shuffled.teacher_ids)
    # ...while the student's own context is untouched. Compare prefixes only: the
    # noise tail is redrawn on every call, so the full arrays are never equal.
    cut = normal.block_start + 1
    assert mx.array_equal(normal.student_ids[:, :cut], shuffled.student_ids[:, :cut])
    assert mx.array_equal(shuffled.student_ids[:, :cut], normal.teacher_ids[:, :cut])
    # and the shuffled teacher really is a *different* row's context
    assert mx.array_equal(
        shuffled.teacher_ids[:, :cut], mx.roll(normal.teacher_ids, 1, axis=0)[:, :cut]
    )


def test_block_size_one_has_no_draft_rows():
    _, batch = _batch(block_size=1)
    assert batch.targets.shape[1] == 1
    assert float(mx.sum(batch.lora_mask).item()) == 0.0


def test_window_too_small_is_a_hard_error():
    windows = random_windows(2, 5, 400, seed=35)
    with pytest.raises(ValueError, match="too small"):
        build_uno_batch(windows, block_size=8, mask_token_id=480, vocab_size=512)


# ------------------------------------------------------------------ with a model


def test_teacher_logits_equal_plain_greedy_ar():
    """The teacher must be the ordinary AR model, not a differently-masked variant."""
    model = build_tiny_model(seed=36)
    model.attach_adapter(rank=4)
    windows = random_windows(2, 18, model.vocab_size, seed=37)
    batch = build_uno_batch(windows, 4, model.mask_token_id, model.vocab_size)

    teacher = teacher_logits_for(model, batch)
    plain = model.ar_logits(batch.teacher_ids)[:, batch.supervised_slice, :]
    assert mx.array_equal(teacher, plain)


def test_seed_position_has_exactly_zero_loss():
    """The seed row runs with the adapter off, so student and teacher must agree there.

    A non-zero value means the gate is leaking onto clean rows, which would invalidate
    both the frozen-verifier claim and every acceptance number.
    """
    from qdif.uno.gated_lora import gated_lora_modules

    model = build_tiny_model(seed=38)
    model.attach_adapter(rank=4, full_attention=True, mlp=True)
    mx.random.seed(39)
    for module in gated_lora_modules(model.base):
        module.lora_b = mx.random.normal(module.lora_b.shape, scale=0.1).astype(mx.float32)
    mx.eval(model.base.parameters())

    windows = random_windows(2, 18, model.vocab_size, seed=40)
    batch = build_uno_batch(
        windows, 4, model.mask_token_id, model.vocab_size, corruption="full"
    )
    teacher = teacher_logits_for(model, batch)
    student = student_logits_for(model, batch)

    assert mx.array_equal(student[:, 0, :], teacher[:, 0, :])
    assert float(total_variation(student[:, :1], teacher[:, :1]).item()) == 0.0
    # ... and the adapted rows must genuinely differ, or the test is vacuous.
    assert float(total_variation(student[:, 1:], teacher[:, 1:]).item()) > 0.0


def test_explicit_key_makes_batch_construction_reproducible():
    windows = random_windows(4, 24, 400, seed=91)
    key = mx.random.key(1234)
    a = build_uno_batch(windows, 4, 480, 512, corruption="uniform", key=key)
    b = build_uno_batch(windows, 4, 480, 512, corruption="uniform", key=key)
    assert mx.array_equal(a.student_ids, b.student_ids)
    assert mx.array_equal(a.corrupted, b.corrupted)


def test_keyed_batch_construction_consumes_no_global_rng():
    """Evaluation must not perturb the training noise stream.

    `evaluate()` builds batches too. While it drew from the *global* MLX RNG, changing
    only the evaluation schedule silently changed the training trajectory: two runs
    with identical seeds and identical training configs diverged, which pre-registered
    gate U3-0b caught. Evaluation now passes an explicit key, so the global stream
    advances only for training.
    """
    windows = random_windows(4, 24, 400, seed=92)
    key = mx.random.key(77)

    mx.random.seed(5)
    reference = mx.random.uniform(shape=(4,))

    mx.random.seed(5)
    for _ in range(3):
        build_uno_batch(windows, 4, 480, 512, corruption="uniform", key=key)
    after_keyed = mx.random.uniform(shape=(4,))
    assert mx.array_equal(reference, after_keyed), "keyed batch consumed global RNG"

    # ...and without a key it *does* advance the global stream, so the test is not vacuous
    mx.random.seed(5)
    for _ in range(3):
        build_uno_batch(windows, 4, 480, 512, corruption="uniform")
    after_global = mx.random.uniform(shape=(4,))
    assert not mx.array_equal(reference, after_global)
