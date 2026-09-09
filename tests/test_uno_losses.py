"""Distillation objectives: identities, bounds, and gradient sanity."""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from qdif.uno.losses import (
    agreement_metrics,
    combine,
    cross_entropy,
    reverse_kl,
    total_variation,
)


def test_total_variation_is_zero_for_identical_distributions():
    logits = mx.random.normal((6, 40))
    assert float(total_variation(logits, logits).item()) == pytest.approx(0.0, abs=1e-6)


def test_total_variation_is_two_for_disjoint_point_masses():
    """L1 between two one-hots on different classes is 2 (we use IFM's unhalved form)."""
    student = mx.array([[100.0, 0.0, 0.0]])
    teacher = mx.array([[0.0, 100.0, 0.0]])
    assert float(total_variation(student, teacher).item()) == pytest.approx(2.0, abs=1e-4)


def test_total_variation_is_bounded_and_normalised_per_token():
    student = mx.random.normal((16, 50))
    teacher = mx.random.normal((16, 50))
    value = float(total_variation(student, teacher).item())
    assert 0.0 < value <= 2.0


def test_total_variation_chunking_does_not_change_the_value():
    student = mx.random.normal((5, 97))
    teacher = mx.random.normal((5, 97))
    whole = float(total_variation(student, teacher, chunk_size=1000).item())
    chunked = float(total_variation(student, teacher, chunk_size=7).item())
    assert whole == pytest.approx(chunked, abs=1e-5)


def test_total_variation_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="logit shapes differ"):
        total_variation(mx.zeros((2, 5)), mx.zeros((2, 6)))


def test_reverse_kl_is_zero_for_identical_distributions():
    logits = mx.random.normal((4, 30))
    assert float(reverse_kl(logits, logits).item()) == pytest.approx(0.0, abs=1e-5)


def test_reverse_kl_is_non_negative():
    for seed in range(5):
        mx.random.seed(seed)
        value = float(reverse_kl(mx.random.normal((8, 25)), mx.random.normal((8, 25))).item())
        assert value >= -1e-5


def test_cross_entropy_matches_the_closed_form_on_a_uniform_distribution():
    vocab = 16
    logits = mx.zeros((3, vocab))
    targets = mx.array([0, 5, 15], dtype=mx.int32)
    assert float(cross_entropy(logits, targets).item()) == pytest.approx(
        math.log(vocab), abs=1e-5
    )


def test_cross_entropy_rejects_misaligned_targets():
    with pytest.raises(ValueError, match="do not match rows"):
        cross_entropy(mx.zeros((3, 8)), mx.array([0, 1], dtype=mx.int32))


def test_combine_applies_the_configured_weights():
    ce, kl, tv = mx.array(2.0), mx.array(3.0), mx.array(5.0)
    assert float(combine(ce, kl, tv, 1.0, 0.0, 0.0).item()) == pytest.approx(2.0)
    assert float(combine(ce, kl, tv, 0.0, 0.0, 1.0).item()) == pytest.approx(5.0)
    assert float(combine(ce, kl, tv, 0.5, 2.0, 1.0).item()) == pytest.approx(12.0)


def test_combine_refuses_an_all_zero_objective():
    zero = mx.array(0.0)
    with pytest.raises(ValueError, match="at least one objective weight"):
        combine(zero, zero, zero, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        combine(zero, zero, zero, -1.0, 0.0, 1.0)


def test_total_variation_gradient_points_towards_the_teacher():
    """One gradient step on TV must reduce TV. The objective has to be trainable."""
    teacher = mx.array([[3.0, 0.0, -1.0, 0.5]])
    student = mx.array([[0.0, 2.0, 0.0, 0.0]])

    def loss_fn(logits):
        return total_variation(logits, teacher)

    before = float(loss_fn(student).item())
    grad = mx.grad(loss_fn)(student)
    after = float(loss_fn(student - 0.5 * grad).item())
    assert after < before


def test_agreement_metrics_report_perfect_agreement_and_low_entropy():
    peaked = mx.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    metrics = agreement_metrics(peaked, peaked)
    assert metrics["top1_agreement"] == pytest.approx(1.0)
    assert metrics["teacher_top1_prob"] > 0.99
    assert metrics["teacher_entropy"] < 0.05


def test_agreement_metrics_detect_disagreement():
    student = mx.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    teacher = mx.array([[0.0, 10.0, 0.0], [10.0, 0.0, 0.0]])
    assert agreement_metrics(student, teacher)["top1_agreement"] == pytest.approx(0.5)


def test_agreement_metrics_report_maximum_entropy_for_a_uniform_teacher():
    uniform = mx.zeros((1, 3))
    metrics = agreement_metrics(uniform, uniform)
    assert metrics["teacher_entropy"] == pytest.approx(math.log(3), abs=1e-4)
    assert metrics["teacher_top1_prob"] == pytest.approx(1 / 3, abs=1e-4)
