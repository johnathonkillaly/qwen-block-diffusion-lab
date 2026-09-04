"""The Act IV-U3 decoder cost model.

Pure arithmetic, so it is tested exactly rather than approximately. The important
property is that `committed_tokens` matches the verifier's actual commit rule — if
these two ever disagree, every projected acceptance threshold is wrong.
"""

from __future__ import annotations

import pytest

from qdif.uno.cost_model import (
    CycleCost,
    ceiling,
    committed_tokens,
    distribution_from_counts,
    expected_committed,
    geometric_distribution,
    predicted_tokens_per_second,
    predicted_tpf,
    required_per_slot_accept,
    speedup_table,
)
from qdif.uno.verifier import greedy_accept

COST = CycleCost(proposal_ms=10.0, verify_ms=9.0, commit_ms=1.0)


def test_committed_tokens_matches_the_real_verifier():
    """The cost model's commit rule must be the verifier's commit rule.

    Built by running the actual `greedy_accept` with forced logits rather than by
    restating the formula, so the two cannot drift apart.
    """
    import mlx.core as mx

    vocab = 16
    for block_size in (2, 4, 8):
        block = list(range(1, block_size + 1))
        for accepted in range(block_size):
            # verifier agrees on the first `accepted` specs, then diverges
            targets = list(block[1:]) + [block_size + 1]
            if accepted < block_size - 1:
                targets[accepted] = 15  # a token not in the block
            rows = mx.zeros((block_size, vocab))
            onehot = mx.array(targets, dtype=mx.int32)[:, None]
            logits = rows + (mx.arange(vocab)[None, :] == onehot).astype(mx.float32)

            result = greedy_accept(block, logits)
            assert result.accepted_specs == accepted
            assert result.num_committed == committed_tokens(block_size, accepted), (
                f"K={block_size} accepted={accepted}: verifier committed "
                f"{result.num_committed}, model says {committed_tokens(block_size, accepted)}"
            )


def test_full_acceptance_yields_block_plus_one():
    assert committed_tokens(4, 3) == 5
    assert committed_tokens(2, 1) == 3
    assert committed_tokens(8, 7) == 9


def test_zero_acceptance_still_commits_two():
    for block_size in (2, 4, 8):
        assert committed_tokens(block_size, 0) == 2


def test_committed_tokens_rejects_impossible_counts():
    with pytest.raises(ValueError, match="outside"):
        committed_tokens(4, 4)
    with pytest.raises(ValueError, match="outside"):
        committed_tokens(4, -1)


def test_geometric_distribution_is_a_distribution():
    for block_size in (2, 4, 8):
        for probability in (0.0, 0.25, 0.5, 0.9, 1.0):
            distribution = geometric_distribution(block_size, probability)
            assert sum(distribution.values()) == pytest.approx(1.0)
            assert all(v >= 0 for v in distribution.values())


def test_geometric_extremes():
    assert geometric_distribution(4, 0.0)[0] == pytest.approx(1.0)
    assert geometric_distribution(4, 1.0)[3] == pytest.approx(1.0)


def test_tpf_is_expected_tokens_over_two_forwards():
    distribution = geometric_distribution(4, 1.0)
    assert predicted_tpf(4, distribution) == pytest.approx(2.5)
    assert predicted_tpf(2, geometric_distribution(2, 1.0)) == pytest.approx(1.5)
    # never below 1.0: a cycle always commits at least 2 tokens for 2 forwards
    assert predicted_tpf(4, geometric_distribution(4, 0.0)) == pytest.approx(1.0)


def test_expected_committed_is_monotone_in_acceptance():
    values = [
        expected_committed(4, geometric_distribution(4, p))
        for p in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    ]
    assert values == sorted(values)
    assert values[0] == pytest.approx(2.0)
    assert values[-1] == pytest.approx(5.0)


def test_predicted_throughput_uses_measured_stage_costs():
    distribution = geometric_distribution(4, 1.0)
    # 5 tokens per cycle, 20 ms per cycle -> 250 tok/s
    assert predicted_tokens_per_second(4, distribution, COST) == pytest.approx(250.0)


def test_required_acceptance_inverts_the_forward_model():
    ar = 100.0
    for target in (1.1, 1.5, 2.0):
        per_slot = required_per_slot_accept(4, COST, ar, target)
        assert per_slot is not None
        achieved = predicted_tokens_per_second(
            4, geometric_distribution(4, per_slot), COST
        )
        assert achieved / ar >= target - 1e-3
        # and it is the *smallest* such value: a hair less must miss
        lower = predicted_tokens_per_second(
            4, geometric_distribution(4, max(0.0, per_slot - 0.01)), COST
        )
        assert lower / ar < target + 1e-6


def test_unreachable_targets_return_none_rather_than_clamping():
    """An honest 'not possible' beats a silently clipped number."""
    ar = 1000.0  # AR far faster than the block decoder can ever be
    assert required_per_slot_accept(4, COST, ar, 2.0) is None
    rows = speedup_table(4, COST, ar, targets=(2.0,))
    assert rows[0]["reachable"] is False
    assert rows[0]["required_per_slot_accept"] is None


def test_speedup_table_is_ordered_and_complete():
    rows = speedup_table(4, COST, 100.0)
    assert [r["target_speedup"] for r in rows] == [1.10, 1.20, 1.30, 1.50, 2.00]
    reachable = [r for r in rows if r["reachable"]]
    demands = [r["required_per_slot_accept"] for r in reachable]
    assert demands == sorted(demands), "harder targets must demand more acceptance"


def test_ceiling_reports_the_structural_maximum():
    result = ceiling(4, COST, 100.0)
    assert result["max_tpf"] == pytest.approx(2.5)
    assert result["max_tokens_per_cycle"] == 5
    assert result["max_tokens_per_second"] == pytest.approx(250.0)


def test_distribution_from_counts_normalises():
    distribution = distribution_from_counts({0: 3, 1: 1})
    assert distribution == {0: 0.75, 1: 0.25}
    with pytest.raises(ValueError, match="empty"):
        distribution_from_counts({})
