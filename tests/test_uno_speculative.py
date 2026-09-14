"""Semantics of the new decoder variants: refinement, adaptive K, n-gram control.

Every test here asserts a *property* of the output, not merely that the call returned.
The properties are the ones the brief's §23 names, and the first of them is the one
that matters: whatever the drafter does, the committed tokens must be exactly the
frozen model's greedy continuation.

`assert_varied` guards every identity assertion. A degenerate reference sequence makes
"speculative == AR" pass vacuously, and that is exactly how a real off-by-one in the
verifier once survived the whole suite.
"""

from __future__ import annotations

import pytest

from qdif.uno.decode import ar_greedy_generate, uno_greedy_generate
from qdif.uno.speculative import (
    ExpectedValueK,
    EntropyThresholdK,
    FixedK,
    SurvivalK,
    _ngram_propose,
    adaptive_greedy_generate,
    ngram_greedy_generate,
)
from qdif.uno.tiny import assert_varied, build_tiny_model

PROMPT = [11, 42, 7, 99, 3, 61, 25, 8]
TOKENS = 24
NOISE_SEED = 20260913


@pytest.fixture(scope="module")
def model():
    m = build_tiny_model(seed=3)
    m.attach_adapter(rank=4, full_attention=True, mlp=True)
    return m


@pytest.fixture(scope="module")
def reference(model):
    tokens, _ = ar_greedy_generate(model, PROMPT, max_tokens=TOKENS)
    return assert_varied(tokens)


# ------------------------------------------------------------------- identity


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_adaptive_fixed_k_is_lossless(model, reference, k):
    """Gate 1. Adaptive decoding at a constant K reproduces the AR tokens exactly."""
    tokens, stats, _ = adaptive_greedy_generate(
        model, PROMPT, max_tokens=TOKENS,
        policy=FixedK(name="fixed", allowed=(k,), default_k=k),
        noise_stream_seed=NOISE_SEED,
    )
    assert tokens == reference
    assert stats.tokens == len(reference)


@pytest.mark.parametrize("k", [2, 4, 8])
def test_adaptive_matches_the_frozen_decoder_exactly(model, k):
    """The new decoder is a superset of the old one, not a reimplementation of it.

    At a constant K and the same noise stream, `adaptive_greedy_generate` must agree
    with `decode.uno_greedy_generate` on tokens *and* on the acceptance accounting.
    Without this, an Act IV-U number and a new number would not be comparable.
    """
    old_tokens, old_stats = uno_greedy_generate(
        model, PROMPT, max_tokens=TOKENS, block_size=k,
        transaction_mode="snapshot", noise_stream_seed=NOISE_SEED,
    )
    new_tokens, new_stats, _ = adaptive_greedy_generate(
        model, PROMPT, max_tokens=TOKENS,
        policy=FixedK(name="fixed", allowed=(k,), default_k=k),
        transaction_mode="snapshot", noise_stream_seed=NOISE_SEED,
    )
    assert new_tokens == old_tokens
    assert new_stats.accepted_per_cycle == old_stats.accepted_per_cycle
    assert new_stats.committed_per_cycle == old_stats.committed_per_cycle
    assert new_stats.forwards == old_stats.forwards


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_refinement_is_lossless(model, reference, steps):
    """Refinement may change *which* tokens are proposed. It may never change which
    tokens are committed."""
    tokens, stats, _ = adaptive_greedy_generate(
        model, PROMPT, max_tokens=TOKENS,
        policy=FixedK(name="fixed", allowed=(4,), default_k=4),
        noise_stream_seed=NOISE_SEED, refine_steps=steps,
    )
    assert tokens == reference
    # Each refinement pass is a real extra forward and must be charged as one.
    assert stats.forwards >= stats.cycles * steps


@pytest.mark.parametrize("policy_factory", [
    lambda: EntropyThresholdK(name="entropy", allowed=(2, 4, 8)),
    lambda: SurvivalK(name="survival", allowed=(2, 4, 8)),
    lambda: ExpectedValueK(name="ev", allowed=(2, 4, 8),
                           cost_ms={2: 45.8, 4: 49.2, 8: 55.0}),
])
def test_every_adaptive_policy_is_lossless(model, reference, policy_factory):
    """Gate 1 under a varying block width — the case where a cache-rewind error would
    show up, because the commit length now differs from cycle to cycle."""
    tokens, stats, extra = adaptive_greedy_generate(
        model, PROMPT, max_tokens=TOKENS,
        policy=policy_factory(), noise_stream_seed=NOISE_SEED,
    )
    assert tokens == reference
    assert len(extra["trace"]) == stats.cycles


def test_ngram_control_is_lossless(model, reference):
    """The control has to be lossless too, or it is not a comparable baseline."""
    tokens, _ = ngram_greedy_generate(
        model, PROMPT, max_tokens=TOKENS, block_size=4, order=2,
    )
    assert tokens == reference


# ------------------------------------------------------------------ accounting


def test_committed_equals_accepted_plus_two(model):
    """The identity Act IV-U4 relies on, re-checked for a *varying* K.

    If this breaks, the horizon survival curve stops being the expected accepted
    prefix and every economic argument built on it is void.
    """
    _, stats, extra = adaptive_greedy_generate(
        model, PROMPT, max_tokens=TOKENS,
        policy=SurvivalK(name="survival", allowed=(2, 4, 8)),
        noise_stream_seed=NOISE_SEED,
    )
    # The final cycle is truncated by the token budget, so it is excluded.
    for row in extra["trace"][:-1]:
        assert row["committed"] == row["accepted"] + 2


def test_rejected_suffix_is_never_committed(model, reference):
    """Brief §23. A cycle that accepts `a` of its proposals must contribute exactly
    `a + 2` tokens to the output, never more.

    Stated as an invariant over the whole decode rather than per cycle: the committed
    stream reconstructed from the trace must be the reference prefix, so no rejected
    proposal can have leaked in and been overwritten later.
    """
    tokens, stats, extra = adaptive_greedy_generate(
        model, PROMPT, max_tokens=TOKENS,
        policy=SurvivalK(name="survival", allowed=(2, 4, 8)),
        noise_stream_seed=NOISE_SEED,
    )
    assert tokens == reference
    assert sum(row["committed"] for row in extra["trace"]) == len(tokens)
    for row in extra["trace"]:
        # Never more than the block could legitimately yield: K-1 speculative slots,
        # plus the adapter-off seed token, plus the free lookahead on a full block.
        assert row["accepted"] <= row["k"] - 1
        assert row["committed"] <= row["k"] + 1


def test_scheduler_never_selects_an_unsupported_k(model):
    """Brief §23. A policy that proposes a width outside its allowed set would either
    crash the draft pass or silently decode at a width nothing was calibrated for."""
    allowed = (2, 4)
    _, _, extra = adaptive_greedy_generate(
        model, PROMPT, max_tokens=TOKENS,
        policy=ExpectedValueK(name="ev", allowed=allowed,
                              cost_ms={2: 45.8, 4: 49.2}),
        noise_stream_seed=NOISE_SEED,
    )
    assert {row["k"] for row in extra["trace"]} <= set(allowed)


def test_adaptive_decoding_is_deterministic(model):
    """Same noise stream, same policy, same tokens and the same *width schedule*.

    A scheduler whose choices drifted between runs would make every paired comparison
    meaningless.
    """
    runs = [
        adaptive_greedy_generate(
            model, PROMPT, max_tokens=TOKENS,
            policy=SurvivalK(name="survival", allowed=(2, 4, 8)),
            noise_stream_seed=NOISE_SEED,
        )
        for _ in range(2)
    ]
    assert runs[0][0] == runs[1][0]
    assert [r["k"] for r in runs[0][2]["trace"]] == [r["k"] for r in runs[1][2]["trace"]]


def test_confidence_is_collected_only_when_asked(model):
    _, _, off = adaptive_greedy_generate(
        model, PROMPT, max_tokens=8,
        policy=FixedK(name="fixed", allowed=(4,), default_k=4),
        noise_stream_seed=NOISE_SEED, collect_confidence=False,
    )
    _, _, on = adaptive_greedy_generate(
        model, PROMPT, max_tokens=8,
        policy=FixedK(name="fixed", allowed=(4,), default_k=4),
        noise_stream_seed=NOISE_SEED, collect_confidence=True,
    )
    assert off["trace"][0]["draft_top1"] == []
    assert len(on["trace"][0]["draft_top1"]) == 3
    assert all(0.0 <= v <= 1.0 for v in on["trace"][0]["draft_top1"])


# --------------------------------------------------------------- n-gram lookup


def test_ngram_propose_finds_the_most_recent_match():
    history = [1, 2, 3, 9, 9, 1, 2, 3, 7, 7, 1, 2, 3]
    assert _ngram_propose(history, block_size=3, order=3) == [9, 9] or \
           _ngram_propose(history, block_size=3, order=3) == [7, 7]
    # Most recent, specifically: the [1,2,3] at index 5 is followed by 7, 7.
    assert _ngram_propose(history, block_size=3, order=3) == [7, 7]


def test_ngram_propose_returns_none_without_a_match():
    assert _ngram_propose([1, 2, 3, 4, 5], block_size=4, order=3) is None
    assert _ngram_propose([1, 2], block_size=4, order=3) is None


def test_ngram_propose_never_returns_a_short_block():
    """A short proposal would desynchronise the verify block width from `block_size`."""
    history = [1, 2, 3, 4, 1, 2, 3]
    guess = _ngram_propose(history, block_size=8, order=3)
    assert guess is None or len(guess) == 7
