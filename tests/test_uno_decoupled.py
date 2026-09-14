"""Act IV-U6 decoupled decoder: the semantics gate U6-0 depends on.

Three drafters are used, because one would leave paths untested. The real diffusion
drafter on a tiny model rarely gets a whole block accepted, so it almost never reaches
stage 2. An **oracle** drafter proposes the target's own continuation, so every stage
runs. An **adversarial** drafter proposes wrong tokens, so no continuation ever fires.
Every identity assertion routes its reference through `assert_varied`.
"""

from __future__ import annotations

import pytest

from qdif.uno.decode import ar_greedy_generate, uno_greedy_generate
from qdif.uno.decoupled import decoupled_greedy_generate
from qdif.uno.tiny import assert_varied, build_tiny_model

PROMPT = [11, 42, 7, 99, 3, 61, 25, 8]
TOKENS = 24
NOISE_SEED = 20260914


@pytest.fixture(scope="module")
def model():
    m = build_tiny_model(seed=3)
    m.attach_adapter(rank=4, full_attention=True, mlp=True)
    return m


@pytest.fixture(scope="module")
def reference(model):
    tokens, _ = ar_greedy_generate(model, PROMPT, max_tokens=TOKENS + 16)
    return assert_varied(tokens)


def _oracle(reference):
    """Proposes exactly the target's greedy continuation from the current position."""
    def propose(context, width):
        g = len(context) - len(PROMPT)
        out = list(reference[g : g + width])
        return out + [out[-1] if out else 1] * (width - len(out))
    return propose


def _adversary(reference):
    """Correct first token (the seed row is always exact), wrong everything after."""
    def propose(context, width):
        g = len(context) - len(PROMPT)
        return [reference[g]] + [(reference[g] + 17 + i) % 400 + 1 for i in range(width - 1)]
    return propose


# --------------------------------------------------------------- identity with incumbent


@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("staged", [False, True])
def test_equal_widths_are_the_incumbent_decoder(model, k, staged):
    """D == V must be `uno_greedy_generate`, token for token and cycle for cycle.
    Without this, B_dec vs B would not be a harness-equivalence control."""
    old_tokens, old = uno_greedy_generate(
        model, PROMPT, max_tokens=TOKENS, block_size=k,
        transaction_mode="snapshot", noise_stream_seed=NOISE_SEED)
    new_tokens, new, extra = decoupled_greedy_generate(
        model, PROMPT, max_tokens=TOKENS, draft_width=k, verify_width=k, staged=staged,
        noise_stream_seed=NOISE_SEED)
    assert new_tokens == old_tokens
    assert new.accepted_per_cycle == old.accepted_per_cycle
    assert new.committed_per_cycle == old.committed_per_cycle
    assert new.forwards == old.forwards
    assert new.forward_tokens == old.forward_tokens
    assert all(len(r["stages"]) == 1 for r in extra["trace"])


# ---------------------------------------------------------------------- losslessness


ARMS = [(8, 4, True), (8, 2, True), (6, 4, True), (6, 3, True),
        (8, 4, False), (8, 2, False), (6, 4, False), (6, 3, False),
        (8, 1, True), (5, 2, True), (16, 4, True)]


@pytest.mark.parametrize("draft,verify,staged", ARMS)
def test_diffusion_drafter_is_lossless(model, reference, draft, verify, staged):
    tokens, stats, _ = decoupled_greedy_generate(
        model, PROMPT, max_tokens=TOKENS, draft_width=draft, verify_width=verify,
        staged=staged, noise_stream_seed=NOISE_SEED)
    assert tokens == reference[:TOKENS]
    assert stats.tokens == TOKENS


@pytest.mark.parametrize("draft,verify,staged", ARMS)
def test_oracle_and_adversarial_drafters_are_lossless(model, reference, draft, verify, staged):
    for proposer in (_oracle(reference), _adversary(reference)):
        tokens, _, _ = decoupled_greedy_generate(
            model, PROMPT, max_tokens=TOKENS, draft_width=draft, verify_width=verify,
            staged=staged, proposer=proposer)
        assert tokens == reference[:TOKENS]


# ------------------------------------------------------------------ staging semantics


def test_oracle_drafter_runs_every_stage_of_a_staged_cycle(model, reference):
    """S(8,2) with a perfect drafter: stage 1 [seed,p1,p2], stage 2 [p3,p4,p5],
    stage 3 [p6,p7,p8]. Each commits 3 tokens, so a full cycle commits D + 1 = 9 tokens
    from one draft and three verify forwards."""
    tokens, stats, extra = decoupled_greedy_generate(
        model, PROMPT, max_tokens=TOKENS, draft_width=8, verify_width=2, staged=True,
        proposer=_oracle(reference))
    assert tokens == reference[:TOKENS]
    full = extra["trace"][:-1]  # the final cycle is truncated by the token budget
    assert full, "expected at least one untruncated cycle"
    for record in full:
        assert [s["verify_tokens"] for s in record["stages"]] == [3, 3, 3]
        assert [s["committed"] for s in record["stages"]] == [3, 3, 3]
        assert record["committed"] == 9
        assert record["unused_draft_tokens"] == 0
    # One draft forward per cycle regardless of stage count.
    assert extra["verify_forwards"] == sum(len(r["stages"]) for r in extra["trace"])


def test_truncate_never_runs_a_second_stage_even_with_a_perfect_drafter(model, reference):
    _, _, extra = decoupled_greedy_generate(
        model, PROMPT, max_tokens=TOKENS, draft_width=8, verify_width=4, staged=False,
        proposer=_oracle(reference))
    assert all(len(r["stages"]) == 1 for r in extra["trace"])
    assert all(r["unused_draft_tokens"] == 4 for r in extra["trace"][:-1])


def test_adversarial_drafter_never_continues(model, reference):
    """With V > 1, stage 1 can never be fully accepted, so no suffix is ever reused."""
    _, _, extra = decoupled_greedy_generate(
        model, PROMPT, max_tokens=TOKENS, draft_width=8, verify_width=4, staged=True,
        proposer=_adversary(reference))
    assert all(len(r["stages"]) == 1 for r in extra["trace"])


def test_continuation_only_after_a_full_block_with_matching_lookahead(model, reference):
    """Brief §14 variant 3: never verify a suffix whose predecessor was rejected."""
    for proposer in (None, _oracle(reference), _adversary(reference)):
        _, _, extra = decoupled_greedy_generate(
            model, PROMPT, max_tokens=TOKENS, draft_width=8, verify_width=2, staged=True,
            noise_stream_seed=NOISE_SEED, proposer=proposer)
        for record in extra["trace"]:
            stages = record["stages"]
            for earlier, later in zip(stages, stages[1:]):
                assert earlier["full_block"] and earlier["lookahead_matched_draft"]
                assert earlier["continues"]


def test_rejected_tokens_are_never_committed_and_widths_are_bounded(model, reference):
    for draft, verify, staged in ARMS:
        tokens, stats, extra = decoupled_greedy_generate(
            model, PROMPT, max_tokens=TOKENS, draft_width=draft, verify_width=verify,
            staged=staged, noise_stream_seed=NOISE_SEED)
        assert sum(r["committed"] for r in extra["trace"]) == len(tokens)
        for record in extra["trace"]:
            for stage in record["stages"]:
                assert stage["verify_tokens"] <= verify + 1
                assert stage["accepted"] <= stage["offered"]
                # a stage keeps at most its accepted slots plus one target token (plus the
                # adapter-off token on stage 1)
                assert stage["committed"] <= stage["accepted"] + (2 if stage["stage"] == 1 else 1)


def test_one_draft_forward_per_cycle(model):
    _, stats, extra = decoupled_greedy_generate(
        model, PROMPT, max_tokens=TOKENS, draft_width=8, verify_width=2, staged=True,
        noise_stream_seed=NOISE_SEED)
    prefill = 1
    assert stats.forwards == prefill + stats.cycles + extra["verify_forwards"]


def test_decoding_is_deterministic(model):
    runs = [decoupled_greedy_generate(
        model, PROMPT, max_tokens=TOKENS, draft_width=6, verify_width=3, staged=True,
        noise_stream_seed=NOISE_SEED) for _ in range(2)]
    assert runs[0][0] == runs[1][0]
    assert [len(r["stages"]) for r in runs[0][2]["trace"]] == \
           [len(r["stages"]) for r in runs[1][2]["trace"]]


@pytest.mark.parametrize("draft,verify", [(0, 1), (4, 0), (2, 4)])
def test_invalid_widths_are_rejected(model, draft, verify):
    with pytest.raises(ValueError):
        decoupled_greedy_generate(model, PROMPT, max_tokens=4, draft_width=draft,
                                  verify_width=verify)
