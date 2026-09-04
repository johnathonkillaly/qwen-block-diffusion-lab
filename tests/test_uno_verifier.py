"""Block verification.

`test_parallel_verifier_matches_sequential_reference` is the load-bearing test in this
file. If the single-forward verifier and the L-separate-calls reference ever disagree,
every acceptance and speedup number in the results is void.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from qdif.uno.tiny import build_tiny_model, random_windows
from qdif.uno.verifier import (
    greedy_accept,
    greedy_accept_batch,
    sequential_reference_accept_length,
)

VOCAB = 8


def _logits_forcing(targets: list[int]) -> mx.array:
    """`[L, VOCAB]` whose argmax at row i is `targets[i]`."""
    rows = mx.zeros((len(targets), VOCAB))
    onehot = mx.array(targets, dtype=mx.int32)[:, None]
    return rows + (mx.arange(VOCAB)[None, :] == onehot).astype(mx.float32)


def test_full_acceptance_commits_block_plus_lookahead():
    block = [1, 2, 3, 4]  # clean token 1, then specs 2, 3, 4
    # verifier predicts 2, 3, 4 then a lookahead 5
    result = greedy_accept(block, _logits_forcing([2, 3, 4, 5]))
    assert result.accepted_specs == 3
    assert result.offered_specs == 3
    assert result.full_block
    # the clean token is committed too, and the lookahead is free: L+1 tokens
    assert result.committed == [1, 2, 3, 4, 5]
    assert result.num_committed == len(block) + 1


def test_partial_acceptance_substitutes_the_verifier_token():
    block = [1, 2, 3, 4]
    # agrees on 2, then wants 9 instead of 3
    result = greedy_accept(block, _logits_forcing([2, 7, 4, 5]))
    assert result.accepted_specs == 1
    assert not result.full_block
    assert result.committed == [1, 2, 7]


def test_zero_acceptance_still_commits_one_correct_token():
    """A cycle must never stall: the verifier's own token is always committed."""
    block = [1, 2, 3, 4]
    result = greedy_accept(block, _logits_forcing([7, 7, 7, 7]))
    assert result.accepted_specs == 0
    # still two tokens: the clean one, plus the verifier's correction
    assert result.committed == [1, 7]


def test_block_size_one_is_plain_autoregression():
    """K=1 offers no speculation, but still yields 2 tokens from its 2 forwards."""
    result = greedy_accept([1], _logits_forcing([6]))
    assert result.offered_specs == 0
    assert result.accepted_specs == 0
    assert result.committed == [1, 6]
    assert result.full_block


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match=r"\[L, V\]"):
        greedy_accept([1, 2, 3], _logits_forcing([1, 2]))


def test_batch_helper_matches_per_row():
    blocks = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int32)
    # row 0 accepts both specs (2, 3) and takes lookahead 0;
    # row 1 accepts spec 5 then rejects 6 in favour of the verifier's 2.
    logits = mx.stack([_logits_forcing([2, 3, 0]), _logits_forcing([5, 2, 1])])
    results = greedy_accept_batch(blocks, logits)
    assert [r.committed for r in results] == [[1, 2, 3, 0], [4, 5, 2]]
    assert [r.full_block for r in results] == [True, False]


@pytest.mark.parametrize("block_size", [2, 4, 8])
def test_parallel_verifier_matches_sequential_reference(block_size):
    """One causal forward must give the same accepted prefix as L separate AR calls.

    The proposals are deliberately a mix of the model's own continuation and random
    tokens, so the acceptance length varies across the possible range instead of
    always being 0 or L.
    """
    model = build_tiny_model(seed=11)
    context = random_windows(1, 16, model.vocab_size, seed=12)[0]

    accepted_lengths = set()
    for trial in range(12):
        mx.random.seed(100 + trial)
        # Seed the block with the model's true next token, then corrupt a suffix.
        logits = model.ar_logits(context[None])
        true_next = int(mx.argmax(logits[0, -1]).item())
        block = [true_next]
        running = mx.concatenate([context, mx.array([true_next], dtype=context.dtype)])
        for step in range(block_size - 1):
            step_logits = model.ar_logits(running[None])
            token = int(mx.argmax(step_logits[0, -1]).item())
            if step >= trial % block_size:  # corrupt from a varying point onwards
                token = int(mx.random.randint(1, 400, (1,)).item())
            block.append(token)
            running = mx.concatenate([running, mx.array([token], dtype=context.dtype)])

        verify_ids = mx.concatenate([context, mx.array(block, dtype=context.dtype)])
        verify_logits = model.ar_logits(verify_ids[None])[0, -block_size:]
        fast = greedy_accept(block, verify_logits)
        slow = sequential_reference_accept_length(model, context, block)
        assert fast.accepted_specs == slow, (
            f"block_size={block_size} trial={trial}: "
            f"parallel said {fast.accepted_specs}, sequential said {slow}"
        )
        accepted_lengths.add(slow)

    assert len(accepted_lengths) > 1, (
        "the test never produced differing acceptance lengths, so it did not "
        "actually exercise the comparison"
    )


def test_verifier_lookahead_is_the_models_own_next_token():
    """On full acceptance the extra committed token must be the AR continuation."""
    model = build_tiny_model(seed=13)
    context = random_windows(1, 12, model.vocab_size, seed=14)[0]

    block = []
    running = context
    for _ in range(4):
        token = int(mx.argmax(model.ar_logits(running[None])[0, -1]).item())
        block.append(token)
        running = mx.concatenate([running, mx.array([token], dtype=context.dtype)])

    verify_ids = mx.concatenate([context, mx.array(block, dtype=context.dtype)])
    verify_logits = model.ar_logits(verify_ids[None])[0, -4:]
    result = greedy_accept(block, verify_logits)
    assert result.full_block
    expected_lookahead = int(mx.argmax(model.ar_logits(running[None])[0, -1]).item())
    assert result.committed == block + [expected_lookahead]
