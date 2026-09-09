"""Transactional decode state: snapshot and rewind must equal replay, exactly.

Replay is the oracle. Every other backend is diffed against it — on cache state, on
next-token logits, and on a *long* greedy continuation, because subtle state
corruption shows up several tokens later rather than immediately.

Act IV-U shipped a verifier off-by-one that survived the entire suite because the
fixture emitted a constant token. Every test here therefore routes its reference
through `assert_varied`, and the block fixtures include repeated token ids so that
correctness cannot accidentally depend on token uniqueness.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from qdif.uno.cache_utils import cache_length
from qdif.uno.recurrence import text_model_block_forward
from qdif.uno.tiny import assert_varied, build_tiny_model, random_windows
from qdif.uno.transaction import TransactionStats, begin_transaction

PROMPT = [5, 91, 33, 210, 7, 64, 128, 19, 44, 201]
#: Deliberately contains repeats (7 twice, 33 twice) so that prefix selection cannot
#: be right "by accident" through unique-id matching.
BLOCK = [19, 33, 7, 88, 7, 33, 150, 61]


@pytest.fixture(scope="module")
def model():
    m = build_tiny_model(seed=71)
    m.attach_adapter(rank=4, full_attention=True, mlp=True)
    mx.random.seed(72)
    from qdif.uno.gated_lora import gated_lora_modules

    for module in gated_lora_modules(m.base):
        module.lora_b = mx.random.normal(module.lora_b.shape, scale=0.05).astype(mx.float32)
    mx.eval(m.base.parameters())
    return m


def _prefill(model):
    caches = model.make_cache()
    model.ar_logits(mx.array([PROMPT], dtype=mx.int32), cache=caches)
    mx.eval([c.state for c in caches])
    return caches


def _commit(model, block, keep, mode, stats=None):
    caches = _prefill(model)
    txn = begin_transaction(model, caches, mode=mode, stats=stats or TransactionStats())
    logits = txn.run_block(mx.array([block], dtype=mx.int32))
    mx.eval(logits)
    txn.commit_prefix(keep)
    return caches, logits, txn.stats


def _continue_greedy(model, caches, first_token, steps=12):
    """Greedy continuation from committed state — the strictest state equality test."""
    tokens = []
    token = first_token
    for _ in range(steps):
        logits = model.ar_logits(mx.array([[token]], dtype=mx.int32), cache=caches)
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
    return tokens


def _cache_signature(model, caches):
    """Everything a subsequent forward can observe."""
    signature = []
    for index, layer in enumerate(model.base.layers):
        cache = caches[index]
        if getattr(layer, "is_linear", False):
            signature.append(("conv", index, cache[0]))
            signature.append(("state", index, cache[1]))
        else:
            signature.append(("offset", index, int(cache.offset)))
            signature.append(("keys", index, cache.keys[..., : cache.offset, :]))
    return signature


def _assert_signatures_close(a, b, tolerance=0.0):
    assert len(a) == len(b)
    for (kind_a, index_a, value_a), (kind_b, index_b, value_b) in zip(a, b):
        assert (kind_a, index_a) == (kind_b, index_b)
        if kind_a == "offset":
            assert value_a == value_b, f"offset mismatch at layer {index_a}"
            continue
        error = float(mx.max(mx.abs(value_a.astype(mx.float32) - value_b.astype(mx.float32))).item())
        assert error <= tolerance, f"{kind_a} layer {index_a}: max abs error {error}"


# --------------------------------------------------------------- recording path


def test_recording_forward_reproduces_the_stock_model():
    """The unrolled recurrence must not change the model's output."""
    model = build_tiny_model(seed=73)
    ids = random_windows(1, 16, model.vocab_size, seed=74)

    caches_a = model.make_cache()
    reference = model.ar_logits(ids, cache=caches_a)
    mx.eval(reference)

    caches_b = model.make_cache()
    from qdif.uno.gated_lora import lora_disabled

    with lora_disabled():
        recorded, record = text_model_block_forward(model.base, ids, caches_b)
    mx.eval(recorded)

    error = float(mx.max(mx.abs(reference - recorded)).item())
    assert error < 2e-3, f"recording forward diverged from stock: max abs error {error}"
    assert mx.array_equal(mx.argmax(reference, axis=-1), mx.argmax(recorded, axis=-1))
    _assert_signatures_close(
        _cache_signature(model, caches_a), _cache_signature(model, caches_b), tolerance=2e-3
    )


def test_record_holds_one_state_per_token_boundary():
    model = build_tiny_model(seed=75)
    ids = random_windows(1, 6, model.vocab_size, seed=76)
    caches = model.make_cache()
    from qdif.uno.gated_lora import lora_disabled

    with lora_disabled():
        _, record = text_model_block_forward(model.base, ids, caches)
    deltanet_layers = [i for i, l in enumerate(model.base.layers) if l.is_linear]
    assert sorted(record.layers) == deltanet_layers
    for layer_record in record.layers.values():
        assert len(layer_record.states) == 6 + 1  # S_0 … S_S
    assert record.nbytes > 0


# ------------------------------------------------------------------- snapshot


@pytest.mark.parametrize("keep", [0, 1, 2, 4, 7, 8])
def test_snapshot_commit_matches_replay_state(model, keep):
    """Approach B must land the cache in exactly the state Approach A produces."""
    replay_caches, _, _ = _commit(model, BLOCK, keep, "replay")
    snap_caches, _, _ = _commit(model, BLOCK, keep, "snapshot")
    assert cache_length(replay_caches) == cache_length(snap_caches) == len(PROMPT) + keep
    _assert_signatures_close(
        _cache_signature(model, replay_caches),
        _cache_signature(model, snap_caches),
        tolerance=2e-3,
    )


@pytest.mark.parametrize("keep", [0, 1, 2, 4, 7, 8])
def test_snapshot_continuation_matches_replay(model, keep):
    """The strict test: identical greedy continuations from the committed state."""
    replay_caches, _, _ = _commit(model, BLOCK, keep, "replay")
    snap_caches, _, _ = _commit(model, BLOCK, keep, "snapshot")
    reference = _continue_greedy(model, replay_caches, first_token=BLOCK[0], steps=16)
    candidate = _continue_greedy(model, snap_caches, first_token=BLOCK[0], steps=16)
    assert_varied(reference)
    assert candidate == reference


def test_snapshot_performs_no_extra_full_forward(model):
    """Gate U2-2, instrumented rather than inferred from timing."""
    for keep in (0, 1, 3, 5, 8):
        _, _, replay_stats = _commit(model, BLOCK, keep, "replay")
        _, _, snap_stats = _commit(model, BLOCK, keep, "snapshot")
        assert snap_stats.full_forwards == 1
        assert snap_stats.replay_forwards == 0
        assert snap_stats.state_only_commits == 1
        expected_replay = 1 if 0 < keep < len(BLOCK) else 0
        assert replay_stats.replay_forwards == expected_replay


@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
def test_snapshot_matches_replay_at_every_prefix_for_every_block_size(model, block_size):
    block = BLOCK[:block_size]
    for keep in range(block_size + 1):
        replay_caches, _, _ = _commit(model, block, keep, "replay")
        snap_caches, _, _ = _commit(model, block, keep, "snapshot")
        reference = _continue_greedy(model, replay_caches, block[0], steps=10)
        candidate = _continue_greedy(model, snap_caches, block[0], steps=10)
        assert candidate == reference, f"K={block_size} keep={keep}"


def test_repeated_tokens_do_not_confuse_prefix_selection(model):
    """A block of one repeated id must still commit the right *count* of tokens."""
    block = [42] * 6
    lengths = set()
    for keep in range(7):
        caches, _, _ = _commit(model, block, keep, "snapshot")
        lengths.add(cache_length(caches))
        assert cache_length(caches) == len(PROMPT) + keep
    assert len(lengths) == 7


def test_rollback_restores_the_pre_block_state(model):
    baseline = _prefill(model)
    reference = _continue_greedy(model, baseline, first_token=BLOCK[0], steps=12)
    assert_varied(reference)

    caches = _prefill(model)
    txn = begin_transaction(model, caches, mode="snapshot")
    txn.run_block(mx.array([BLOCK], dtype=mx.int32))
    txn.rollback()
    assert cache_length(caches) == len(PROMPT)
    assert _continue_greedy(model, caches, first_token=BLOCK[0], steps=12) == reference


def test_commit_prefix_rejects_out_of_range(model):
    caches = _prefill(model)
    txn = begin_transaction(model, caches, mode="snapshot")
    txn.run_block(mx.array([BLOCK], dtype=mx.int32))
    with pytest.raises(ValueError, match="outside"):
        txn.commit_prefix(len(BLOCK) + 1)


def test_transaction_cannot_be_reused(model):
    caches = _prefill(model)
    txn = begin_transaction(model, caches, mode="snapshot")
    txn.run_block(mx.array([BLOCK], dtype=mx.int32))
    txn.commit_prefix(2)
    with pytest.raises(RuntimeError, match="already closed"):
        txn.commit_prefix(3)


def test_unknown_mode_is_rejected(model):
    with pytest.raises(ValueError, match="unknown transaction mode"):
        begin_transaction(model, model.make_cache(), mode="teleport")


# --------------------------------------------------------------------- rewind


@pytest.mark.parametrize("keep", [0, 2, 4, 6])
def test_rewind_matches_snapshot_continuation(model, keep):
    """Gate U2-4. Algebraic undo must reach the same place as the recorded state."""
    snap_caches, _, _ = _commit(model, BLOCK, keep, "snapshot")
    rewind_caches, _, stats = _commit(model, BLOCK, keep, "rewind")
    reference = _continue_greedy(model, snap_caches, BLOCK[0], steps=16)
    candidate = _continue_greedy(model, rewind_caches, BLOCK[0], steps=16)
    assert_varied(reference)
    assert candidate == reference, f"rewind diverged at keep={keep}"
    assert stats.full_forwards == 1
    assert stats.replay_forwards == 0


def test_rewind_reports_its_fallbacks(model):
    _, _, stats = _commit(model, BLOCK, 3, "rewind")
    assert stats.rewind_steps > 0
    assert stats.rewind_fallbacks >= 0
