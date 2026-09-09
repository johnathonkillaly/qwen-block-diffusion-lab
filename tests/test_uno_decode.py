"""Decoding correctness.

The central claim of the method is that Uno decoding is **lossless under greedy
decoding**: it changes how fast tokens arrive, never which tokens arrive. That is
Gate 3, and it is tested here at every block size, with a trained-looking adapter and
an untrained one, cached and uncached.

It must hold for an adapter with *arbitrary* weights. A method that is only lossless
when the draft happens to be good is not lossless; it is lucky.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from qdif.uno.cache_utils import cache_length, restore_caches, snapshot_caches
from qdif.uno.decode import ar_greedy_generate, uno_greedy_generate
from qdif.uno.gated_lora import gated_lora_modules
from qdif.uno.model import fingerprint_backbone
from qdif.uno.tiny import assert_varied, build_tiny_model, random_windows

PROMPT = [5, 91, 33, 210, 7, 64, 128, 19]


def _model(seed: int = 21, trained: bool = False, scale: float = 0.05):
    model = build_tiny_model(seed=seed)
    model.attach_adapter(rank=4, full_attention=True, mlp=True)
    if trained:
        mx.random.seed(seed + 1000)
        for module in gated_lora_modules(model.base):
            module.lora_b = mx.random.normal(module.lora_b.shape, scale=scale).astype(
                mx.float32
            )
        mx.eval(model.base.parameters())
    return model


def test_ar_cached_and_uncached_agree():
    model = _model()
    cached, _ = ar_greedy_generate(model, PROMPT, max_tokens=24, use_cache=True)
    uncached, _ = ar_greedy_generate(model, PROMPT, max_tokens=24, use_cache=False)
    assert assert_varied(cached) == uncached


@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
@pytest.mark.parametrize("trained", [False, True])
def test_uno_reproduces_ar_greedy_exactly(block_size, trained):
    """Gate 3. 100% token equality against the frozen model's own greedy output."""
    model = _model(trained=trained)
    reference, ar_stats = ar_greedy_generate(model, PROMPT, max_tokens=32)
    assert_varied(reference)
    tokens, stats = uno_greedy_generate(
        model, PROMPT, max_tokens=32, block_size=block_size, noise_mode="random_uniform"
    )
    assert tokens == reference, (
        f"block_size={block_size} trained={trained}: Uno diverged from AR greedy at "
        f"position {next((i for i, (a, b) in enumerate(zip(tokens, reference)) if a != b), len(reference))}"
    )
    assert stats.tokens == ar_stats.tokens == 32


@pytest.mark.parametrize("block_size", [2, 4, 8])
def test_uno_cached_matches_uncached(block_size):
    """Cache rewind on a recurrent backbone must not change a single token."""
    model = _model(trained=True)
    cached, cached_stats = uno_greedy_generate(
        model, PROMPT, max_tokens=28, block_size=block_size,
        noise_mode="deterministic_uniform", noise_seed=7, use_cache=True,
    )
    uncached, uncached_stats = uno_greedy_generate(
        model, PROMPT, max_tokens=28, block_size=block_size,
        noise_mode="deterministic_uniform", noise_seed=7, use_cache=False,
    )
    assert assert_varied(cached) == uncached
    # Identical noise means the proposals, and therefore the acceptances, must match.
    assert cached_stats.accepted_specs == uncached_stats.accepted_specs
    assert cached_stats.committed_per_cycle == uncached_stats.committed_per_cycle


def test_deterministic_noise_gives_reproducible_decoding():
    model = _model(trained=True)
    first, first_stats = uno_greedy_generate(
        model, PROMPT, max_tokens=20, block_size=4,
        noise_mode="deterministic_uniform", noise_seed=99,
    )
    second, second_stats = uno_greedy_generate(
        model, PROMPT, max_tokens=20, block_size=4,
        noise_mode="deterministic_uniform", noise_seed=99,
    )
    assert first == second
    assert first_stats.accepted_per_cycle == second_stats.accepted_per_cycle


def test_block_size_one_is_the_no_speculation_control():
    """K=1 offers nothing to verify, so it must be AR output at AR cost.

    Two forwards per cycle yield two tokens (the clean one and the lookahead), i.e.
    exactly 1.0 tokens per forward — the same rate as the AR baseline. That is the
    point of the control: no speedup, and no loss either.
    """
    model = _model(trained=True)
    reference, _ = ar_greedy_generate(model, PROMPT, max_tokens=16)
    tokens, stats = uno_greedy_generate(model, PROMPT, max_tokens=16, block_size=1)
    assert tokens == assert_varied(reference)
    assert stats.offered_specs == 0
    assert all(n == 2 for n in stats.committed_per_cycle[:-1])
    assert stats.tokens_per_forward <= 1.0


def test_forward_accounting_is_consistent():
    model = _model(trained=True)
    reference, _ = ar_greedy_generate(model, PROMPT, max_tokens=30)
    assert_varied(reference)
    _, stats = uno_greedy_generate(model, PROMPT, max_tokens=30, block_size=4)
    # 2 forwards per cycle, plus one replay per partially-accepted cycle,
    # plus a single prefill.
    expected = 1 + 2 * stats.cycles + stats.replay_forwards
    assert stats.forwards == expected
    assert stats.replay_forwards <= stats.cycles
    assert stats.full_blocks + stats.replay_forwards >= stats.cycles - 1


def test_decoding_does_not_touch_the_backbone():
    model = _model(trained=True)
    before = fingerprint_backbone(model.base)
    uno_greedy_generate(model, PROMPT, max_tokens=24, block_size=4)
    ar_greedy_generate(model, PROMPT, max_tokens=24)
    assert fingerprint_backbone(model.base).digest == before.digest


def test_cache_snapshot_restore_is_exact():
    """Rewinding a hybrid cache must reproduce the pre-forward logits bit-for-bit."""
    model = _model(trained=True)
    caches = model.make_cache()
    context = mx.array([PROMPT], dtype=mx.int32)
    model.ar_logits(context, cache=caches)
    mx.eval([c.state for c in caches])

    snapshot = snapshot_caches(caches)
    frontier = cache_length(caches)
    baseline = model.ar_logits(mx.array([[42]], dtype=mx.int32), cache=caches)
    mx.eval(baseline)

    restore_caches(caches, snapshot)
    assert cache_length(caches) == frontier
    # Push unrelated tokens through, then rewind again.
    model.ar_logits(mx.array([[7, 8, 9]], dtype=mx.int32), cache=caches)
    restore_caches(caches, snapshot)
    assert cache_length(caches) == frontier

    replayed = model.ar_logits(mx.array([[42]], dtype=mx.int32), cache=caches)
    assert mx.array_equal(baseline, replayed)


def test_uncached_uno_matches_uncached_ar():
    """The slow, cacheless reference path is itself lossless."""
    model = _model(trained=True)
    reference, _ = ar_greedy_generate(model, PROMPT, max_tokens=20, use_cache=False)
    tokens, _ = uno_greedy_generate(
        model, PROMPT, max_tokens=20, block_size=4, use_cache=False
    )
    assert tokens == assert_varied(reference)


def test_stop_token_truncates_mid_block():
    model = _model(trained=True)
    reference, _ = ar_greedy_generate(model, PROMPT, max_tokens=40)
    assert_varied(reference)
    stop = reference[9]
    expected = reference[: reference.index(stop) + 1]
    tokens, _ = uno_greedy_generate(
        model, PROMPT, max_tokens=40, block_size=8, stop_ids={stop}
    )
    assert tokens == expected


@pytest.mark.parametrize("mode", ["snapshot", "rewind"])
@pytest.mark.parametrize("block_size", [2, 4, 8])
def test_transaction_modes_produce_identical_output(mode, block_size):
    """Act IV-U2 Gate U2-1: swapping the state backend must change nothing but cost."""
    model = _model(trained=True)
    reference, replay_stats = uno_greedy_generate(
        model, PROMPT, max_tokens=32, block_size=block_size,
        noise_mode="deterministic_uniform", noise_seed=5, transaction_mode="replay",
    )
    assert_varied(reference)
    tokens, stats = uno_greedy_generate(
        model, PROMPT, max_tokens=32, block_size=block_size,
        noise_mode="deterministic_uniform", noise_seed=5, transaction_mode=mode,
    )
    assert tokens == reference, f"{mode} diverged from replay at K={block_size}"
    assert stats.committed_per_cycle == replay_stats.committed_per_cycle
    assert stats.accepted_specs == replay_stats.accepted_specs


@pytest.mark.parametrize("block_size", [2, 4, 8])
def test_snapshot_removes_every_replay_forward(block_size):
    """Gate U2-2, counted explicitly rather than inferred from wall clock."""
    model = _model(trained=True)
    _, replay_stats = uno_greedy_generate(
        model, PROMPT, max_tokens=32, block_size=block_size,
        noise_mode="deterministic_uniform", noise_seed=5, transaction_mode="replay",
    )
    _, snap_stats = uno_greedy_generate(
        model, PROMPT, max_tokens=32, block_size=block_size,
        noise_mode="deterministic_uniform", noise_seed=5, transaction_mode="snapshot",
    )
    assert replay_stats.replay_forwards > 0, "no partial acceptances — test is vacuous"
    assert snap_stats.replay_forwards == 0
    assert snap_stats.forwards == 1 + 2 * snap_stats.cycles
    assert snap_stats.forwards < replay_stats.forwards
    assert snap_stats.tokens_per_forward > replay_stats.tokens_per_forward


# ------------------------------------ Act IV-U4: benchmark noise as a named stream


@pytest.mark.parametrize("block_size", [2, 4, 8])
def test_bench_noise_stream_makes_decoding_reproducible(block_size):
    """Same seed, same draft noise, same acceptance — across separate calls."""
    model = _model(trained=True)
    a_tokens, a = uno_greedy_generate(
        model, PROMPT, max_tokens=24, block_size=block_size,
        transaction_mode="snapshot", noise_stream_seed=1234,
    )
    b_tokens, b = uno_greedy_generate(
        model, PROMPT, max_tokens=24, block_size=block_size,
        transaction_mode="snapshot", noise_stream_seed=1234,
    )
    assert a_tokens == b_tokens
    assert a.accepted_per_cycle == b.accepted_per_cycle
    assert a.committed_per_cycle == b.committed_per_cycle


def test_bench_noise_stream_consumes_no_global_rng():
    """Benchmarking one checkpoint must not shift the noise the next one sees."""
    model = _model(trained=True)
    mx.random.seed(17)
    reference = mx.random.uniform(shape=(4,))

    mx.random.seed(17)
    for _ in range(3):
        uno_greedy_generate(model, PROMPT, max_tokens=16, block_size=4,
                            transaction_mode="snapshot", noise_stream_seed=99)
    assert mx.array_equal(reference, mx.random.uniform(shape=(4,)))

    # non-vacuity: without a stream seed the draft noise *does* move the global stream
    mx.random.seed(17)
    for _ in range(3):
        uno_greedy_generate(model, PROMPT, max_tokens=16, block_size=4,
                            transaction_mode="snapshot")
    assert not mx.array_equal(reference, mx.random.uniform(shape=(4,)))


def test_different_bench_seeds_give_different_noise():
    """Asserted on the draft blocks, not on acceptance.

    The tiny fixture's random adapter accepts nothing at K=8, so `accepted_per_cycle`
    is all zeros whatever the noise is: an assertion there would pass or fail for
    reasons unrelated to the stream. Compare the noise itself.
    """
    from qdif.uno.decode import _cycle_key
    from qdif.uno.noise import build_draft_block

    seeds = mx.array([PROMPT[-1]], dtype=mx.int32)

    def block(stream_seed, cycle):
        return build_draft_block(
            seeds, 8, "random_uniform", 480, 512, key=_cycle_key(stream_seed, cycle)
        ).tolist()

    for cycle in range(1, 6):
        assert block(1, cycle) != block(2, cycle)


def test_bench_noise_is_indexed_by_cycle_not_by_call_order():
    """Cycle c must see the same noise however many cycles a run took to get there.

    Two checkpoints accept different numbers of tokens and therefore run different
    numbers of cycles. If the noise were chained rather than indexed, they would be
    measured against different noise from their first divergence onward and part of
    any acceptance difference would be the draw.
    """
    from qdif.uno.decode import _cycle_key
    from qdif.uno.noise import make_noise

    def block(cycle):
        return make_noise((1, 3), "random_uniform", 480, 512,
                          key=_cycle_key(555, cycle)).tolist()

    assert block(7) == block(7)
    assert block(7) != block(8)
    # ...and reaching cycle 7 via other cycles first changes nothing
    for c in range(7):
        block(c)
    assert block(7) == block(7)


def test_uno_stays_lossless_with_an_explicit_noise_stream():
    """Gate 3 must not depend on where the draft noise came from."""
    model = _model(trained=True)
    reference, _ = ar_greedy_generate(model, PROMPT, max_tokens=24)
    assert_varied(reference)
    for seed in (0, 7, 20260904):
        tokens, _ = uno_greedy_generate(
            model, PROMPT, max_tokens=24, block_size=4,
            transaction_mode="snapshot", noise_stream_seed=seed,
        )
        assert tokens == reference


# ------------------------------------------- Act IV-U4: cost model v2 accounting


@pytest.mark.parametrize("block_size", [2, 4, 8])
def test_cycle_time_is_fully_attributed(block_size):
    """proposal + verify + commit + overhead must account for the cycle loop.

    U3's cost model over-predicted throughput at every checkpoint by a consistent
    +3.9-10.2%, which is what a missing additive term looks like. This asserts the
    stages now tile the cycle, so the v2 model has nowhere left to hide a bias.
    """
    model = _model(trained=True)
    _, stats = uno_greedy_generate(
        model, PROMPT, max_tokens=32, block_size=block_size,
        transaction_mode="snapshot", noise_stream_seed=3,
    )
    accounted = (
        stats.proposal_seconds + stats.verify_seconds
        + stats.commit_seconds + stats.overhead_seconds
    )
    assert stats.cycle_seconds > 0
    assert stats.overhead_seconds > 0, "overhead must be measured, not assumed zero"
    assert accounted <= stats.cycle_seconds + 1e-6
    assert stats.unattributed_seconds < 0.05 * stats.cycle_seconds, (
        f"{stats.unattributed_seconds:.4f}s of {stats.cycle_seconds:.4f}s unattributed"
    )
    # the cycle loop is the bulk of the run, the rest being prefill
    assert stats.cycle_seconds <= stats.wall_seconds + 1e-6


def test_stats_dict_exposes_the_v2_cost_terms():
    model = _model(trained=True)
    _, stats = uno_greedy_generate(
        model, PROMPT, max_tokens=16, block_size=4,
        transaction_mode="snapshot", noise_stream_seed=4,
    )
    payload = stats.to_dict()
    for key in ("overhead_ms_per_cycle", "unattributed_ms_per_cycle",
                "overhead_seconds", "cycle_seconds"):
        assert key in payload
    assert payload["overhead_ms_per_cycle"] > 0
