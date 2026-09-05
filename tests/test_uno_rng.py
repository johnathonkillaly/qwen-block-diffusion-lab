"""PRNG stream separation and positional determinism (Act IV-U4 §2).

Act IV-U3's gate U3-0b failed because evaluation and training shared the global MLX
stream. The metric that would have revealed it — the K=4 teacher-forced agreement —
matched to four decimal places between the two runs; the divergence was visible only in
the adapter tensors. So the tests here assert on the *mechanism*, not on a downstream
number, and each one includes a non-vacuity check.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from qdif.uno.rng import (
    BENCH_NOISE,
    DATA_SAMPLE,
    EVAL_NOISE,
    STREAM_NAMES,
    TRAIN_NOISE,
    TRAIN_RATE,
    numpy_seed,
    stream_key,
    stream_report,
    stream_seed,
)


def _draw(key):
    return mx.random.randint(0, 1_000_000, (32,), key=key).tolist()


def test_streams_are_disjoint():
    """No two named streams may produce the same draws at the same step."""
    streams = list(STREAM_NAMES)
    seen = {}
    for stream in streams:
        values = tuple(_draw(stream_key(2026, stream)))
        assert values not in seen, (
            f"{STREAM_NAMES[stream]} collides with {STREAM_NAMES[seen[values]]}"
        )
        seen[values] = stream
    assert len(seen) == len(streams)


def test_draw_depends_only_on_seed_stream_step():
    a = _draw(stream_key(7, TRAIN_NOISE, 1234))
    b = _draw(stream_key(7, TRAIN_NOISE, 1234))
    assert a == b
    assert a != _draw(stream_key(8, TRAIN_NOISE, 1234))
    assert a != _draw(stream_key(7, TRAIN_RATE, 1234))
    assert a != _draw(stream_key(7, TRAIN_NOISE, 1235))


def test_adjacent_steps_decorrelate():
    """Folding the step in must hash, not offset.

    If `stream_key(s, S, t)` were `mx.random.key(base + t)` on a generator that did not
    mix its seed, consecutive steps would produce overlapping sequences and the
    training noise would be correlated across steps in a way nothing downstream checks.
    """
    a = _draw(stream_key(11, TRAIN_NOISE, 500))
    b = _draw(stream_key(11, TRAIN_NOISE, 501))
    overlap = len(set(a) & set(b))
    assert overlap <= 2, f"{overlap}/32 draws shared between adjacent steps"


def test_stream_key_consumes_no_global_state():
    mx.random.seed(3)
    reference = mx.random.uniform(shape=(4,))
    mx.random.seed(3)
    for step in range(50):
        _draw(stream_key(99, EVAL_NOISE, step))
    assert mx.array_equal(reference, mx.random.uniform(shape=(4,)))

    # non-vacuity: an unkeyed draw of the same shape *does* move the global stream
    mx.random.seed(3)
    for _ in range(50):
        mx.random.randint(0, 1_000_000, (32,))
    assert not mx.array_equal(reference, mx.random.uniform(shape=(4,)))


def test_numpy_seed_is_in_range_and_stream_specific():
    a = numpy_seed(5, DATA_SAMPLE)
    b = numpy_seed(5, BENCH_NOISE)
    assert 0 <= a < 2**32 and 0 <= b < 2**32
    assert a != b


def test_stream_report_names_every_stream():
    report = stream_report(20260903)
    assert report["seed"] == 20260903
    assert set(report["streams"]) == set(STREAM_NAMES.values())
    assert len(set(report["streams"].values())) == len(STREAM_NAMES)


def test_stream_seed_is_64_bit():
    for step in (0, 1, 10**6):
        value = stream_seed(20260903, TRAIN_NOISE, step)
        assert 0 <= value < 2**64


@pytest.mark.parametrize("stream", sorted(STREAM_NAMES))
def test_every_stream_has_a_name(stream):
    assert STREAM_NAMES[stream]
