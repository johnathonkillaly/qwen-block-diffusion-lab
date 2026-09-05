"""Named, separated PRNG streams.

Act IV-U3's pre-registered gate U3-0b failed for a reason worth encoding in a module:
evaluation and training were drawing from the **same global MLX stream**, so changing
only the *evaluation schedule* (`eval_every` 50 -> 100, `eval_batches` 4 -> 8) shifted
every subsequent training noise draw. Two runs with identical seeds and identical
training configuration diverged in all 256 adapter tensors. The headline metric matched
to four decimal places, so the defect was invisible to every metric we report and was
caught only by comparing tensors directly.

The fix is not "remember to pass a key". It is to make every stochastic component name
its stream, and to derive that stream's key from `(seed, stream, step)` by hashing:

    key = mx.random.key(splitmix64(seed, stream, step))

Two properties follow, and both are tested in `tests/test_uno_rng.py`:

1. **Isolation.** Two different streams never share draws, and consuming one does not
   advance another. Nothing about the evaluation schedule can reach training.
2. **Positional determinism.** The draw at step `s` depends only on `(seed, stream, s)`,
   never on how many draws came before it. A run resumed from a step-3200 checkpoint
   therefore sees exactly the noise an uninterrupted run would have seen at step 3200,
   which is what makes `--resume-from` a continuation rather than a restart wearing a
   continuation's clothes.

Property 2 is the one that matters for Act IV-U4. Without it, "resume" can only mean
"start a fresh noise stream from here", and the burden of proving that the join is
harmless falls on prose rather than on a test.
"""

from __future__ import annotations

import mlx.core as mx

_MASK64 = 0xFFFFFFFFFFFFFFFF

#: Stream identifiers. The values are arbitrary but must never be reused or reordered:
#: changing one changes the noise a given `(seed, step)` produces, which would silently
#: break the "resume reproduces the uninterrupted run" guarantee for existing runs.
TRAIN_NOISE = 0x5452_4149_4E5F_4E5A  # "TRAIN_NZ"
TRAIN_RATE = 0x5452_4149_4E5F_5254  # "TRAIN_RT"
EVAL_NOISE = 0x4556_414C_5F_4E5A_5A  # eval corruption
EVAL_SAMPLE = 0x4556_414C_5F_5350_4C  # eval window sampling
DATA_SAMPLE = 0x4441_5441_5F_5350_4C  # training window sampling
BENCH_NOISE = 0x4245_4E43_485F_4E5A  # decode-time draft noise

STREAM_NAMES = {
    TRAIN_NOISE: "train_noise",
    TRAIN_RATE: "train_corruption_rate",
    EVAL_NOISE: "eval_noise",
    EVAL_SAMPLE: "eval_sampling",
    DATA_SAMPLE: "data_sampling",
    BENCH_NOISE: "bench_noise",
}


def _mix64(value: int) -> int:
    """splitmix64. Same finalizer `noise.py` uses, kept local to avoid a cycle."""
    value &= _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def stream_seed(seed: int, stream: int, step: int = 0) -> int:
    """A 64-bit seed for `(seed, stream, step)`, decorrelated in every argument."""
    mixed = _mix64(int(seed) * 0x9E3779B97F4A7C15)
    mixed = _mix64(mixed ^ (int(stream) * 0xC2B2AE3D27D4EB4F))
    return _mix64(mixed ^ (int(step) * 0x165667B19E3779F9))


def stream_key(seed: int, stream: int, step: int = 0) -> mx.array:
    """The MLX PRNG key for `(seed, stream, step)`. Consumes no global state."""
    return mx.random.key(stream_seed(seed, stream, step))


def numpy_seed(seed: int, stream: int, step: int = 0) -> int:
    """A 32-bit seed for numpy's `default_rng`, from the same hash."""
    return stream_seed(seed, stream, step) & 0xFFFFFFFF


def stream_report(seed: int) -> dict:
    """What every stream's base key is, for the run record.

    Written into checkpoints and result files so that "which streams existed and how
    were they derived" is answerable from an artifact rather than from the source tree
    at some unknown commit.
    """
    return {
        "seed": int(seed),
        "streams": {
            name: f"{stream_seed(seed, stream):#018x}"
            for stream, name in STREAM_NAMES.items()
        },
    }
