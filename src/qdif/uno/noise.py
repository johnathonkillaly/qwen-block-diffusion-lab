"""Draft-block noise, matching `nano_vllm_uno/engine/noise.py`.

Uno's public training runs with `noise="uniform"`: corrupted rows hold **random token
ids drawn from `[1, mask_token_id)`**, not a single absorbing `[MASK]`. That is worth
stating twice, because it is the opposite of what Act III found worked for canvas
denoising -- and the two results do not conflict. See `docs/act4u_uno_source_notes.md`
§6: here the noise token is a placeholder the adapter must learn to *ignore*, not a
slot it must reconstruct from.

`mask` mode is kept selectable so that "uniform beats mask for this objective" is
something we measure rather than assume.
"""

from __future__ import annotations

import mlx.core as mx

NOISE_MODES = ("random_uniform", "deterministic_uniform", "mask")


def _mix_u64(value: int) -> int:
    """splitmix64. Identical to `_mix_u64` in Uno's engine/noise.py."""
    value &= 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF


def noise_bounds(mask_token_id: int, vocab_size: int) -> tuple[int, int]:
    """The half-open id range for uniform noise: `[1, mask_token_id)`."""
    if mask_token_id is None or int(mask_token_id) <= 1:
        raise ValueError("uniform diffusion noise requires a valid mask_token_id > 1")
    return 1, min(int(mask_token_id), int(vocab_size))


def make_noise(
    shape: tuple[int, ...],
    mode: str,
    mask_token_id: int,
    vocab_size: int,
    seed: int | None = None,
    key: mx.array | None = None,
) -> mx.array:
    """Noise token ids of `shape`.

    `deterministic_uniform` reproduces a given block from `seed` alone, which is what
    makes adapter+verifier decoding bit-reproducible.

    `key` supplies an explicit PRNG key so the caller consumes **no global RNG state**.
    Evaluation must pass one: otherwise an eval draws from the same global stream as
    training, and merely changing the evaluation *schedule* silently changes the
    training trajectory. That defect made two runs with identical seeds diverge, and
    was caught by pre-registered gate U3-0b.
    """
    if mode not in NOISE_MODES:
        raise ValueError(f"Unsupported noise mode {mode!r}; expected one of {NOISE_MODES}")
    low, high = noise_bounds(mask_token_id, vocab_size)

    if mode == "mask":
        return mx.full(shape, high, dtype=mx.int32)
    if mode == "random_uniform":
        return mx.random.randint(low, high, shape, key=key).astype(mx.int32)

    if seed is None:
        raise ValueError("deterministic_uniform noise requires a seed")
    count = 1
    for dim in shape:
        count *= dim
    span = max(1, high - low)
    base = _mix_u64(int(seed) * 0x9E3779B185EBCA87)
    values = [
        low + (_mix_u64(base + slot * 0x27D4EB2F165667C5) % span) for slot in range(count)
    ]
    return mx.array(values, dtype=mx.int32).reshape(shape)


def build_draft_block(
    seed_tokens: mx.array,
    block_size: int,
    mode: str,
    mask_token_id: int,
    vocab_size: int,
    seed: int | None = None,
) -> mx.array:
    """`[seed, noise_1, ..., noise_{L-1}]`, one row per sequence.

    The seed is the last committed token. Its row runs with the adapter **off**, so
    the model's prediction at that position is the exact frozen-AR next token.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    batch = seed_tokens.shape[0]
    seeds = seed_tokens.reshape(batch, 1).astype(mx.int32)
    if block_size == 1:
        return seeds
    noise = make_noise(
        (batch, block_size - 1), mode, mask_token_id, vocab_size, seed=seed
    )
    return mx.concatenate([seeds, noise], axis=1)


def draft_lora_mask(batch: int, block_size: int, prefix_len: int = 0) -> mx.array:
    """The Uno gating pattern: adapter off on the prefix and the seed, on elsewhere.

    Mirrors `two_pass_decoding.py:draft_stage`, which builds a ones mask and then sets
    `lora_mask_batch[:, 0] = 0.0`. `prefix_len > 0` covers the uncached case, where the
    context tokens are in the same forward as the block and must also be adapter-free.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    prefix = mx.zeros((batch, prefix_len + 1), dtype=mx.float32)
    if block_size == 1:
        return prefix
    return mx.concatenate([prefix, mx.ones((batch, block_size - 1), dtype=mx.float32)], axis=1)
