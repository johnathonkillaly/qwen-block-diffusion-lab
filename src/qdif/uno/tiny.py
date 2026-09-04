"""A tiny randomly-initialised Qwen3.5 for tests.

The point is fidelity of *structure*, not of weights: this builds a real
`mlx_lm.models.qwen3_5.Model` with the genuine hybrid layer stack — Gated DeltaNet
layers interleaved with full-attention layers, the same caches, the same rope — at a
size that loads in well under a second.

That matters because the interesting failure modes in this experiment are structural
(cache rewind on a recurrence, mask alignment, adapter gating), not numerical. Testing
them against a 4B checkpoint would make the suite unrunnable; testing them against a
mock transformer would not exercise the recurrence at all.
"""

from __future__ import annotations

import mlx.core as mx

from .model import UnoModel

TINY_CONFIG = {
    "model_type": "qwen3_5",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,  # -> layers 3 is full attention, 0/1/2 are DeltaNet
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 512,
    "rms_norm_eps": 1e-6,
    # Deliberately UNTIED, unlike the real Qwen3.5-4B-Base.
    #
    # With tied embeddings a randomly-initialised model of this size is degenerate:
    # its greedy continuation is a single token repeated forever. Every decoding test
    # then passes vacuously, because an off-by-one in a constant sequence is
    # invisible. That is not hypothetical — it is exactly how a real indexing bug in
    # the verifier survived the whole suite and had to be caught on the 4B backbone.
    # An untied readout gives ~19 distinct tokens in 20, and `assert_varied` below
    # makes the guard explicit rather than incidental.
    "tie_word_embeddings": False,
    "full_attention_interval": 4,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "max_position_embeddings": 4096,
    "rope_parameters": {"type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
}


def build_tiny_model(seed: int = 0, mask_token_id: int = 480, **overrides) -> UnoModel:
    """A `UnoModel` around a tiny hybrid backbone with reproducible random weights."""
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    mx.random.seed(seed)
    config = dict(TINY_CONFIG)
    config.update(overrides)
    model = Model(ModelArgs(model_type="qwen3_5", text_config=config))
    mx.eval(model.parameters())
    return UnoModel(model, tokenizer=None, mask_token_id=mask_token_id)


def assert_varied(tokens: list[int], minimum: int = 5) -> list[int]:
    """Refuse to let a degenerate reference sequence make a test vacuous.

    A decoding test that compares two constant sequences proves nothing. Any test
    asserting "Uno output == AR output" must route its reference through this.
    """
    distinct = len(set(tokens))
    if distinct < minimum:
        raise AssertionError(
            f"reference continuation has only {distinct} distinct tokens "
            f"(need >= {minimum}); the model is degenerate and this test would "
            f"pass vacuously. Sequence: {tokens[:16]}"
        )
    return tokens


def random_windows(batch: int, width: int, vocab: int, seed: int = 0) -> mx.array:
    """Random token windows for smoke tests. Ids stay clear of the noise band."""
    mx.random.seed(seed)
    return mx.random.randint(1, min(vocab, 400), (batch, width)).astype(mx.int32)
