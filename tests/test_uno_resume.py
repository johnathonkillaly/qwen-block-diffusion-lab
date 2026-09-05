"""Exact resume: a continuation must be indistinguishable from never stopping.

Act IV-U4 §2 asks for checkpoints that are *actually* resumable. The bar this file sets
is the strongest available one and the only one that cannot be satisfied by accident:

    train N steps straight through
    ==  train N/2, checkpoint, reload, train the remaining N/2

with **bitwise** equality of every adapter tensor. Anything weaker — comparing losses,
comparing a metric to three decimals — is exactly the check that let Act IV-U3's gate
U3-0b through until the tensors themselves were compared.

Four pieces of state have to be right for this to hold, and each has its own test
below, so a failure says which one broke:

  1. adapter weights          (`load_checkpoint`)
  2. AdamW moments            (`load_checkpoint`)
  3. training noise           keyed on `(seed, TRAIN_NOISE, step)`, not on draw order
  4. data position            `WindowSource.fast_forward` / `load_state_dict`
"""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from qdif.uno.data import WindowSource
from qdif.uno.tiny import build_tiny_model, random_windows
from qdif.uno.trainer import UnoTrainConfig, config_hash, train_uno


def _source(seed: int = 5, pool: int = 40, batch: int = 4, width: int = 24):
    windows = np.asarray(random_windows(pool, width, 400, seed=seed).tolist(), dtype=np.int64)
    return WindowSource(windows, batch, seed=seed)


def _config(**kwargs) -> UnoTrainConfig:
    base = dict(
        block_size=4, steps=8, batch_size=4, window=24, lora_rank=4,
        learning_rate=1e-4, warmup_steps=2, eval_every=0, log_every=0, seed=4242,
    )
    base.update(kwargs)
    return UnoTrainConfig(**base)


def _adapter(model) -> dict:
    return {
        name: value
        for name, value in tree_flatten(model.base.trainable_parameters())
        if name.endswith(("lora_a", "lora_b"))
    }


# ------------------------------------------------------------------ the big one


def test_resumed_run_is_bitwise_identical_to_an_uninterrupted_one(tmp_path):
    straight = build_tiny_model(seed=71)
    train_uno(
        straight, _source(), _source(9), _config(steps=8),
        output_dir=tmp_path / "straight", echo=lambda *a: None,
    )
    reference = _adapter(straight)

    first = build_tiny_model(seed=71)
    train_uno(
        first, _source(), _source(9), _config(steps=4, checkpoint_steps=[4]),
        output_dir=tmp_path / "split", echo=lambda *a: None,
    )

    second = build_tiny_model(seed=71)
    train_source = _source().fast_forward(4)
    train_uno(
        second, train_source, _source(9), _config(steps=8),
        output_dir=tmp_path / "split", echo=lambda *a: None,
        resume_from=tmp_path / "split" / "step-4",
    )
    resumed = _adapter(second)

    assert set(resumed) == set(reference)
    mismatched = [
        name for name, value in reference.items()
        if not mx.array_equal(value, resumed[name])
    ]
    assert not mismatched, (
        f"{len(mismatched)}/{len(reference)} adapter tensors differ after resume; "
        f"first: {mismatched[:3]}"
    )


def test_the_resume_test_can_fail(tmp_path):
    """Non-vacuity: a resume that skips the data fast-forward must NOT match.

    Without this, the test above would still pass if the two arms happened to train on
    identical data by construction, and it would be checking nothing.
    """
    straight = build_tiny_model(seed=72)
    train_uno(straight, _source(), _source(9), _config(steps=8), echo=lambda *a: None)
    reference = _adapter(straight)

    first = build_tiny_model(seed=72)
    train_uno(
        first, _source(), _source(9), _config(steps=4, checkpoint_steps=[4]),
        output_dir=tmp_path / "wrong", echo=lambda *a: None,
    )
    second = build_tiny_model(seed=72)
    train_uno(
        second, _source(), _source(9), _config(steps=8),  # no fast_forward
        echo=lambda *a: None, resume_from=tmp_path / "wrong" / "step-4",
    )
    assert any(
        not mx.array_equal(value, _adapter(second)[name])
        for name, value in reference.items()
    ), "resuming on the wrong data position produced identical weights — test is vacuous"


# -------------------------------------------------------- the four state pieces


def test_training_noise_is_positional_not_sequential():
    """Step 400's noise must not depend on whether steps 0-399 were run."""
    from qdif.uno.rng import TRAIN_NOISE, stream_key
    from qdif.uno.teacher import build_uno_batch

    windows = random_windows(4, 24, 400, seed=73)

    def batch_at(step):
        return build_uno_batch(
            windows, 4, 480, 512, corruption="uniform",
            key=stream_key(4242, TRAIN_NOISE, step),
        )

    direct = batch_at(400)
    for step in range(390, 400):
        batch_at(step)
    after_replay = batch_at(400)
    assert mx.array_equal(direct.student_ids, after_replay.student_ids)
    assert mx.array_equal(direct.corrupted, after_replay.corrupted)
    # non-vacuity: a different step really does give different noise
    assert not mx.array_equal(direct.student_ids, batch_at(401).student_ids)


def test_window_source_fast_forward_matches_actual_draws():
    walked = _source()
    drawn = [next(walked) for _ in range(13)]
    jumped = _source().fast_forward(12)
    assert mx.array_equal(drawn[12], next(jumped))
    assert jumped.draws == 13 == walked.draws


def test_fast_forward_reproduces_reshuffles():
    """The pool is 40 windows at batch 4, so 10 draws force a reshuffle."""
    walked = _source()
    drawn = [next(walked) for _ in range(25)]
    jumped = _source().fast_forward(24)
    assert mx.array_equal(drawn[24], next(jumped))


def test_window_source_state_round_trip():
    source = _source()
    for _ in range(17):
        next(source)
    state = source.state_dict()
    assert state["draws"] == 17

    restored = _source().load_state_dict(state)
    assert mx.array_equal(next(source), next(restored))


def test_window_source_state_rejects_a_mismatched_pool():
    state = _source().state_dict()
    with pytest.raises(ValueError, match="windows"):
        _source(pool=80).load_state_dict(state)


def test_window_source_state_rejects_a_mismatched_batch_size():
    state = _source().state_dict()
    with pytest.raises(ValueError, match="batch size"):
        _source(batch=8).load_state_dict(state)


# ------------------------------------------------------------ checkpoint payload


def test_checkpoint_contains_every_field_u4_requires(tmp_path):
    model = build_tiny_model(seed=74)
    train_uno(
        model, _source(), _source(9), _config(steps=4, checkpoint_steps=[4]),
        output_dir=tmp_path, echo=lambda *a: None,
    )
    state = json.loads((tmp_path / "step-4" / "checkpoint_state.json").read_text())

    assert state["step"] == 4
    assert state["optimizer_tensors"] > 0          # optimizer state
    assert state["lr"] > 0                         # scheduler state
    assert state["block_size"] == 4
    assert len(state["config_hash"]) == 64         # config hash
    assert state["rng"]["seed"] == 4242            # PRNG streams
    assert set(state["rng"]["streams"]) >= {"train_noise", "eval_noise", "bench_noise"}
    assert state["data_position"]["train"]["draws"] == 4   # data position
    assert (tmp_path / "step-4" / "adapter.safetensors").is_file()
    assert (tmp_path / "step-4" / "optimizer.safetensors").is_file()


def test_config_hash_ignores_run_length_but_not_the_objective():
    base = _config(steps=8)
    assert config_hash(base) == config_hash(_config(steps=99999))
    assert config_hash(base) == config_hash(_config(steps=8, checkpoint_steps=[1, 2]))
    assert config_hash(base) == config_hash(_config(steps=8, eval_every=7))
    assert config_hash(base) != config_hash(_config(steps=8, block_size=8))
    assert config_hash(base) != config_hash(_config(steps=8, learning_rate=1e-3))
    assert config_hash(base) != config_hash(_config(steps=8, corruption="full"))


def test_resume_past_the_end_is_an_error(tmp_path):
    model = build_tiny_model(seed=75)
    train_uno(
        model, _source(), _source(9), _config(steps=4, checkpoint_steps=[4]),
        output_dir=tmp_path, echo=lambda *a: None,
    )
    with pytest.raises(ValueError, match="nothing to do"):
        train_uno(
            build_tiny_model(seed=75), _source(), _source(9), _config(steps=4),
            echo=lambda *a: None, resume_from=tmp_path / "step-4",
        )


def test_resume_records_its_provenance(tmp_path):
    model = build_tiny_model(seed=76)
    train_uno(
        model, _source(), _source(9), _config(steps=4, checkpoint_steps=[4]),
        output_dir=tmp_path, echo=lambda *a: None,
    )
    result = train_uno(
        build_tiny_model(seed=76), _source().fast_forward(4), _source(9),
        _config(steps=6), echo=lambda *a: None, resume_from=tmp_path / "step-4",
    )
    assert result["resume"]["start_step"] == 4
    assert result["resume"]["end_step"] == 6
    assert result["resume"]["resumed_from"].endswith("step-4")
    assert len(result["history"]) == 2
    assert result["history"][0]["step"] == 4
    assert result["data_position"]["train"]["draws"] == 6
