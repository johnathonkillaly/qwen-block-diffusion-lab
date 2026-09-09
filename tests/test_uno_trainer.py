"""Training loop: integrity enforcement, the saturation guard, and adapter I/O."""

from __future__ import annotations

import itertools
import json

import mlx.core as mx
import pytest

from qdif.uno.metrics import entropy_buckets, pearson, slot_metrics
from qdif.uno.tiny import build_tiny_model, random_windows
from qdif.uno.trainer import UnoTrainConfig, load_adapter, save_adapter, train_uno


def _windows(seed: int = 0, batch: int = 4, width: int = 24):
    counter = itertools.count()
    while True:
        yield random_windows(batch, width, 400, seed=seed + next(counter))


def _config(**kwargs) -> UnoTrainConfig:
    base = dict(
        block_size=4, steps=6, batch_size=4, window=24, lora_rank=4,
        learning_rate=1e-4, warmup_steps=2, eval_every=0, log_every=0,
    )
    base.update(kwargs)
    return UnoTrainConfig(**base)


def test_training_runs_and_leaves_the_backbone_untouched():
    model = build_tiny_model(seed=51)
    result = train_uno(model, _windows(0), _windows(900), _config(), echo=lambda *a: None)
    assert result["integrity"]["backbone_unchanged"]
    assert result["integrity"]["digest_before"] == result["integrity"]["digest_after"]
    assert len(result["history"]) == 6
    assert all(record["grad_norm"] >= 0 for record in result["history"])


def test_only_the_adapter_is_trainable_during_training():
    from mlx.utils import tree_flatten

    model = build_tiny_model(seed=52)
    train_uno(model, _windows(0), _windows(900), _config(), echo=lambda *a: None)
    names = [n for n, _ in tree_flatten(model.base.trainable_parameters())]
    assert names and all(n.endswith(("lora_a", "lora_b")) for n in names)
    assert model.parameter_report()["percent_trainable"] < 100.0


def test_saturation_guard_raises_instead_of_reporting_a_null_result():
    """A runaway LR must fail loudly, not freeze at constant loss with |g| = 0.

    This is the exact failure that produced a plausible-looking "the adapter learned
    nothing" run at lr 3e-4 on the 4B backbone.
    """
    model = build_tiny_model(seed=53)
    with pytest.raises(RuntimeError, match="training collapsed|non-finite"):
        train_uno(
            model,
            _windows(0),
            _windows(900),
            _config(steps=60, learning_rate=5.0, warmup_steps=0, grad_clip=0.0),
            echo=lambda *a: None,
        )


def test_history_records_the_diagnostic_that_identifies_saturation():
    model = build_tiny_model(seed=54)
    result = train_uno(model, _windows(0), _windows(900), _config(), echo=lambda *a: None)
    assert all("student_logit_absmax" in record for record in result["history"])


def test_block_curriculum_walks_the_configured_sizes():
    config = _config(steps=12, block_curriculum=[2, 4, 8])
    assert config.block_size_at(0) == 2
    assert config.block_size_at(5) == 4
    assert config.block_size_at(11) == 8


def test_alpha_defaults_to_sixteen_times_rank_like_ifm():
    assert _config(lora_rank=16).resolved_alpha() == 256
    assert _config(lora_rank=128).resolved_alpha() == 2048
    assert _config(lora_rank=16, lora_alpha=32).resolved_alpha() == 32


def test_adapter_round_trip_preserves_behaviour(tmp_path):
    model = build_tiny_model(seed=55)
    train_uno(model, _windows(0), _windows(900), _config(), echo=lambda *a: None)
    ids = random_windows(2, 16, model.vocab_size, seed=56)
    mask = mx.ones(ids.shape, dtype=mx.float32)
    before = model.draft_logits(ids, mask)

    path = tmp_path / "adapter.safetensors"
    count = save_adapter(model, path)
    assert count == len(model.adapters) * 2

    fresh = build_tiny_model(seed=55)
    fresh.attach_adapter(rank=4, alpha=64, full_attention=True, mlp=True)
    assert not mx.array_equal(before, fresh.draft_logits(ids, mask))
    load_adapter(fresh, path)
    assert mx.array_equal(before, fresh.draft_logits(ids, mask))


def test_saved_adapter_contains_no_backbone_tensors(tmp_path):
    model = build_tiny_model(seed=57)
    model.attach_adapter(rank=4)
    path = tmp_path / "adapter.safetensors"
    save_adapter(model, path)
    names = list(mx.load(str(path)).keys())
    assert names
    assert all(n.endswith(("lora_a", "lora_b")) for n in names)


# ------------------------------------------------------------------- metrics


def test_slot_metrics_report_perfect_agreement_with_the_teacher():
    logits = mx.random.normal((3, 4, 20))
    targets = mx.argmax(logits, axis=-1)
    metrics = slot_metrics(logits, logits, targets)
    assert metrics["agree_specs"] == pytest.approx(1.0)
    assert metrics["mean_accepted_prefix"] == pytest.approx(3.0)
    assert metrics["full_block_rate"] == pytest.approx(1.0)
    assert metrics["gold_specs"] == pytest.approx(1.0)


def test_accepted_prefix_stops_at_the_first_disagreement():
    """Agreement at slots 1 and 3 but not 2 must count as a prefix of 1, not 2."""
    teacher = mx.zeros((1, 4, 5))
    teacher = teacher + mx.array([0.0, 1.0, 0.0, 0.0, 0.0])  # argmax 1 everywhere
    student = mx.array(
        [[[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0]]],
        dtype=mx.float32,
    )
    metrics = slot_metrics(student, teacher, mx.zeros((1, 4), dtype=mx.int32))
    assert metrics["mean_accepted_prefix"] == pytest.approx(1.0)
    assert metrics["full_block_rate"] == pytest.approx(0.0)


def test_entropy_buckets_are_ordered_by_teacher_entropy():
    mx.random.seed(3)
    student = mx.random.normal((40, 12))
    teacher = mx.random.normal((40, 12)) * mx.arange(1, 41).reshape(40, 1) * 0.2
    buckets = entropy_buckets(student, teacher, num_buckets=4)
    assert len(buckets) == 4
    entropies = [b["mean_entropy"] for b in buckets]
    assert entropies == sorted(entropies)
    assert sum(b["n"] for b in buckets) == 40


def test_pearson_matches_known_values():
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) == 0.0
    assert pearson([1, 2], []) == 0.0


def test_checkpoint_round_trip_restores_adapter_and_optimizer(tmp_path):
    """A checkpoint must be resumable, not merely evaluable.

    Act IV-U saved only the adapter tensors, so "continuing" a run would have meant a
    fresh AdamW with zeroed moments on a different data order. This test exists so that
    regression cannot happen silently.
    """
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from qdif.uno.trainer import load_checkpoint, save_checkpoint

    model = build_tiny_model(seed=81)
    model.attach_adapter(rank=4, full_attention=True, mlp=True)
    optimizer = optim.AdamW(learning_rate=1e-3)

    # take a couple of steps so the optimizer has non-trivial moments
    import mlx.nn as nn

    from qdif.uno.losses import total_variation
    from qdif.uno.teacher import build_uno_batch

    windows = random_windows(4, 24, 400, seed=82)
    batch = build_uno_batch(windows, 4, model.mask_token_id, model.vocab_size, corruption="full")
    teacher = mx.stop_gradient(model.ar_logits(batch.teacher_ids)[:, batch.supervised_slice, :])

    def forward():
        student = model.draft_logits(batch.student_ids, batch.lora_mask)[
            :, batch.supervised_slice, :
        ]
        return total_variation(student, teacher)

    for _ in range(3):
        loss, grads = nn.value_and_grad(model.base, forward)()
        optimizer.update(model.base, grads)
        mx.eval(model.base.parameters(), optimizer.state)

    saved = save_checkpoint(model, optimizer, 3, tmp_path / "step-3")
    assert (saved / "adapter.safetensors").is_file()
    assert (saved / "optimizer.safetensors").is_file()
    state = json.loads((saved / "checkpoint_state.json").read_text())
    assert state["step"] == 3 and state["optimizer_tensors"] > 0

    reference = model.draft_logits(batch.student_ids, batch.lora_mask)
    optimizer_reference = {
        name: value for name, value in tree_flatten(optimizer.state)
        if isinstance(value, mx.array)
    }

    fresh = build_tiny_model(seed=81)
    fresh.attach_adapter(rank=4, alpha=64, full_attention=True, mlp=True)
    fresh_optimizer = optim.AdamW(learning_rate=1e-3)
    assert not mx.array_equal(reference, fresh.draft_logits(batch.student_ids, batch.lora_mask))

    load_checkpoint(fresh, fresh_optimizer, saved)
    assert mx.array_equal(reference, fresh.draft_logits(batch.student_ids, batch.lora_mask))
    restored = {
        name: value for name, value in tree_flatten(fresh_optimizer.state)
        if isinstance(value, mx.array)
    }
    assert set(restored) == set(optimizer_reference)
    for name, value in optimizer_reference.items():
        assert mx.array_equal(value, restored[name]), f"optimizer moment {name} not restored"


def test_training_writes_checkpoints_at_configured_steps(tmp_path):
    model = build_tiny_model(seed=83)
    config = _config(steps=6, checkpoint_steps=[2, 4])
    train_uno(model, _windows(0), _windows(900), config,
              output_dir=tmp_path, echo=lambda *a: None)
    assert (tmp_path / "step-2" / "adapter.safetensors").is_file()
    assert (tmp_path / "step-4" / "optimizer.safetensors").is_file()
    assert not (tmp_path / "step-6").exists()  # 6 not requested
