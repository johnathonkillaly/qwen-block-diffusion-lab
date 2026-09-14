"""The Uno training loop.

Two forwards per step (teacher then student — see `teacher.py` for why one is not
possible on this architecture), TV between them, Adam on the adapter only.

Integrity is enforced structurally rather than by convention:

  * the backbone is frozen before the optimizer is constructed, so a base weight has
    no gradient tensor to receive an update;
  * the fingerprint is taken before the first step and after the last, and a mismatch
    raises rather than warns.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .losses import combine, cross_entropy, reverse_kl, total_variation
from .metrics import slot_metrics
from .rng import EVAL_NOISE, TRAIN_NOISE, stream_key, stream_report
from .teacher import build_uno_batch


@dataclass
class UnoTrainConfig:
    """Everything a run needs. Defaults follow IFM where a value is published."""

    block_size: int = 4
    #: IFM run a curriculum 2,4,6,8,12,16 (half an epoch each). `[]` means fixed.
    block_curriculum: list[int] = field(default_factory=list)
    steps: int = 200
    batch_size: int = 4
    window: int = 128
    # IFM's value, and it is not merely conservative — see the saturation guard below.
    # At 3e-4 the student's logits blow up (absmax 24 -> 76 in ten steps), the softmax
    # saturates to a one-hot, and TV's gradient  p_s·(sign − Σ sign·p_s)  vanishes
    # identically. Training then freezes at a constant loss with |g| = 0 and looks
    # like a null result rather than a divergence. Measured: 3e-4 dies, 5e-5 and 1e-5
    # both train.
    learning_rate: float = 1e-5
    warmup_steps: int = 20
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # objective — IFM defaults are TV-only
    tv_weight: float = 1.0
    ce_weight: float = 0.0
    kl_weight: float = 0.0
    ce_target: str = "data"  # "data" | "teacher_argmax"

    # corruption
    noise_mode: str = "random_uniform"
    corruption: str = "uniform"  # "uniform" (Uno/SDAR) | "full" (matches inference)
    #: Draw the draft-row corruption at this block width and keep the rightmost
    #: `block_size - 1` columns, so arms at different `block_size` see identical noise
    #: at every position they share. Act IV-U5 sets this to the largest block any arm
    #: trains at; `None` reproduces U3/U4 exactly. See `teacher.build_uno_batch`.
    noise_align_width: int | None = None

    # adapter
    lora_rank: int = 16
    lora_alpha: float | None = None  # None -> 16 * rank, IFM's convention
    lora_dropout: float = 0.0
    lora_full_attention: bool = True
    lora_mlp: bool = True
    lora_deltanet: bool = False
    lora_deltanet_gates: bool = False

    # controls
    shuffle_targets: bool = False  # Control 2
    seed: int = 20260903

    eval_every: int = 25
    eval_batches: int = 4
    log_every: int = 10

    #: Steps at which to write a full checkpoint (adapter + optimizer + RNG). Act IV-U
    #: saved only the adapter, which made exact resume impossible and forced Act IV-U3
    #: to re-run from scratch for a continuous trajectory. Do not repeat that.
    checkpoint_steps: list[int] = field(default_factory=list)

    def resolved_alpha(self) -> float:
        return 16.0 * self.lora_rank if self.lora_alpha is None else self.lora_alpha

    def block_size_at(self, step: int, start_step: int = 0) -> int:
        """The block size at absolute `step`, for a run that began at `start_step`.

        Stages divide the run's *own* span into equal parts. Measuring progress from
        step 0 instead would put a run resumed at 12,800 of 16,000 entirely inside the
        final stage, so a `4 -> 6 -> 8` curriculum would silently train only at 8 --
        a curriculum arm that is not a curriculum. IFM's stages are equal token
        budgets (`uno_3epoch_curriculum.yaml`), which for a fixed sequence length and
        batch size is equal steps.
        """
        if not self.block_curriculum:
            return self.block_size
        span = max(self.steps - start_step, 1)
        progress = max(step - start_step, 0)
        stage = min(
            len(self.block_curriculum) - 1,
            progress * len(self.block_curriculum) // span,
        )
        return self.block_curriculum[stage]

    def to_dict(self) -> dict:
        return asdict(self)


def _learning_rate(config: UnoTrainConfig, step: int) -> float:
    """Linear warmup, then flat. IFM disable decay to keep fine-tuning resumable."""
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    return config.learning_rate


def evaluate(
    model,
    windows_iter,
    config: UnoTrainConfig,
    block_size: int | None = None,
    num_batches: int | None = None,
) -> dict:
    """Teacher-forced evaluation at the corruption level decoding actually uses.

    Evaluation always runs `corruption="full"`: at inference every draft row holds
    noise, so a metric taken at a partially-clean block would flatter the adapter.
    """
    block_size = block_size or config.block_size
    num_batches = num_batches or config.eval_batches
    # A private key stream. Evaluation must not consume the global RNG that training
    # draws from -- see `rng.py` and gate U3-0b.
    #
    # Deliberately still the U3 formula rather than `stream_key(seed, EVAL_NOISE)`:
    # the in-training eval curve is plotted continuously across the U3 run and its U4
    # continuation, and changing the eval noise at step 3200 would put a step in that
    # curve that looks like an effect of training. The stream is already isolated,
    # which is the property that matters; `EVAL_NOISE` names it in the run record.
    eval_key = mx.random.key(config.seed + 999_983)
    totals: dict[str, list] = {}
    rows = 0
    tv_total = 0.0

    for index in range(num_batches):
        windows = next(windows_iter)
        batch = build_uno_batch(
            windows,
            block_size=block_size,
            mask_token_id=model.mask_token_id,
            vocab_size=model.vocab_size,
            noise_mode=config.noise_mode,
            corruption="full",
            shuffle_targets=config.shuffle_targets,
            key=mx.random.split(eval_key, num_batches)[index],
            noise_align_width=config.noise_align_width,
        )
        teacher = mx.stop_gradient(
            model.ar_logits(batch.teacher_ids)[:, batch.supervised_slice, :]
        )
        student = mx.stop_gradient(
            model.draft_logits(batch.student_ids, batch.lora_mask)[
                :, batch.supervised_slice, :
            ]
        )
        metrics = slot_metrics(student, teacher, batch.targets)
        tv_total += float(total_variation(student, teacher).item()) * batch.batch_size
        rows += batch.batch_size
        for key, value in metrics.items():
            totals.setdefault(key, []).append(value)

    def merge(key: str):
        values = totals[key]
        first = values[0]
        if isinstance(first, list):
            return [round(sum(col) / len(col), 6) for col in zip(*values)]
        if isinstance(first, (int, float)):
            return round(sum(values) / len(values), 6)
        return first

    summary = {key: merge(key) for key in totals}
    summary["tv"] = round(tv_total / max(rows, 1), 6)
    summary["eval_rows"] = rows
    return summary


def train_uno(
    model,
    train_windows,
    val_windows,
    config: UnoTrainConfig,
    output_dir: Path | str | None = None,
    echo=print,
    resume_from: Path | str | None = None,
) -> dict:
    """Run a full training job. Returns the history and the integrity record.

    `train_windows` / `val_windows` are infinite iterators of `[B, W]` int arrays.

    `resume_from` continues an earlier run from a checkpoint directory. The result is a
    genuine continuation and not a restart, because all four pieces of state are
    restored rather than reinitialised:

      * adapter weights and AdamW's first/second moments (`load_checkpoint`);
      * the step counter, so the learning-rate schedule and the checkpoint schedule
        both pick up where they left off;
      * the training noise, which is keyed on `(seed, TRAIN_NOISE, step)` and therefore
        reproduces exactly what an uninterrupted run would have drawn at that step;
      * the data position, which the caller fast-forwards before passing the iterator
        in (`WindowSource.fast_forward`).

    Anything less is a restart wearing a continuation's clothes, and Act IV-U already
    paid for that once.
    """
    mx.random.seed(config.seed)
    if model.lora_report is None:
        model.attach_adapter(
            rank=config.lora_rank,
            alpha=config.resolved_alpha(),
            dropout=config.lora_dropout,
            full_attention=config.lora_full_attention,
            mlp=config.lora_mlp,
            deltanet=config.lora_deltanet,
            deltanet_gates=config.lora_deltanet_gates,
        )

    params = model.parameter_report()
    echo(
        f"[uno] adapter r={config.lora_rank} alpha={config.resolved_alpha():g} "
        f"-> {model.lora_report.num_injected} modules, "
        f"{params['trainable_params']:,} trainable "
        f"({params['percent_trainable']:.4f}% of backbone)"
    )

    fingerprint_before = model.fingerprint(full=True)
    echo(f"[uno] backbone fingerprint {fingerprint_before.digest[:16]}… "
         f"({fingerprint_before.num_params:,} params)")

    optimizer = optim.AdamW(
        learning_rate=config.learning_rate, weight_decay=config.weight_decay
    )

    start_step = 0
    resumed: dict | None = None
    if resume_from is not None:
        resumed = load_checkpoint(model, optimizer, resume_from)
        start_step = int(resumed["step"])
        echo(
            f"[uno] resumed from {resume_from} at step {start_step} "
            f"({resumed.get('optimizer_tensors', 0)} optimizer tensors restored)"
        )
        saved_hash = resumed.get("config_hash")
        if saved_hash and saved_hash != config_hash(config):
            echo(
                f"[uno] WARNING: config hash changed since the checkpoint "
                f"({saved_hash[:12]}… -> {config_hash(config)[:12]}…). "
                f"This is a continuation only in the sense that the weights carry over."
            )
        if start_step >= config.steps:
            raise ValueError(
                f"checkpoint is at step {start_step} but config.steps is {config.steps}; "
                f"nothing to do"
            )

    saturation_streak = 0

    def loss_fn(batch, teacher_logits):
        logits = model.draft_logits(batch.student_ids, batch.lora_mask)
        student = logits[:, batch.supervised_slice, :]
        rows = student.shape[0] * student.shape[1]
        tv = (
            total_variation(student, teacher_logits, num_tokens=rows)
            if config.tv_weight > 0
            else mx.zeros(())
        )
        kl = (
            reverse_kl(student, teacher_logits, num_tokens=rows)
            if config.kl_weight > 0
            else mx.zeros(())
        )
        if config.ce_weight > 0:
            targets = (
                mx.argmax(teacher_logits, axis=-1)
                if config.ce_target == "teacher_argmax"
                else batch.targets
            )
            ce = cross_entropy(student, targets, num_tokens=rows)
        else:
            ce = mx.zeros(())
        absmax = mx.max(mx.abs(mx.stop_gradient(student)))
        return combine(
            ce, kl, tv, config.ce_weight, config.kl_weight, config.tv_weight
        ), (ce, kl, tv, absmax)

    history: list[dict] = []
    evals: list[dict] = []
    start = time.perf_counter()

    for step in range(start_step, config.steps):
        block_size = config.block_size_at(step, start_step)
        windows = next(train_windows)
        batch = build_uno_batch(
            windows,
            block_size=block_size,
            mask_token_id=model.mask_token_id,
            vocab_size=model.vocab_size,
            noise_mode=config.noise_mode,
            corruption=config.corruption,
            shuffle_targets=config.shuffle_targets,
            # Keyed on the step index, not on how many draws came before: this is what
            # makes a resumed run see the noise the uninterrupted run would have seen.
            key=stream_key(config.seed, TRAIN_NOISE, step),
            noise_align_width=config.noise_align_width,
        )
        teacher_logits = mx.stop_gradient(
            model.ar_logits(batch.teacher_ids)[:, batch.supervised_slice, :]
        )
        teacher_absmax = float(mx.max(mx.abs(teacher_logits)).item())

        optimizer.learning_rate = _learning_rate(config, step)

        # `nn.value_and_grad` differentiates w.r.t. `model.base.trainable_parameters()`,
        # which after `configure_trainable` is exactly the adapter.
        def forward():
            return loss_fn(batch, teacher_logits)

        (loss, (ce, kl, tv, absmax)), grads = nn.value_and_grad(model.base, forward)()

        if config.grad_clip and config.grad_clip > 0:
            grads, grad_norm = optim.clip_grad_norm(grads, config.grad_clip)
        else:
            grad_norm = mx.zeros(())
        optimizer.update(model.base, grads)
        mx.eval(model.base.parameters(), optimizer.state, loss)

        loss_value = float(loss.item())
        if not (loss_value == loss_value and abs(loss_value) != float("inf")):
            raise RuntimeError(f"non-finite loss at step {step}: {loss_value}")

        # Kill criterion 6, made unmissable. A saturated softmax gives TV a gradient of
        # p_s·(sign − Σ sign·p_s) ≈ 0, so an over-large learning rate does not diverge
        # loudly — it freezes at a constant loss with |g| = 0 and reads as "the adapter
        # learned nothing". That is a plumbing failure wearing a null result's clothes,
        # so it raises rather than being logged and forgotten.
        grad_value = float(grad_norm.item())
        logit_absmax = float(absmax.item())
        if grad_value < 1e-3 and logit_absmax > 2.0 * teacher_absmax:
            saturation_streak += 1
        else:
            saturation_streak = 0
        if saturation_streak >= 5:
            raise RuntimeError(
                f"training collapsed at step {step}: gradient norm {grad_value:.2e} for "
                f"{saturation_streak} consecutive steps while student logits reached "
                f"absmax {logit_absmax:.1f} against a teacher absmax of "
                f"{teacher_absmax:.1f}. The softmax has saturated and TV's gradient has "
                f"vanished. Lower learning_rate (currently {config.learning_rate:g}; "
                f"IFM use 1e-5) or lower lora_alpha."
            )

        record = {
            "step": step,
            "block_size": block_size,
            "loss": round(loss_value, 6),
            "tv": round(float(tv.item()), 6),
            "ce": round(float(ce.item()), 6),
            "kl": round(float(kl.item()), 6),
            "grad_norm": round(grad_value, 6),
            "student_logit_absmax": round(logit_absmax, 3),
            "lr": optimizer.learning_rate.item()
            if hasattr(optimizer.learning_rate, "item")
            else optimizer.learning_rate,
        }
        history.append(record)
        if config.log_every and step % config.log_every == 0:
            echo(
                f"[uno] step {step:4d} B={block_size} loss {loss_value:.4f} "
                f"tv {record['tv']:.4f} |g| {record['grad_norm']:.3f} "
                f"logit_absmax {logit_absmax:.1f}"
            )

        if (step + 1) in set(config.checkpoint_steps):
            target = Path(output_dir) / f"step-{step + 1}" if output_dir else None
            if target is not None:
                save_checkpoint(
                    model, optimizer, step + 1, target,
                    extra={
                        "train_tv": record["tv"],
                        "lr": record["lr"],
                        "block_size": block_size,
                        "config_hash": config_hash(config),
                        "seed": config.seed,
                        "rng": stream_report(config.seed),
                        "data_position": {
                            "train": _source_state(train_windows),
                            "val": _source_state(val_windows),
                        },
                    },
                )
                echo(f"[uno] checkpoint -> {target}")

        if config.eval_every and (step + 1) % config.eval_every == 0:
            metrics = evaluate(model, val_windows, config, block_size=block_size)
            metrics["step"] = step + 1
            evals.append(metrics)
            echo(
                f"[uno] eval @{step + 1} B={block_size} "
                f"tv {metrics['tv']:.4f} "
                f"agree(specs) {metrics['agree_specs']:.3f} "
                f"prefix {metrics['mean_accepted_prefix']:.2f}/{metrics['offered_specs']} "
                f"per-slot {metrics['agree_per_slot']}"
            )

    wall = time.perf_counter() - start
    fingerprint_after = model.fingerprint(full=True)
    changed = fingerprint_before.diff(fingerprint_after)
    if changed:
        raise RuntimeError(
            "EXPERIMENT INVALID: the frozen backbone changed during training. "
            f"{len(changed)} tensors moved, first: {changed[:5]}"
        )
    echo(
        f"[uno] backbone unchanged after {config.steps - start_step} steps "
        f"({start_step} -> {config.steps}, {wall:.1f}s)"
    )

    result = {
        "config": config.to_dict(),
        "config_hash": config_hash(config),
        "rng": stream_report(config.seed),
        "resume": {
            "resumed_from": str(resume_from) if resume_from else None,
            "start_step": start_step,
            "end_step": config.steps,
            "checkpoint_state": resumed,
        },
        "data_position": {
            "train": _source_state(train_windows),
            "val": _source_state(val_windows),
        },
        "params": params,
        "lora": model.lora_report.to_dict(),
        "integrity": {
            "digest_before": fingerprint_before.digest,
            "digest_after": fingerprint_after.digest,
            "backbone_unchanged": True,
            "num_backbone_tensors": fingerprint_before.num_tensors,
            "num_backbone_params": fingerprint_before.num_params,
        },
        "history": history,
        "evals": evals,
        "wall_seconds": round(wall, 2),
    }

    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        save_adapter(model, output / "adapter.safetensors")
        echo(f"[uno] wrote {output}/result.json and adapter.safetensors")
    return result


def config_hash(config: UnoTrainConfig) -> str:
    """A stable digest of everything that defines the training arm.

    `steps`, `checkpoint_steps`, `eval_every`, `eval_batches` and `log_every` are
    excluded on purpose: extending a run or changing how often it is measured must not
    read as "a different experiment". Everything that changes what is optimised is in.
    """
    import hashlib

    ignore = {"steps", "checkpoint_steps", "eval_every", "eval_batches", "log_every"}
    payload = {k: v for k, v in sorted(config.to_dict().items()) if k not in ignore}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _source_state(source) -> dict | None:
    """`WindowSource.state_dict()` if the iterator has one, else `None`.

    The overfit harness passes a plain generator; a checkpoint from it simply records
    no data position rather than failing.
    """
    getter = getattr(source, "state_dict", None)
    return getter() if callable(getter) else None


def save_checkpoint(model, optimizer, step: int, directory: Path | str,
                    extra: dict | None = None) -> Path:
    """Write everything needed to resume *exactly*: adapter, optimizer, RNG, position.

    Act IV-U persisted only `lora_a`/`lora_b`. That is enough to evaluate a
    checkpoint but not to continue training from it: AdamW's first and second moments
    and the data-iterator position are both lost, so a "continuation" would silently
    be a fresh optimizer on a different data order. Saving the optimizer state makes
    the difference between a resumable run and a restart-in-disguise.

    Act IV-U4 §2 requires seven things in a checkpoint. Where each lives:

    | adapter weights | `adapter.safetensors`                                        |
    | optimizer state | `optimizer.safetensors`                                      |
    | scheduler state | `step` + `lr`; the schedule is a pure function of the step    |
    | training step   | `step`                                                       |
    | PRNG state      | `rng` — stream *bases*; the draw is keyed on `(seed, step)`,  |
    |                 | so there is no evolving state to lose                        |
    | data position   | `data_position`                                              |
    | config hash     | `config_hash`                                                |

    The PRNG row is the one worth reading twice. Nothing here saves a mutable RNG
    state, because after `rng.py` there is none to save: the noise at step `s` is a
    function of `(seed, stream, s)` alone.
    """
    from mlx.utils import tree_flatten

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_adapter(model, directory / "adapter.safetensors")

    optimizer_state = {
        name: value
        for name, value in tree_flatten(optimizer.state)
        if isinstance(value, mx.array)
    }
    if optimizer_state:
        mx.save_safetensors(str(directory / "optimizer.safetensors"), optimizer_state)

    payload = {"step": step, "optimizer_tensors": len(optimizer_state)}
    if extra:
        payload.update(extra)
    (directory / "checkpoint_state.json").write_text(json.dumps(payload, indent=2) + "\n")
    return directory


def load_checkpoint(model, optimizer, directory: Path | str) -> dict:
    """Restore adapter and optimizer state written by `save_checkpoint`."""
    from mlx.utils import tree_unflatten

    directory = Path(directory)
    load_adapter(model, directory / "adapter.safetensors")
    optimizer_path = directory / "optimizer.safetensors"
    if optimizer_path.is_file():
        optimizer.state = tree_unflatten(list(mx.load(str(optimizer_path)).items()))
        mx.eval(optimizer.state)
    return json.loads((directory / "checkpoint_state.json").read_text())


def save_adapter(model, path: Path | str) -> int:
    """Persist only the adapter tensors. The backbone is never written."""
    from mlx.utils import tree_flatten

    tensors = {
        name: value
        for name, value in tree_flatten(model.base.trainable_parameters())
        if name.endswith(("lora_a", "lora_b"))
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), tensors)
    return len(tensors)


def load_adapter(model, path: Path | str) -> int:
    """Load adapter tensors back onto a model whose adapter is already attached."""
    from mlx.utils import tree_unflatten

    tensors = mx.load(str(path))
    model.base.update(tree_unflatten(list(tensors.items())))
    mx.eval(model.base.parameters())
    return len(tensors)
