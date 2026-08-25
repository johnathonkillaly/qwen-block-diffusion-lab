"""PyTorch trainer for the diffusion objective (Apple MPS / CUDA / CPU).

Small on purpose. The scientific content is in `qdif.diffusion`; this file only
assembles a model, runs steps, and records diagnostics honestly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ..config import ExperimentConfig
from ..data.collator import DiffusionBatch, DiffusionCollator, batches
from ..data.datasets import build_dataset
from ..diffusion.objective import diffusion_loss, diffusion_metrics
from ..models.qwen35_adapter import load_qwen35_text, resolve_device, resolve_dtype
from ..models.qwen35_diffusion import DiffusionQwen, parameter_report
from ..models.registry import resolve_model_path
from .diagnostics import (
    NoiseBucketAccumulator,
    RunLogger,
    StepRecord,
    format_reconstruction,
    grad_norm,
    memory_label,
    param_norm,
    peak_memory_bytes,
)
from .lora import freeze_base, inject_lora


def _tail_mean(history: list, key: str, fraction: float = 0.25) -> float | None:
    """Mean of a metric over the last `fraction` of steps, ignoring NaNs.

    Per-step metrics jump around because every step samples a fresh timestep; a
    verdict read off the final step alone is mostly reading that noise.
    """
    if not history:
        return None
    tail = history[-max(1, int(len(history) * fraction)) :]
    values = [
        h.metrics[key] for h in tail if key in h.metrics and h.metrics[key] == h.metrics[key]
    ]
    return sum(values) / len(values) if values else None


@dataclass
class TrainingSetup:
    model: DiffusionQwen
    tokenizer: object
    dataset: object
    collator: DiffusionCollator
    device: torch.device
    dtype: torch.dtype
    params: dict
    lora_report: object | None
    model_path: Path


def build_setup(cfg: ExperimentConfig, echo=print) -> TrainingSetup:
    """Resolve the checkpoint, load the text decoder, attach LoRA, build data."""
    if cfg.training.backend != "torch":
        raise NotImplementedError(
            f"training.backend={cfg.training.backend!r} is not implemented. "
            "Only the torch backend runs today; see docs/ARCHITECTURE.md for why "
            "and for the MLX backend seam."
        )
    if cfg.training.quantization != "none":
        raise NotImplementedError(
            f"training.quantization={cfg.training.quantization!r} is not implemented on "
            "this backend. bitsandbytes has no Apple-silicon build; see docs/MEMORY.md."
        )

    torch.manual_seed(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    dtype = resolve_dtype(cfg.training.dtype)

    model_path = resolve_model_path(
        cfg.model.id, cfg.model.local_path, local_files_only=cfg.model.local_files_only
    )
    echo(f"[setup] model path: {model_path}")

    base, tokenizer = load_qwen35_text(
        model_path,
        dtype=dtype,
        device=device,
        attn_implementation=cfg.model.attn_implementation,
    )

    lora_report = None
    if cfg.training.mode == "lora":
        lora_report = inject_lora(
            base,
            rank=cfg.lora.rank,
            alpha=cfg.lora.alpha,
            dropout=cfg.lora.dropout,
            target_preset=cfg.lora.target_preset,
            target_modules=cfg.lora.target_modules,
            layer_indices=cfg.lora.layer_indices,
            init_scale=cfg.lora.init_scale,
        )
        freeze_base(base)
        echo(
            f"[setup] LoRA r={cfg.lora.rank} a={cfg.lora.alpha} preset={cfg.lora.target_preset} "
            f"-> {lora_report.num_injected} modules, {lora_report.lora_params:,} params"
        )
    elif cfg.training.mode == "full":
        echo("[setup] mode=full: every base parameter is trainable. This is NOT the default path.")
        for p in base.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(f"unknown training.mode {cfg.training.mode!r}")

    model = DiffusionQwen(base, cfg.diffusion).to(device)
    # Auxiliary diffusion modules are always trainable, in both lora and full mode.
    for module in model.auxiliary_modules().values():
        for p in module.parameters():
            p.requires_grad_(True)

    params = parameter_report(model)
    echo(
        f"[setup] params total {params['total_params']:,} | trainable "
        f"{params['trainable_params']:,} ({params['percent_trainable']:.4f}%) | "
        f"lora {params['lora_params']:,} | aux {params['auxiliary_params']:,}"
    )

    dataset = build_dataset(cfg.data, tokenizer, cfg.diffusion.canvas_length)
    echo(f"[setup] dataset: {len(dataset)} examples of {cfg.data.prefix_length}+{cfg.diffusion.canvas_length} tokens")

    gen = torch.Generator().manual_seed(cfg.training.seed)
    collator = DiffusionCollator(
        cfg.diffusion,
        vocab_size=base.config.vocab_size,
        mask_token_id=getattr(tokenizer, "mask_token_id", None) or tokenizer.eos_token_id,
        generator=gen,
    )

    return TrainingSetup(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        collator=collator,
        device=device,
        dtype=dtype,
        params=params,
        lora_report=lora_report,
        model_path=Path(model_path),
    )


def training_step(
    setup: TrainingSetup,
    batch: DiffusionBatch,
    cfg: ExperimentConfig,
    compute_metrics: bool = True,
):
    """One forward + loss. Returns (loss, metrics, logits)."""
    model = setup.model
    batch = batch.to(setup.device)

    self_cond_logits = None
    if cfg.diffusion.self_conditioning and torch.rand(1).item() < cfg.diffusion.self_conditioning_prob:
        # Standard self-conditioning: a first pass with no prior estimate, detached,
        # then a second pass conditioned on it. Only the second pass gets gradients.
        with torch.no_grad():
            self_cond_logits = model(
                canvas_ids=batch.canvas_xt, t=batch.t, prefix_ids=batch.prefix_ids
            ).logits.detach()

    out = model(
        canvas_ids=batch.canvas_xt,
        t=batch.t,
        prefix_ids=batch.prefix_ids,
        self_cond_logits=self_cond_logits,
    )
    loss_out = diffusion_loss(
        out.logits, batch.canvas_x0, batch.corrupted_mask, loss_on=cfg.diffusion.loss_on
    )

    metrics = {}
    if compute_metrics:
        metrics = diffusion_metrics(
            out.logits.detach(), batch.canvas_x0, batch.canvas_xt, batch.corrupted_mask
        )
    return loss_out, metrics, out.logits


def train(cfg: ExperimentConfig, echo=print, fixed_batch: bool = False) -> dict:
    """Run `cfg.training.max_steps` optimisation steps.

    Args:
        fixed_batch: reuse one batch for every step -- the overfit-one-batch test.
            The timestep and corruption pattern are still redrawn each step, so the
            model must learn to denoise the batch, not memorise one noise realisation.
    """
    setup = build_setup(cfg, echo=echo)
    model, tokenizer = setup.model, setup.tokenizer
    logger = RunLogger(cfg.run_dir, echo=echo)
    buckets = NoiseBucketAccumulator()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters; check training.mode and LoRA targets")
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / max(cfg.training.warmup_steps, 1))
        if cfg.training.warmup_steps
        else 1.0,
    )

    gen = torch.Generator().manual_seed(cfg.training.seed + 1)
    stream = batches(setup.dataset, cfg.training.batch_size, generator=gen)
    # The overfit batch is the *first* `batch_size` dataset rows, not a shuffled
    # draw, so the run is reproducible and `qdif reconstruct` evaluates exactly the
    # examples that were memorised.
    frozen_examples = (
        [setup.dataset[i] for i in range(cfg.training.batch_size)] if fixed_batch else None
    )

    logger.log_json(
        "setup.json",
        {
            "config": cfg.to_dict(),
            "model_path": str(setup.model_path),
            "device": str(setup.device),
            "dtype": str(setup.dtype),
            "params": setup.params,
            "lora": vars(setup.lora_report) if setup.lora_report else None,
            "dataset_size": len(setup.dataset),
            "fixed_batch": fixed_batch,
        },
    )

    history: list[StepRecord] = []
    model.train()
    for step in range(cfg.training.max_steps):
        t_start = time.time()
        examples = frozen_examples if fixed_batch else next(stream)
        # Move here, not inside training_step: the diagnostics below read the same
        # batch object and must not compare a CPU tensor against device logits.
        batch = setup.collator(examples).to(setup.device)

        loss_out, metrics, logits = training_step(setup, batch, cfg)
        loss = loss_out.loss / cfg.training.grad_accum
        loss.backward()

        gnorm = grad_norm(trainable)
        if cfg.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.training.max_grad_norm)

        if (step + 1) % cfg.training.grad_accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if setup.device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.time() - t_start
        n_tokens = batch.batch_size * cfg.diffusion.canvas_length

        record = StepRecord(
            step=step,
            loss=float(loss_out.loss.detach()),
            learning_rate=float(scheduler.get_last_lr()[0]),
            t_mean=float(batch.t.mean()),
            t_values=[round(float(v), 4) for v in batch.t],
            corruption_fraction=float(batch.corrupted_mask.float().mean()),
            changed_fraction=float((batch.canvas_xt != batch.canvas_x0).float().mean()),
            grad_norm=gnorm,
            param_norm=param_norm(trainable),
            tokens_per_sec=n_tokens / elapsed,
            examples_per_sec=batch.batch_size / elapsed,
            peak_memory_gb=peak_memory_bytes(setup.device) / 1e9,
            memory_kind=memory_label(setup.device),
            metrics=metrics,
        )
        history.append(record)
        buckets.update(record.t_values, metrics.get("identity_accuracy", float("nan")))
        if step % cfg.training.log_every == 0 or step == cfg.training.max_steps - 1:
            logger.log_step(record)

        if cfg.training.show_reconstruction and (
            step % cfg.training.sample_every == 0 or step == cfg.training.max_steps - 1
        ):
            logger.log_reconstruction(
                format_reconstruction(
                    tokenizer,
                    x0=batch.canvas_x0[0],
                    xt=batch.canvas_xt[0],
                    pred=logits[0].argmax(-1),
                    prefix=batch.prefix_ids[0],
                    step=step,
                    t=float(batch.t[0]),
                )
            )

        if cfg.training.save_every and (step + 1) % cfg.training.save_every == 0:
            from .checkpoint import save_checkpoint

            save_checkpoint(model, cfg, cfg.run_dir / "checkpoint", step=step)

    from .checkpoint import save_checkpoint

    ckpt = save_checkpoint(model, cfg, cfg.run_dir / "checkpoint", step=cfg.training.max_steps)

    summary = {
        "steps": len(history),
        "first_loss": history[0].loss if history else None,
        "final_loss": history[-1].loss if history else None,
        "best_loss": min((h.loss for h in history), default=None),
        "first_identity_accuracy": history[0].metrics.get("identity_accuracy") if history else None,
        "final_identity_accuracy": history[-1].metrics.get("identity_accuracy") if history else None,
        "final_corrupted_accuracy": history[-1].metrics.get("corrupted_accuracy") if history else None,
        "final_copy_rate": history[-1].metrics.get("copy_rate") if history else None,
        "final_copy_baseline_accuracy": (
            history[-1].metrics.get("copy_baseline_accuracy") if history else None
        ),
        "final_lift_over_copy": history[-1].metrics.get("lift_over_copy") if history else None,
        "final_next_token_accuracy": history[-1].metrics.get("next_token_accuracy") if history else None,
        # Single-step metrics are noisy because each step draws a new timestep, so
        # the verdict uses a tail average rather than the last step alone.
        "mean_corrupted_accuracy_last_quarter": _tail_mean(history, "corrupted_accuracy"),
        "mean_lift_last_quarter": _tail_mean(history, "lift_over_copy"),
        "accuracy_by_noise_bucket": buckets.summary(),
        "peak_memory_gb": max((h.peak_memory_gb for h in history), default=0.0),
        "memory_kind": memory_label(setup.device),
        "mean_tokens_per_sec": (
            sum(h.tokens_per_sec for h in history) / len(history) if history else 0.0
        ),
        "checkpoint": str(ckpt),
        "run_dir": str(cfg.run_dir),
        "params": setup.params,
    }
    logger.log_json("summary.json", summary)
    return summary
