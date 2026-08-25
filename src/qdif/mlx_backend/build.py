"""Assemble a v0.2 experiment: load -> wrap DeltaNets -> inject LoRA -> freeze.

One function so that every entry point (smoke test, probe, trainer, sampler) builds
exactly the same object from the same config, and ablations A-E differ only by config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from .deltanet import wrap_deltanet_layers
from .loader import LoadReport, load_qwen35_mlx, resolve_mlx_model_path
from .lora import configure_trainable, inject_lora
from .model import MLXDiffusionQwen, parameter_report


@dataclass
class MLXSetup:
    model: MLXDiffusionQwen
    tokenizer: Any
    load_report: LoadReport
    bidi_report: dict
    lora_report: Any
    trainable_report: dict
    params: dict
    model_path: Path
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "model_path": str(self.model_path),
            "load": self.load_report.to_dict(),
            "bidirectional_deltanet": {
                k: v for k, v in self.bidi_report.items() if k != "shared_weight_ids"
            },
            "lora": {
                "modules": self.lora_report.num_injected,
                "families": self.lora_report.families,
                "rank": self.lora_report.rank,
                "alpha": self.lora_report.alpha,
                "params": self.lora_report.lora_params,
            }
            if self.lora_report
            else None,
            "trainable": self.trainable_report,
            "params": self.params,
            "notes": self.notes,
        }


def build_mlx_setup(cfg: ExperimentConfig, echo=print) -> MLXSetup:
    import mlx.core as mx

    if cfg.training.dtype not in ("bf16", "bfloat16"):
        raise NotImplementedError(
            f"training.dtype={cfg.training.dtype!r}: v0.2 runs BF16 base + BF16 LoRA by "
            "design. Qwen3.5 shows larger-than-usual quantization error, and this "
            "milestone prioritises correctness over footprint (see docs/MEMORY.md)."
        )
    if cfg.training.quantization != "none":
        raise NotImplementedError(
            f"training.quantization={cfg.training.quantization!r} is refused for "
            "Qwen3.5 in v0.2. BF16 base + LoRA is the specified precision."
        )

    path = resolve_mlx_model_path(cfg.model.id, cfg.model.local_path)
    echo(f"[setup] checkpoint: {path}")

    model, tokenizer, load_report = load_qwen35_mlx(path)
    echo(
        f"[setup] loaded {load_report.total_params:,} params "
        f"({load_report.param_bytes / 1e9:.2f} GB, {load_report.dtype}) "
        f"in {load_report.load_seconds:.1f}s"
    )
    echo(f"[setup] unsloth gated-delta VJP: {load_report.unsloth_gdn_patch}")

    bcfg = cfg.bidirectional_deltanet
    if not bcfg.share_base_weights:
        raise NotImplementedError(
            "bidirectional_deltanet.share_base_weights=false is not implemented on "
            "purpose: duplicating DeltaNet weights would turn a 4B backbone into a "
            "larger one and change what the experiment measures."
        )

    # Always wrap, even for ablation A. The wrapper is bit-exact with upstream when
    # `bidirectional=False`, so A and B differ by one boolean and nothing else.
    bidi_report = wrap_deltanet_layers(
        model,
        fusion=bcfg.fusion,
        gate_init=bcfg.gate_init,
        layer_indices=bcfg.layer_indices,
        enabled=bcfg.enabled,
    )
    echo(
        f"[setup] bidirectional DeltaNet: enabled={bcfg.enabled} fusion={bcfg.fusion} "
        f"layers={bidi_report['num_wrapped']} fusion_params={bidi_report['fusion_params']:,}"
    )

    diffusion = MLXDiffusionQwen(model, cfg.diffusion, load_report.hidden_size)

    lora_report = None
    if cfg.training.mode == "lora":
        lora_report = inject_lora(
            model,
            rank=cfg.lora.rank,
            alpha=cfg.lora.alpha,
            dropout=cfg.lora.dropout,
            init_scale=cfg.lora.init_scale,
            full_attention=cfg.lora.full_attention,
            deltanet=cfg.lora.deltanet,
            deltanet_gates=cfg.lora.deltanet_gates,
            mlp=cfg.lora.mlp,
            layer_indices=cfg.lora.layer_indices,
        )
        echo(
            f"[setup] LoRA r={cfg.lora.rank} a={cfg.lora.alpha} "
            f"families={lora_report.families} -> {lora_report.num_injected} modules, "
            f"{lora_report.lora_params:,} params"
        )

    trainable_report = configure_trainable(diffusion)
    mx.eval(diffusion.parameters())
    params = parameter_report(diffusion)
    echo(
        f"[setup] trainable {params['trainable_params']:,} "
        f"({params['percent_trainable']:.4f}%) by family {params['trainable_by_family']}"
    )

    return MLXSetup(
        model=diffusion,
        tokenizer=tokenizer,
        load_report=load_report,
        bidi_report=bidi_report,
        lora_report=lora_report,
        trainable_report=trainable_report,
        params=params,
        model_path=Path(path),
        notes=list(load_report.notes),
    )
