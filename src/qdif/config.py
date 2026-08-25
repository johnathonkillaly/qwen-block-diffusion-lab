"""Typed experiment configuration, loaded from YAML.

Everything the harness does is driven by one of these. Dataclasses (rather than raw
dicts) so that typos in a config file fail loudly at load time instead of silently
selecting a default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class ModelConfig:
    #: Hugging Face repo id. Used for provenance/reporting and as a download fallback.
    id: str = "Qwen/Qwen3.5-4B-Base"
    #: Explicit local directory. If null, the model registry resolves `id` against
    #: local caches (HF hub cache, SHUTTLE, LM Studio) before considering a download.
    local_path: str | None = None
    #: Never hit the network. Research default: True, so a scaffold run cannot
    #: accidentally start an 8 GB download.
    local_files_only: bool = True
    #: Qwen3.5 checkpoints are multimodal (`Qwen3_5ForConditionalGeneration`).
    #: We load the text decoder only; see docs/QWEN35_NOTES.md.
    text_only: bool = True
    #: "eager" | "sdpa". We need arbitrary 4-D float masks; both support that.
    attn_implementation: str = "sdpa"


@dataclass
class LoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    #: Named preset from `qdif.training.lora.TARGET_PRESETS`, or "custom".
    target_preset: str = "full_attn"
    #: Used when target_preset == "custom": list of module-name suffixes.
    target_modules: list[str] = field(default_factory=list)
    #: Restrict adaptation to a subset of decoder layers by index. Empty = all.
    #: e.g. [3, 7, 11, 15, 19, 23] adapts only the full-attention layers of a 24L model.
    layer_indices: list[int] = field(default_factory=list)
    init_scale: float = 0.01


@dataclass
class DiffusionConfig:
    #: Length of the noisy block the model must denoise.
    canvas_length: int = 32
    #: "uniform" (random-vocabulary replacement) or "mask" (absorbing-state).
    corruption: str = "uniform"
    #: Timestep sampler: "uniform" | "cosine".
    schedule: str = "uniform"
    t_min: float = 0.0
    t_max: float = 1.0
    #: Additive sinusoidal timestep embedding injected at canvas positions.
    timestep_conditioning: bool = True
    self_conditioning: bool = False
    #: Probability of running the extra self-conditioning forward during training.
    self_conditioning_prob: float = 0.5
    #: Whether canvas positions may attend to each other bidirectionally in
    #: full-attention layers. Setting False gives a causal control run.
    bidirectional_canvas: bool = True
    #: Token id used by the "mask" corruption mode. None -> resolved from tokenizer.
    mask_token_id: int | None = None


@dataclass
class TrainingConfig:
    backend: str = "torch"  # "torch" | "mlx"
    mode: str = "lora"  # "lora" | "full"
    dtype: str = "bf16"  # "bf16" | "fp32" | "fp16"
    device: str = "auto"  # "auto" | "mps" | "cuda" | "cpu"
    quantization: str = "none"  # "none" | "int8" | "int4"  (see docs/MEMORY.md)
    batch_size: int = 1
    grad_accum: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    max_steps: int = 20
    log_every: int = 1
    sample_every: int = 10
    save_every: int = 0
    seed: int = 1234
    #: Emit a CLEAN/NOISY/PREDICTED triple in the log every `sample_every` steps.
    show_reconstruction: bool = True


@dataclass
class SamplerConfig:
    steps: int = 8
    adaptive: bool = False
    #: Stop early when mean top-1 probability across the canvas exceeds this.
    confidence_threshold: float = 0.95
    temperature: float = 0.0  # 0 -> argmax
    #: "confidence" commits the most-confident positions first; "all" rewrites
    #: every position each step.
    strategy: str = "confidence"
    #: Number of blocks to generate in block-autoregressive mode.
    num_blocks: int = 1


@dataclass
class DataConfig:
    #: "dev" (built-in tiny public-domain corpus) or "hf".
    source: str = "dev"
    hf_dataset: str | None = None
    hf_split: str = "train"
    text_field: str = "text"
    prefix_length: int = 16
    max_examples: int = 64


@dataclass
class RunConfig:
    name: str = "scaffold"
    output_dir: str = "runs"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run: RunConfig = field(default_factory=RunConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.run.output_dir) / self.run.name

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls, data: dict[str, Any], path: str):
    """Instantiate a dataclass from a dict, rejecting unknown keys."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) under '{path}': {sorted(unknown)}. Known: {sorted(known)}"
        )
    # `from __future__ import annotations` makes field.type a string, so resolve
    # the real annotations before recursing into nested config sections.
    hints = get_type_hints(cls)
    kwargs = {}
    for name, value in data.items():
        ftype = hints[name]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _build(ftype, value, f"{path}.{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return _build(ExperimentConfig, raw, "<root>")


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
