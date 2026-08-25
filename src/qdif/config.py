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

    # --- v0.2 / MLX backend: target families selected by flag rather than preset,
    # so ablations can turn one family on at a time and the report says exactly
    # which modules are trainable.
    full_attention: bool = True
    deltanet: bool = False
    deltanet_gates: bool = False
    mlp: bool = False


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
    # --- Act III (FLARE-inspired AR + diffusion transfer) ---
    #: Weight on the diffusion term in L_total = L_AR + lambda_diff * L_diff.
    #: FLARE writes unweighted sums over a token-balanced partition; we take per-token
    #: means of each term, which reproduces that 1:1 balance. See docs/ACT3_OBJECTIVE.md.
    lambda_diff: float = 1.0
    #: Weight on the clean causal AR term. 0.0 recovers the Act I/II pure-diffusion
    #: objective, retained as an ablation.
    ar_weight: float = 1.0
    #: FLARE's complementary mask views: M and M^c partition the block, so every canvas
    #: token gets exactly one diffusion signal per step. Costs a second noisy forward.
    complementary_views: bool = True
    #: Mask exactly round(t*C) positions rather than sampling Bernoulli(t) per position.
    exact_mask_count: bool = True
    #: Which canvas positions contribute to the loss: "all" or "corrupted".
    #: "all" is the standard x0-prediction objective, but under *uniform* corruption
    #: it makes copying the input a strong local optimum -- see docs/EXPERIMENTS.md,
    #: experiment 001. "corrupted" scores only the positions noise actually changed,
    #: which removes the copy shortcut at the cost of no gradient at t = 0.
    loss_on: str = "all"


@dataclass
class BidirectionalDeltaNetConfig:
    """v0.2: run the pretrained Gated DeltaNet recurrence in both directions.

    The recurrence itself is causal by construction (see docs/QWEN35_NOTES.md), so
    the only way to get backward information flow through it is to evaluate the
    *same* pretrained weights on the reversed canvas and fuse the two directional
    representations. See docs/BIDIRECTIONAL_DELTANET.md.
    """

    enabled: bool = False
    #: Evaluate the same parameter tensors twice rather than instantiating a second
    #: DeltaNet. Setting this False is not implemented -- a 4B backbone must not
    #: silently become an 8B backbone.
    share_base_weights: bool = True
    #: "fusion" -> our aligned forward/reverse dual recurrence (the v0.2 hypothesis).
    #: "flare"  -> FLARE-style block-end recurrent-state readout (arXiv:2606.01774v2):
    #:            no reverse pass; every canvas position reads the completed block-end
    #:            state. A mechanism transplant, NOT a FLARE baseline -- see
    #:            docs/FLARE_COMPARISON.md.
    mode: str = "fusion"
    #: "mean" | "scalar_gate" | "token_gate" | "concat_proj", plus the controls
    #: "forward_scaled_control" | "shuffled_reverse_control". Ignored when mode='flare'.
    fusion: str = "mean"
    #: Initial value of the forward-path gate for the learned fusions. Close to 1.0
    #: means "start from the pretrained causal computation and learn to admit the
    #: reverse direction", rather than destroying the pretrained path at step 0.
    gate_init: float = 0.95
    #: Restrict bidirectional treatment to these DeltaNet layer indices. Empty = all.
    layer_indices: list[int] = field(default_factory=list)
    #: Train a separate LoRA on the reverse pass. Not implemented in this milestone;
    #: the reverse pass shares the forward pass's adapters.
    train_reverse_lora: bool = False


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
    #: Held-out split. For wikitext this is a DIFFERENT SET OF ARTICLES, so the
    #: held-out claim is document-disjoint, not merely window-disjoint.
    eval_split: str = "test"
    eval_max_examples: int = 128
    #: Minimum characters for a paragraph to be used at all (wikitext has many
    #: near-empty lines and bare section headers).
    min_chars: int = 200
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
    bidirectional_deltanet: BidirectionalDeltaNetConfig = field(
        default_factory=BidirectionalDeltaNetConfig
    )
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
