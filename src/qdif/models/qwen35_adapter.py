"""Loading and introspecting a Qwen3.5 text decoder.

Qwen3.5 checkpoints ship as `Qwen3_5ForConditionalGeneration` (text decoder +
vision tower + MTP head). This project is text-only, so we load
`Qwen3_5ForCausalLM`, which transformers already wires to drop `model.visual.*`
and `mtp.*` (`_keys_to_ignore_on_load_unexpected`) and to rewrite
`model.language_model.*` -> `model.*`. No surgery required; the vision tower is
simply never instantiated.

`inspect_architecture` produces the report that drives docs/QWEN35_NOTES.md and
docs/QWEN38_MIGRATION.md, and it reads the *live* module tree rather than
hard-coded assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    if name not in DTYPES:
        raise ValueError(f"unknown dtype {name!r}, expected one of {sorted(DTYPES)}")
    return DTYPES[name]


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class ArchitectureReport:
    model_class: str
    config_class: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    layer_types: list[str]
    full_attention_layers: list[int]
    linear_attention_layers: list[int]
    full_attention_interval: int | None
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attn_output_gate: bool
    linear_num_key_heads: int | None
    linear_num_value_heads: int | None
    linear_key_head_dim: int | None
    linear_value_head_dim: int | None
    linear_conv_kernel_dim: int | None
    tie_word_embeddings: bool
    lm_head_shares_embedding: bool
    rope: dict[str, Any]
    max_position_embeddings: int
    mtp_num_hidden_layers: int
    checkpoint_has_mtp: bool
    checkpoint_has_vision: bool
    vision_loaded: bool
    total_params: int
    module_shapes: dict[str, list[int]] = field(default_factory=dict)
    linear_module_suffixes: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def load_qwen35_text(
    path: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    attn_implementation: str = "sdpa",
):
    """Load the text decoder only. Returns (model, tokenizer)."""
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    path = str(path)
    model = Qwen3_5ForCausalLM.from_pretrained(
        path, dtype=dtype, attn_implementation=attn_implementation
    )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


def checkpoint_tensor_names(path: str | Path) -> list[str]:
    """Tensor names in the on-disk checkpoint (before any key remapping)."""
    import json

    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        return list(json.loads(index.read_text())["weight_map"].keys())
    from safetensors import safe_open

    names: list[str] = []
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            names.extend(f.keys())
    return names


def inspect_architecture(model, checkpoint_path: str | Path | None = None) -> ArchitectureReport:
    """Read the live module tree and produce a structured architecture report."""
    cfg = model.config
    layer_types = list(cfg.layer_types)
    full_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    lin_idx = [i for i, t in enumerate(layer_types) if t == "linear_attention"]

    ckpt_names = checkpoint_tensor_names(checkpoint_path) if checkpoint_path else []

    # Every distinct nn.Linear "suffix" in the decoder, with a count. This is the
    # menu LoRA target presets are chosen from.
    suffixes: dict[str, int] = {}
    shapes: dict[str, list[int]] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parts = name.split(".")
            generic = ".".join(p for p in parts if not p.isdigit())
            suffixes[generic] = suffixes.get(generic, 0) + 1
            shapes.setdefault(generic, [module.in_features, module.out_features])

    lm_head_shared = False
    try:
        lm_head_shared = (
            model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        )
    except AttributeError:
        pass

    notes = [
        "Vision tower is not instantiated by Qwen3_5ForCausalLM; text-only execution "
        "needs no surgery.",
        f"{len(lin_idx)}/{len(layer_types)} layers are Gated DeltaNet (linear_attention) "
        f"and are causal by construction (causal conv1d + tril/triu inside "
        f"chunk_gated_delta_rule). Their attention_mask argument is a padding mask only.",
        f"{len(full_idx)}/{len(layer_types)} layers are full attention and accept an "
        f"arbitrary 4-D additive mask, so they are the only place a bidirectional "
        f"canvas can be realised.",
        "Qwen3_5TextModel.forward accepts attention_mask as a dict keyed by layer type "
        "({'full_attention': ..., 'linear_attention': ...}), which is the injection "
        "point this project uses. No monkeypatching of transformers is required.",
    ]
    if cfg.tie_word_embeddings:
        notes.append(
            "Embeddings are tied to the output head; adapting the embedding also "
            "changes the readout, so LoRA presets leave it alone by default."
        )

    return ArchitectureReport(
        model_class=type(model).__name__,
        config_class=type(cfg).__name__,
        num_hidden_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        layer_types=layer_types,
        full_attention_layers=full_idx,
        linear_attention_layers=lin_idx,
        full_attention_interval=getattr(cfg, "full_attention_interval", None),
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
        attn_output_gate=bool(getattr(cfg, "attn_output_gate", False)),
        linear_num_key_heads=getattr(cfg, "linear_num_key_heads", None),
        linear_num_value_heads=getattr(cfg, "linear_num_value_heads", None),
        linear_key_head_dim=getattr(cfg, "linear_key_head_dim", None),
        linear_value_head_dim=getattr(cfg, "linear_value_head_dim", None),
        linear_conv_kernel_dim=getattr(cfg, "linear_conv_kernel_dim", None),
        tie_word_embeddings=bool(cfg.tie_word_embeddings),
        lm_head_shares_embedding=lm_head_shared,
        rope=dict(getattr(cfg, "rope_parameters", {}) or {}),
        max_position_embeddings=cfg.max_position_embeddings,
        mtp_num_hidden_layers=int(getattr(cfg, "mtp_num_hidden_layers", 0) or 0),
        checkpoint_has_mtp=any(n.startswith("mtp.") for n in ckpt_names),
        checkpoint_has_vision=any(".visual." in n for n in ckpt_names),
        vision_loaded=hasattr(model, "visual") or hasattr(getattr(model, "model", None), "visual"),
        total_params=sum(p.numel() for p in model.parameters()),
        module_shapes=shapes,
        linear_module_suffixes=suffixes,
        notes=notes,
    )
