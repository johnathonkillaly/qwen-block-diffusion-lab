"""Load Qwen3.5 into MLX through the Unsloth-installed stack.

Loading path, and why it is what it is:

`unsloth.FastLanguageModel` on darwin-arm64 dispatches into `unsloth_zoo.mlx`, whose
loader is built for the standard SFT flow (it wants a tokenizer/chat setup and
returns a model wired for `unsloth_zoo.mlx.trainer`). Our forward pass is not an SFT
forward pass, so we take the layer *underneath* it: `mlx_lm.load`, which is the same
loader Unsloth's MLX path uses to materialise the model, and then apply Unsloth's
Qwen3.5-specific Gated DeltaNet optimisation ourselves via
`unsloth_zoo.gated_delta_vjp.patch_gated_delta()`.

That keeps us on the Unsloth engine for the part that matters (the DeltaNet
recurrence and its memory-efficient custom VJP) without pretending a stock SFT
wrapper can express a diffusion objective. docs/UNSLOTH_BACKEND.md records exactly
which Unsloth components activate and which cannot.

Text-only is free here: `Qwen3_5 Model.sanitize` drops `model.visual*` keys, so the
vision tower is never materialised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoadReport:
    path: str
    model_type: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    full_attention_interval: int
    full_attention_layers: list[int]
    linear_attention_layers: list[int]
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    head_repeat_factor: int
    tie_word_embeddings: bool
    total_params: int
    param_bytes: int
    load_seconds: float
    dtype: str
    vision_loaded: bool
    unsloth_gdn_patch: str = "not attempted"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def patch_unsloth_gated_delta() -> str:
    """Apply Unsloth's memory-efficient GatedDeltaNet custom VJP, if it applies.

    `unsloth_zoo.gated_delta_vjp.patch_gated_delta()` monkey-patches mlx_lm's
    `gated_delta` module so the T-step recurrence recomputes states during backward
    instead of holding all T intermediates in the autograd graph. It is written for
    Qwen3.5 specifically. Returns a human-readable status string -- never claims
    success it did not verify.
    """
    try:
        from unsloth_zoo.gated_delta_vjp import patch_gated_delta
    except ImportError as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"
    try:
        patch_gated_delta()
    except Exception as exc:  # noqa: BLE001 - report, never silently continue
        return f"failed ({type(exc).__name__}: {exc})"

    import mlx_lm.models.gated_delta as gd

    fn = getattr(gd, "gated_delta_ops", None)
    where = getattr(fn, "__module__", "?")
    return f"active (gated_delta_ops now from {where})"


def load_qwen35_mlx(
    path: str | Path, use_unsloth_gdn: bool = True, dtype: str = "bfloat16"
):
    """Load the text decoder. Returns (model, tokenizer, LoadReport)."""
    import time

    import mlx.core as mx
    from mlx.utils import tree_flatten

    patch_status = patch_unsloth_gated_delta() if use_unsloth_gdn else "disabled by config"

    from mlx_lm import load

    t0 = time.time()
    model, tokenizer = load(str(path))
    mx.eval(model.parameters())
    load_seconds = time.time() - t0

    args = model.language_model.args
    layers = model.layers
    linear_idx = [i for i, l in enumerate(layers) if l.is_linear]
    full_idx = [i for i, l in enumerate(layers) if not l.is_linear]

    params = tree_flatten(model.parameters())
    total = sum(v.size for _, v in params)
    nbytes = sum(v.nbytes for _, v in params)

    report = LoadReport(
        path=str(path),
        model_type=args.model_type,
        num_hidden_layers=args.num_hidden_layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        vocab_size=args.vocab_size,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=getattr(args, "head_dim", args.hidden_size // args.num_attention_heads),
        full_attention_interval=args.full_attention_interval,
        full_attention_layers=full_idx,
        linear_attention_layers=linear_idx,
        linear_num_key_heads=args.linear_num_key_heads,
        linear_num_value_heads=args.linear_num_value_heads,
        linear_key_head_dim=args.linear_key_head_dim,
        linear_value_head_dim=args.linear_value_head_dim,
        linear_conv_kernel_dim=args.linear_conv_kernel_dim,
        head_repeat_factor=args.linear_num_value_heads // args.linear_num_key_heads,
        tie_word_embeddings=args.tie_word_embeddings,
        total_params=total,
        param_bytes=nbytes,
        load_seconds=load_seconds,
        dtype=str(params[0][1].dtype) if params else dtype,
        vision_loaded=any("visual" in k for k, _ in params),
        unsloth_gdn_patch=patch_status,
    )
    report.notes.append(
        "Vision tower is dropped by Qwen3_5 Model.sanitize(); text-only needs no surgery."
    )
    if report.head_repeat_factor > 1:
        report.notes.append(
            f"linear_num_value_heads / linear_num_key_heads = {report.head_repeat_factor}: "
            f"the DeltaNet head-replication path is LIVE at this scale (it was dead code "
            f"at 0.8B, so v0.1 never exercised it)."
        )
    return model, tokenizer, report


def resolve_mlx_model_path(repo_id: str, local_path: str | None = None) -> Path:
    """Find the checkpoint locally; never download implicitly."""
    import os

    if local_path:
        p = Path(os.path.expanduser(local_path))
        if not p.exists():
            raise FileNotFoundError(f"model.local_path does not exist: {p}")
        return p

    from ..models.registry import scan_local_models

    for m in scan_local_models(name_filter=""):
        if m.repo_id == repo_id and m.fmt == "safetensors" and m.quantization is None:
            return m.path
    raise FileNotFoundError(
        f"{repo_id!r} not found locally as an unquantized safetensors checkpoint. "
        f"Set model.local_path, or download it explicitly."
    )
