"""`UnoModel` — the frozen backbone plus a gated adapter, and nothing else.

Deliberately thin. The backbone is the unmodified mlx_lm `Model`: no wrapper layers,
no custom masks, no bidirectional recurrence. Every forward here is the pretrained
causal forward; the only thing that ever changes is whether the adapter contributes on
a given token row.

That constraint is the experiment. Acts I-III all modified how the backbone reads its
input; Act IV-U modifies nothing and asks whether a shortcut can be bolted on beside
it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

from .gated_lora import (
    GatedLoRALinear,
    configure_trainable,
    gated_lora_modules,
    inject_gated_lora,
    lora_disabled,
    lora_token_mask,
)


@dataclass
class BackboneFingerprint:
    """An integrity record for the frozen weights.

    `digest` is a blake2b over every backbone tensor's raw bytes, in a sorted, stable
    order, with shape and dtype folded in. `checksums` keeps a per-tensor float64 sum
    so that a mismatch can be localised rather than merely detected.
    """

    digest: str
    num_tensors: int
    num_params: int
    checksums: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "digest": self.digest,
            "num_tensors": self.num_tensors,
            "num_params": self.num_params,
        }

    def diff(self, other: "BackboneFingerprint") -> list[str]:
        """Names of tensors whose checksum moved. Empty means unchanged."""
        changed = []
        for name, value in self.checksums.items():
            if name not in other.checksums:
                changed.append(f"{name} (missing)")
            elif other.checksums[name] != value:
                changed.append(name)
        for name in other.checksums:
            if name not in self.checksums:
                changed.append(f"{name} (added)")
        return sorted(changed)


def _is_backbone(name: str) -> bool:
    return not name.endswith(("lora_a", "lora_b"))


def _canonical_name(name: str) -> str:
    """Parameter name with the adapter wrapper's `.base` segment removed.

    Attaching a `GatedLoRALinear` renames `…q_proj.weight` to `…q_proj.base.weight`.
    Folding the raw name into the digest would then make an adapted model hash
    differently from the same weights unadapted — which reads as "the backbone
    changed" when nothing did. Normalising means one digest is comparable across every
    run in the project, adapted or not, which is the guarantee actually worth having.
    """
    return name.replace(".base.", ".")


#: Unsigned integer type of the same width as the value's dtype, so that a bit-exact
#: reinterpret never has to split or merge elements.
_WIDTH_VIEW = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}


def _canonical_bytes(value: mx.array) -> bytes:
    """Bit-exact bytes of an MLX array, deterministically.

    Two traps had to be worked around here, both found on the real 4B checkpoint and
    both of which manifested as a *spurious* "the frozen backbone changed" failure:

    1. `bytes(memoryview(...))` on a lazily-evaluated array is not stable. Force
       evaluation first.
    2. `.view(mx.uint8)` — a reinterpret to a *narrower* dtype, which splits each
       element in two — is not stable either. On `layers.18.linear_attn.conv1d.weight`
       (shape `(8192, 4, 1)`, a tensor `sanitize` reshapes at load) consecutive uint8
       views hashed differently while a same-width uint16 view of the same tensor was
       identical element for element. Reinterpreting at the dtype's own width avoids
       the whole class of problem.

    Both are worth keeping written down: a flaky integrity check is worse than none,
    because it trains you to ignore the one time it fires for real.
    """
    import numpy as np

    evaluated = mx.contiguous(value)
    mx.eval(evaluated)
    itemsize = evaluated.nbytes // max(evaluated.size, 1)
    view_dtype = _WIDTH_VIEW.get(itemsize)
    if view_dtype is None:  # exotic dtype: fall back to an exact float32 promotion
        return np.array(evaluated.astype(mx.float32)).tobytes()
    return np.ascontiguousarray(np.array(evaluated.view(view_dtype))).tobytes()


def fingerprint_backbone(model, full: bool = True) -> BackboneFingerprint:
    """Hash every non-adapter parameter.

    `full=False` skips the byte hash and keeps only per-tensor sums -- enough to catch
    an optimizer writing into a backbone tensor, and much faster. `full=True` is what
    Gate 0 runs, and is the claim we actually make in results.
    """
    hasher = hashlib.blake2b(digest_size=32)
    checksums: dict[str, float] = {}
    num_tensors = 0
    num_params = 0

    entries = sorted(
        (_canonical_name(name), value)
        for name, value in tree_flatten(model.parameters())
        if _is_backbone(name)
    )
    for name, value in entries:
        num_tensors += 1
        num_params += value.size
        checksums[name] = float(mx.sum(value.astype(mx.float32)).item())
        if full:
            hasher.update(name.encode())
            hasher.update(str(value.shape).encode())
            hasher.update(str(value.dtype).encode())
            hasher.update(_canonical_bytes(value))
        else:
            hasher.update(f"{name}{value.shape}{value.dtype}{checksums[name]!r}".encode())

    return BackboneFingerprint(
        digest=hasher.hexdigest(),
        num_tensors=num_tensors,
        num_params=num_params,
        checksums=checksums,
    )


class UnoModel:
    """A frozen mlx_lm model plus gated adapters.

    Not an `nn.Module`: it owns one and routes calls, so that `self.base.parameters()`
    stays exactly the parameter tree mlx_lm produced and the fingerprint means what it
    says.
    """

    def __init__(self, base_model, tokenizer: Any = None, mask_token_id: int | None = None):
        self.base = base_model
        self.tokenizer = tokenizer
        self.lora_report = None
        self.trainable_report = None
        args = base_model.language_model.args
        self.vocab_size = int(args.vocab_size)
        self.num_layers = int(args.num_hidden_layers)
        self.hidden_size = int(args.hidden_size)
        # Uno needs an id that bounds the uniform-noise range. Qwen3.5-4B-Base has no
        # dedicated diffusion mask token, so we take the first id of the reserved
        # special-token band unless told otherwise; noise ids are strictly below it.
        self.mask_token_id = int(
            mask_token_id if mask_token_id is not None else self._default_mask_token_id()
        )

    def _default_mask_token_id(self) -> int:
        tok = self.tokenizer
        for attribute in ("mask_token_id", "pad_token_id", "eos_token_id"):
            value = getattr(tok, attribute, None)
            if isinstance(value, int) and value > 1:
                return value
        return self.vocab_size

    # ------------------------------------------------------------------ setup
    def attach_adapter(self, **kwargs):
        """Inject gated LoRA and freeze everything else. Returns the report."""
        self.lora_report = inject_gated_lora(self.base, **kwargs)
        self.trainable_report = configure_trainable(self.base)
        mx.eval(self.base.parameters())
        return self.lora_report

    @property
    def adapters(self) -> list[GatedLoRALinear]:
        return gated_lora_modules(self.base)

    def make_cache(self):
        return self.base.make_cache()

    def parameter_report(self) -> dict:
        params = tree_flatten(self.base.parameters())
        trainable = tree_flatten(self.base.trainable_parameters())
        total = sum(v.size for _, v in params)
        train_n = sum(v.size for _, v in trainable)
        return {
            "total_params": total,
            "trainable_params": train_n,
            "frozen_params": total - train_n,
            "percent_trainable": 100.0 * train_n / total if total else 0.0,
            "trainable_mb": sum(v.nbytes for _, v in trainable) / 1e6,
            "param_gb": sum(v.nbytes for _, v in params) / 1e9,
        }

    def layer_kinds(self) -> dict[str, list[int]]:
        linear, attention = [], []
        for index, layer in enumerate(self.base.layers):
            (linear if getattr(layer, "is_linear", False) else attention).append(index)
        return {"gated_deltanet": linear, "full_attention": attention}

    # ---------------------------------------------------------------- forwards
    def ar_logits(self, input_ids: mx.array, cache=None) -> mx.array:
        """The pretrained autoregressive forward. Adapter hard-off.

        `logits[:, i]` is the next-token distribution after `input_ids[:, i]`.
        """
        with lora_disabled():
            return self.base(input_ids, cache=cache)

    def draft_logits(
        self, input_ids: mx.array, lora_mask: mx.array | None, cache=None
    ) -> mx.array:
        """The draft forward. Adapter contributes only where `lora_mask` is non-zero."""
        with lora_token_mask(lora_mask):
            return self.base(input_ids, cache=cache)

    def fingerprint(self, full: bool = True) -> BackboneFingerprint:
        return fingerprint_backbone(self.base, full=full)


def load_uno_model(
    model_path,
    use_unsloth_gdn: bool = True,
    mask_token_id: int | None = None,
    echo=print,
):
    """Load Qwen3.5 through the same path every other Act uses, then wrap it."""
    from ..mlx_backend.loader import load_qwen35_mlx

    base, tokenizer, report = load_qwen35_mlx(model_path, use_unsloth_gdn=use_unsloth_gdn)
    echo(
        f"[uno] loaded {report.total_params:,} params "
        f"({report.param_bytes / 1e9:.2f} GB, {report.dtype}) in {report.load_seconds:.1f}s"
    )
    echo(f"[uno] unsloth gated-delta VJP: {report.unsloth_gdn_patch}")
    echo(
        f"[uno] {len(report.full_attention_layers)} full-attention layers "
        f"{report.full_attention_layers}, "
        f"{len(report.linear_attention_layers)} Gated DeltaNet layers"
    )
    return UnoModel(base, tokenizer, mask_token_id=mask_token_id), report
