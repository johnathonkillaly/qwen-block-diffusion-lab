"""Token-conditional ("gated") LoRA — the defining component of Uno.

From the released model card for `s-sahoo/uno-qwen3-8B`:

    "Uno applies the adapter selectively during draft-noise forwards. Seed, prefill,
    verification, and autoregressive rows use the frozen base weights. Loading the
    adapter as an ordinary always-on PEFT adapter does not reproduce Uno decoding."

IFM implement this with a forward hook on every PEFT `lora_A` module that multiplies
its output by a 0/1 token mask (`nano_vllm_uno/training/lora.py:TokenwiseLoraRouter`).
MLX has no hook API and its modules are plain callables, so we push the mask through a
module-level routing context instead. Same semantics, fewer moving parts.

Why this matters, and why it is not merely an optimisation: the backbone must remain
*exactly* the pretrained model on every row that is not diffusion noise. If the
adapter were always on, the verifier would no longer be the frozen AR model and the
central claim of the experiment -- that adapter+verifier reproduces AR greedy output
-- would be unfalsifiable, because there would be no untouched AR model left to
reproduce.

`lora_b` is zero-initialised, so an untrained adapter is a bit-exact no-op.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

#: Standard-attention projections. Present on the 8 full-attention layers only.
FULL_ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
#: Gated DeltaNet projections. `in_proj_qkv`/`out_proj` are the value-carrying pair;
#: the `z`/`b`/`a` projections drive the gates. Uno has no analogue for these -- the
#: choice of which to adapt is ours, and is an experimental variable (Variants A/B/C).
DELTANET_TARGETS = ("in_proj_qkv", "out_proj")
DELTANET_GATE_TARGETS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
#: Uno adapts these on every layer.
MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")


class _Route:
    """Process-wide routing state for the gated adapters.

    Single-threaded by design. MLX training loops are single-threaded and the
    alternative -- threading a mask argument through every `nn.Linear` call inside
    mlx_lm's decoder -- would require forking the upstream model.
    """

    __slots__ = ("enabled", "mask")

    def __init__(self) -> None:
        self.enabled: bool = True
        self.mask: mx.array | None = None  # [B, T], 1.0 where the adapter applies


_ROUTE = _Route()


@contextmanager
def lora_disabled():
    """Run the frozen backbone: every adapter is a hard no-op inside this block."""
    previous_enabled, previous_mask = _ROUTE.enabled, _ROUTE.mask
    _ROUTE.enabled, _ROUTE.mask = False, None
    try:
        yield
    finally:
        _ROUTE.enabled, _ROUTE.mask = previous_enabled, previous_mask


@contextmanager
def lora_token_mask(mask: mx.array | None):
    """Apply the adapter only on rows where `mask` is non-zero.

    `mask` is `[B, T]`. `None` means "every row", which is the always-on PEFT
    behaviour and is **not** Uno decoding -- it exists for the ablation that measures
    what gating buys.
    """
    previous_enabled, previous_mask = _ROUTE.enabled, _ROUTE.mask
    _ROUTE.enabled, _ROUTE.mask = True, mask
    try:
        yield
    finally:
        _ROUTE.enabled, _ROUTE.mask = previous_enabled, previous_mask


def routing_state() -> tuple[bool, mx.array | None]:
    """Current (enabled, mask). Exposed so tests can assert on it."""
    return _ROUTE.enabled, _ROUTE.mask


class GatedLoRALinear(nn.Module):
    """y = Wx + m ⊙ (alpha/r)·B(A(x)), with `m` a per-token 0/1 gate.

    The base `nn.Linear` is held as a child module and never unfrozen.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        init_scale: float = 0.01,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        out_features, in_features = base.weight.shape
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = dropout
        # fp32 adapters: low-rank updates accumulate badly in bf16 under Adam, and
        # they are small enough that the memory cost is irrelevant.
        self.lora_a = mx.random.normal((in_features, rank), scale=init_scale).astype(
            mx.float32
        )
        self.lora_b = mx.zeros((rank, out_features), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.base(x)
        if not _ROUTE.enabled:
            return out
        h = x.astype(mx.float32)
        if self.dropout > 0 and self.training:
            h = nn.Dropout(self.dropout)(h)
        update = ((h @ self.lora_a) @ self.lora_b) * self.scaling
        mask = _ROUTE.mask
        if mask is not None:
            if mask.ndim != 2:
                raise ValueError(f"LoRA token mask must be [B, T], got {mask.shape}")
            if mask.shape != x.shape[:-1]:
                raise ValueError(
                    f"LoRA token mask {mask.shape} does not match activation "
                    f"{tuple(x.shape[:-1])}. The mask must cover exactly the rows in "
                    "this forward, including any cached-prefix offset."
                )
            update = update * mask.astype(mx.float32)[..., None]
        return out + update.astype(out.dtype)


@dataclass
class GatedLoraReport:
    injected: list[str]
    params: int
    rank: int
    alpha: float
    families: list[str] = field(default_factory=list)
    per_family: dict[str, int] = field(default_factory=dict)

    @property
    def num_injected(self) -> int:
        return len(self.injected)

    def to_dict(self) -> dict:
        return {
            "modules": self.num_injected,
            "params": self.params,
            "rank": self.rank,
            "alpha": self.alpha,
            "families": self.families,
            "per_family": self.per_family,
        }


def _wrap(parent, attr: str, rank: int, alpha: float, dropout: float, init_scale: float) -> int:
    base = getattr(parent, attr, None)
    if not isinstance(base, nn.Linear) or isinstance(base, GatedLoRALinear):
        return 0
    layer = GatedLoRALinear(base, rank, alpha, dropout, init_scale)
    setattr(parent, attr, layer)
    return layer.lora_a.size + layer.lora_b.size


def inject_gated_lora(
    model,
    rank: int = 16,
    alpha: float | None = None,
    dropout: float = 0.0,
    init_scale: float = 0.01,
    full_attention: bool = True,
    mlp: bool = True,
    deltanet: bool = False,
    deltanet_gates: bool = False,
    layer_indices: list[int] | None = None,
) -> GatedLoraReport:
    """Attach gated adapters in place. `model` is the raw mlx_lm `Model`.

    `alpha=None` follows Uno's convention of `alpha = 16 * rank`.

    Defaults are the closest available analogue of Uno's target set
    (`q,k,v,o,gate,up,down` on every layer): attention where the model has attention,
    plus MLP everywhere. On Qwen3.5-4B that reaches all 32 MLPs but only the 8
    full-attention layers, because the other 24 are Gated DeltaNet and expose
    different modules entirely.
    """
    if alpha is None:
        alpha = 16.0 * rank
    selected = set(layer_indices) if layer_indices is not None else None
    injected: list[str] = []
    per_family: dict[str, int] = {}
    total = 0

    def account(family: str, n: int) -> None:
        nonlocal total
        total += n
        per_family[family] = per_family.get(family, 0) + n

    for i, layer in enumerate(model.layers):
        if selected is not None and i not in selected:
            continue
        is_linear = bool(getattr(layer, "is_linear", False))

        if full_attention and not is_linear and hasattr(layer, "self_attn"):
            for attr in FULL_ATTENTION_TARGETS:
                n = _wrap(layer.self_attn, attr, rank, alpha, dropout, init_scale)
                if n:
                    injected.append(f"layers.{i}.self_attn.{attr}")
                    account("full_attention", n)

        if (deltanet or deltanet_gates) and is_linear and hasattr(layer, "linear_attn"):
            attrs = DELTANET_GATE_TARGETS if deltanet_gates else DELTANET_TARGETS
            for attr in attrs:
                n = _wrap(layer.linear_attn, attr, rank, alpha, dropout, init_scale)
                if n:
                    injected.append(f"layers.{i}.linear_attn.{attr}")
                    account("deltanet", n)

        if mlp and hasattr(layer, "mlp"):
            for attr in MLP_TARGETS:
                n = _wrap(layer.mlp, attr, rank, alpha, dropout, init_scale)
                if n:
                    injected.append(f"layers.{i}.mlp.{attr}")
                    account("mlp", n)

    families = [
        name
        for name, flag in (
            ("full_attention", full_attention),
            ("mlp", mlp),
            ("deltanet", deltanet),
            ("deltanet_gates", deltanet_gates),
        )
        if flag
    ]
    if not injected:
        raise RuntimeError(
            f"Gated LoRA injection matched no modules (families={families}, "
            f"layer_indices={sorted(selected) if selected else 'all'})."
        )
    return GatedLoraReport(
        injected=injected, params=total, rank=rank, alpha=alpha,
        families=families, per_family=per_family,
    )


def gated_lora_modules(model) -> list[GatedLoRALinear]:
    """Every adapter in the tree, found by walking children (not by name)."""
    found: list[GatedLoRALinear] = []
    seen: set[int] = set()

    def walk(module) -> None:
        if id(module) in seen:
            return
        seen.add(id(module))
        if isinstance(module, GatedLoRALinear):
            found.append(module)
        children = getattr(module, "children", None)
        if children is None:
            return
        for child in children().values():
            if isinstance(child, nn.Module):
                walk(child)
            elif isinstance(child, (list, tuple)):
                for item in child:
                    if isinstance(item, nn.Module):
                        walk(item)
            elif isinstance(child, dict):
                for item in child.values():
                    if isinstance(item, nn.Module):
                        walk(item)

    walk(model)
    return found


def configure_trainable(model) -> dict:
    """Freeze the backbone; unfreeze only `lora_a`/`lora_b`.

    MLX expresses trainability by freezing modules, and `nn.value_and_grad`
    differentiates only w.r.t. `trainable_parameters()`. Freezing everything first and
    unfreezing a named allowlist means a backbone weight cannot receive a gradient
    even in principle -- there is no tensor for one to land on.
    """
    from mlx.utils import tree_flatten

    model.freeze(recurse=True)
    adapters = gated_lora_modules(model)
    if not adapters:
        raise RuntimeError("configure_trainable() found no gated adapters to unfreeze.")
    for module in adapters:
        module.unfreeze(recurse=False, keys=["lora_a", "lora_b"])

    trainable = tree_flatten(model.trainable_parameters())
    offenders = [name for name, _ in trainable if not name.endswith(("lora_a", "lora_b"))]
    if offenders:
        raise RuntimeError(
            "Only LoRA A/B tensors may be trainable; found "
            f"{offenders[:5]} ({len(offenders)} total)."
        )
    return {
        "adapters": len(adapters),
        "trainable_params": sum(v.size for _, v in trainable),
        "trainable_tensors": len(trainable),
    }
