"""Memory estimates for inference and training configurations.

Every estimate is broken into the four things that actually consume memory, because
lumping them together is how people conclude a 27B LoRA run fits in 32 GB:

    parameters      -- weights, at the storage dtype
    gradients       -- one per *trainable* parameter, at compute dtype
    optimizer state -- AdamW keeps 2 fp32 moments per trainable parameter
    activations     -- transient, scales with batch x sequence x hidden x layers
    runtime state   -- KV cache for full-attention layers + DeltaNet recurrent and
                       conv state for linear-attention layers

APPLE SILICON. On MPS these numbers are *unified memory*, shared with the OS and
the CPU. It is not VRAM, and the practical ceiling is well below the machine's
total: `recommendedMaxWorkingSetSize` is typically ~75% of installed RAM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BYTES_PER_DTYPE = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}
#: Quantised weights carry per-group scales/zero-points; ~10% is a realistic allowance.
QUANT_OVERHEAD = {"int8": 1.10, "int4": 1.15}


@dataclass
class MemoryEstimate:
    label: str
    params_bytes: float
    grad_bytes: float = 0.0
    optimizer_bytes: float = 0.0
    activation_bytes: float = 0.0
    runtime_state_bytes: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> float:
        return (
            self.params_bytes
            + self.grad_bytes
            + self.optimizer_bytes
            + self.activation_bytes
            + self.runtime_state_bytes
        )

    def as_row(self) -> dict:
        gb = 1e9
        return {
            "config": self.label,
            "parameters_gb": round(self.params_bytes / gb, 2),
            "gradients_gb": round(self.grad_bytes / gb, 2),
            "optimizer_gb": round(self.optimizer_bytes / gb, 2),
            "activations_gb": round(self.activation_bytes / gb, 2),
            "runtime_state_gb": round(self.runtime_state_bytes / gb, 3),
            "total_gb": round(self.total_bytes / gb, 2),
            "notes": "; ".join(self.notes),
        }


@dataclass
class ModelShape:
    """Just enough architecture to estimate memory, decoupled from any checkpoint."""

    total_params: int
    hidden_size: int
    num_layers: int
    num_full_attention_layers: int
    num_linear_attention_layers: int
    vocab_size: int
    num_key_value_heads: int = 4
    head_dim: int = 256
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    @classmethod
    def from_report(cls, report) -> "ModelShape":
        return cls(
            total_params=report.total_params,
            hidden_size=report.hidden_size,
            num_layers=report.num_hidden_layers,
            num_full_attention_layers=len(report.full_attention_layers),
            num_linear_attention_layers=len(report.linear_attention_layers),
            vocab_size=report.vocab_size,
            num_key_value_heads=report.num_key_value_heads,
            head_dim=report.head_dim,
            linear_num_value_heads=report.linear_num_value_heads or 16,
            linear_key_head_dim=report.linear_key_head_dim or 128,
            linear_value_head_dim=report.linear_value_head_dim or 128,
            linear_conv_kernel_dim=report.linear_conv_kernel_dim or 4,
        )


def runtime_state_bytes(shape: ModelShape, seq_len: int, batch: int, dtype: str = "bf16") -> float:
    """KV cache (full-attention layers) + conv/recurrent state (DeltaNet layers).

    The asymmetry is the architectural point: KV cache grows with sequence length,
    DeltaNet state does not. A hybrid 3:1 stack therefore has roughly a quarter of
    the long-context cache of a pure-attention model of the same depth.
    """
    b = BYTES_PER_DTYPE[dtype]
    kv = (
        2  # K and V
        * shape.num_full_attention_layers
        * batch
        * seq_len
        * shape.num_key_value_heads
        * shape.head_dim
        * b
    )
    recurrent = (
        shape.num_linear_attention_layers
        * batch
        * shape.linear_num_value_heads
        * shape.linear_key_head_dim
        * shape.linear_value_head_dim
        * 4.0  # states are held in fp32 (config: mamba_ssm_dtype)
    )
    key_dim = shape.linear_num_value_heads * shape.linear_key_head_dim
    value_dim = shape.linear_num_value_heads * shape.linear_value_head_dim
    conv = (
        shape.num_linear_attention_layers
        * batch
        * (2 * key_dim + value_dim)
        * shape.linear_conv_kernel_dim
        * b
    )
    return kv + recurrent + conv


def activation_bytes(
    shape: ModelShape, seq_len: int, batch: int, dtype: str = "bf16", training: bool = False
) -> float:
    """Rough activation footprint.

    Inference keeps roughly a couple of hidden-sized buffers per layer. Training must
    keep activations for the backward pass; ~12 hidden-sized tensors per layer is a
    workable rule of thumb for a gated-attention + gated-MLP block without
    checkpointing. Also counts the logits tensor, which is far from negligible with a
    248k vocabulary.
    """
    b = BYTES_PER_DTYPE[dtype]
    per_layer = 12 if training else 2
    hidden = batch * seq_len * shape.hidden_size * b * per_layer * shape.num_layers
    # Logits are computed in the compute dtype then upcast to fp32 for the loss.
    logits = batch * seq_len * shape.vocab_size * (b + 4.0 if training else b)
    return hidden + logits


def estimate(
    shape: ModelShape,
    label: str,
    weight_dtype: str = "bf16",
    compute_dtype: str = "bf16",
    trainable_params: int = 0,
    seq_len: int = 512,
    batch: int = 1,
    training: bool = False,
    notes: list[str] | None = None,
) -> MemoryEstimate:
    params = shape.total_params * BYTES_PER_DTYPE[weight_dtype] * QUANT_OVERHEAD.get(weight_dtype, 1.0)
    grads = trainable_params * BYTES_PER_DTYPE[compute_dtype] if training else 0.0
    # AdamW: exp_avg + exp_avg_sq, fp32.
    opt = trainable_params * 8.0 if training else 0.0
    acts = activation_bytes(shape, seq_len, batch, compute_dtype, training=training)
    state = 0.0 if training else runtime_state_bytes(shape, seq_len, batch, compute_dtype)
    return MemoryEstimate(
        label=label,
        params_bytes=params,
        grad_bytes=grads,
        optimizer_bytes=opt,
        activation_bytes=acts,
        runtime_state_bytes=state,
        notes=notes or [],
    )


def standard_profiles(
    shape: ModelShape, lora_params: int, seq_len: int = 512, batch: int = 1
) -> list[MemoryEstimate]:
    """The six configurations `qdif inspect` reports."""
    aux = lora_params
    return [
        estimate(shape, "BF16 inference", "bf16", "bf16", 0, seq_len, batch, False),
        estimate(
            shape,
            "8-bit inference",
            "int8",
            "bf16",
            0,
            seq_len,
            batch,
            False,
            notes=["MLX quantized weights; not supported for training in this harness"],
        ),
        estimate(
            shape,
            "4-bit inference",
            "int4",
            "bf16",
            0,
            seq_len,
            batch,
            False,
            notes=["MLX 4-bit; expect measurable quality loss at 0.8B scale"],
        ),
        estimate(
            shape,
            "BF16 full training",
            "bf16",
            "bf16",
            shape.total_params,
            seq_len,
            batch,
            True,
            notes=["AdamW fp32 moments dominate; not the default path"],
        ),
        estimate(
            shape,
            "BF16 base + LoRA",
            "bf16",
            "bf16",
            aux,
            seq_len,
            batch,
            True,
            notes=["implemented and tested on MPS"],
        ),
        estimate(
            shape,
            "4-bit base + LoRA (QLoRA)",
            "int4",
            "bf16",
            aux,
            seq_len,
            batch,
            True,
            notes=["NOT implemented on Apple silicon: bitsandbytes is CUDA-only"],
        ),
    ]
