"""Transactional decode state.

The decoder should not know how a rejected speculative suffix gets discarded. It
should say "keep the first `m` of the tokens I just pushed through" and get correct
state back. That is what this module provides:

    txn = begin_transaction(model, caches, mode="snapshot")
    logits = txn.run_block(input_ids)     # one model forward, speculative
    txn.commit_prefix(m)                  # retain exactly m of those tokens
    # or txn.rollback()                   # retain none

Three backends, all exposing identical semantics so they can be diffed against each
other on real workloads:

| mode | how a prefix is committed | extra full model forwards |
|---|---|---|
| `replay` | rewind to the pre-block snapshot, re-run `m` tokens through the model | **1** |
| `snapshot` | select the recorded per-token state | **0** |
| `rewind` | algebraically undo the rejected suffix | **0** |

`replay` is the correctness oracle and reproduces Act IV-U exactly. It is never
deleted, and every other backend is tested against it.

Why `snapshot` needs no extra forward: the state at every token boundary is captured
*during the block forward itself* by unrolling only the recurrence
(`recurrence.py`). Projections, convolution, attention and MLPs still run once.

See `docs/act4u2_transactional_state.md` for the measured state anatomy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from .cache_utils import cache_length, restore_caches, snapshot_caches
from .gated_lora import lora_disabled
from .recurrence import BlockRecord, text_model_block_forward

TRANSACTION_MODES = ("replay", "snapshot", "rewind")


@dataclass
class TransactionStats:
    """Cost accounting. `full_forwards` is the number that decides Gate U2-2."""

    full_forwards: int = 0
    forward_tokens: int = 0
    commits: int = 0
    rollbacks: int = 0
    #: Commits that needed a replay forward through the whole model.
    replay_forwards: int = 0
    replay_tokens: int = 0
    #: Commits served purely from recorded/derived state.
    state_only_commits: int = 0
    #: Per-head algebraic rewinds that fell back to a snapshot (mode="rewind").
    rewind_fallbacks: int = 0
    rewind_steps: int = 0
    peak_record_bytes: int = 0

    def merge(self, other: "TransactionStats") -> None:
        for key, value in other.__dict__.items():
            if key == "peak_record_bytes":
                self.peak_record_bytes = max(self.peak_record_bytes, value)
            else:
                setattr(self, key, getattr(self, key) + value)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class BlockTransaction:
    """One speculative block: run it, then keep a prefix of it.

    Not reusable — `begin_transaction` returns a fresh one per block, which keeps the
    lifetime of the (large) recorded state obvious.
    """

    def __init__(self, model, caches, mode: str, stats: TransactionStats,
                 rewind_max_amplification: float = 100.0):
        if mode not in TRANSACTION_MODES:
            raise ValueError(f"unknown transaction mode {mode!r}; expected {TRANSACTION_MODES}")
        self.model = model
        self.caches = caches
        self.mode = mode
        self.stats = stats
        self.rewind_max_amplification = rewind_max_amplification
        self.base_length = cache_length(caches)
        self._snapshot = snapshot_caches(caches)
        self._record: BlockRecord | None = None
        self._block_ids: mx.array | None = None
        self._closed = False

    # ---------------------------------------------------------------- running
    def run_block(self, input_ids: mx.array) -> mx.array:
        """Push a speculative block through the frozen model. Adapter always off."""
        if self._closed:
            raise RuntimeError("transaction already closed")
        self._block_ids = input_ids
        self.stats.full_forwards += 1
        self.stats.forward_tokens += int(input_ids.shape[1])

        if self.mode == "replay":
            with lora_disabled():
                return self.model.base(input_ids, cache=self.caches)

        with lora_disabled():
            logits, record = text_model_block_forward(
                self.model.base,
                input_ids,
                self.caches,
                keep_factors=(self.mode == "rewind"),
            )
        self._record = record
        self.stats.peak_record_bytes = max(self.stats.peak_record_bytes, record.nbytes)
        return logits

    # --------------------------------------------------------------- committing
    def commit_prefix(self, keep: int) -> None:
        """Retain exactly the first `keep` tokens of the block just run."""
        if self._closed:
            raise RuntimeError("transaction already closed")
        if self._block_ids is None:
            raise RuntimeError("commit_prefix() before run_block()")
        block_length = int(self._block_ids.shape[1])
        if not 0 <= keep <= block_length:
            raise ValueError(f"keep={keep} outside [0, {block_length}]")
        self.stats.commits += 1

        if keep == block_length:
            # The block forward already left every cache in exactly the right place.
            self.stats.state_only_commits += 1
            self._closed = True
            return

        if self.mode == "replay":
            self._commit_by_replay(keep)
        else:
            self._commit_from_state(keep)
        self._closed = True

    def rollback(self) -> None:
        self.stats.rollbacks += 1
        self.commit_prefix(0)

    # ------------------------------------------------------------------ backends
    def _commit_by_replay(self, keep: int) -> None:
        """Approach A: rewind everything, re-run the kept tokens through the model."""
        restore_caches(self.caches, self._snapshot)
        if keep == 0:
            return
        replay_ids = self._block_ids[:, :keep]
        with lora_disabled():
            out = self.model.base(replay_ids, cache=self.caches)
        mx.eval(out)
        self.stats.full_forwards += 1
        self.stats.replay_forwards += 1
        self.stats.replay_tokens += keep
        self.stats.forward_tokens += keep

    def _commit_from_state(self, keep: int) -> None:
        """Approaches B and C: no model forward at all.

        Attention caches rewind by offset. DeltaNet layers take their recorded state
        (`snapshot`) or an algebraically undone one (`rewind`), and their convolution
        window is sliced out of the recorded conv input.
        """
        record = self._record
        if record is None:
            raise RuntimeError("no block record; run_block() must precede commit")

        for index, layer in enumerate(self.model.base.layers):
            cache = self.caches[index]
            if not getattr(layer, "is_linear", False):
                cache.offset = record.kv_offset_before[index] + keep
                continue

            layer_record = record.layers[index]
            keep_rows = layer.linear_attn.conv_kernel_size - 1
            cache[0] = layer_record.conv_state_at(keep, keep_rows)
            if self.mode == "rewind":
                cache[1] = self._rewound_state(
                    layer.linear_attn, layer_record, keep, record.block_length
                )
            else:
                cache[1] = layer_record.state_at(keep)

        self.stats.state_only_commits += 1

    def _rewound_state(self, attn, record, keep: int, block_length: int) -> mx.array:
        """Approach C: undo `block_length - keep` rank-1 updates algebraically.

            S̄_t     = S_t + (β/(1−β))·(S_t k − v) kᵀ
            S_{t-1} = S̄_t / g

        **Two independent things destroy information in the forward pass**, and a
        guard on either alone is not enough:

        1. **β → 1.** The update becomes `S̄(I − kkᵀ) + vkᵀ`, an orthogonal projection.
           The component of `S̄` along `k` is annihilated. Measured on real text, β
           reaches exactly 1.0 in bf16 for 0.043% of head-steps.
        2. **g → 0.** The decay multiplies the whole previous state by `g`, so a small
           `g` *forgets* it; dividing back by `g` amplifies whatever rounding survived.
           Measured decay minimum on real text is 0.0, and on the tiny fixture a single
           step showed `g ≈ 1.1e-9` and a rewind error of 9e4 against a state whose
           own magnitude is 0.019.

        The guard is therefore on the **combined per-head amplification**
        `1/((1−β)·g)`, and a head that exceeds it falls back to its recorded state.
        Fallbacks are counted, because a rewind that silently falls back everywhere is
        a snapshot wearing a costume.
        """
        from mlx_lm.models.gated_delta import compute_g

        state = record.states[block_length].astype(mx.float32)
        # Error compounds across undo steps, so the guard is on the *cumulative*
        # amplification since the last exact state, not on any single step. Falling
        # back resets a head's accumulated error to zero, so the counter resets too.
        cumulative = None

        for step in range(block_length - 1, keep - 1, -1):
            factor = record.factors[step]
            k = factor["k"].astype(mx.float32)  # [B, Hk, Dk]
            v = factor["v"].astype(mx.float32)  # [B, Hv, Dv]
            beta = mx.sigmoid(factor["b"]).astype(mx.float32)  # [B, Hv]
            g = compute_g(attn.A_log, factor["a"], attn.dt_bias).astype(mx.float32)

            repeat = v.shape[1] // k.shape[1]
            if repeat > 1:
                k = mx.repeat(k, repeat, axis=1)

            gap = mx.maximum(1.0 - beta, 0.0)
            amplification = 1.0 / mx.maximum(gap * g, 1e-30)
            if cumulative is None:
                cumulative = mx.ones_like(amplification)
            running = cumulative * amplification
            safe = running <= self.rewind_max_amplification

            coefficient = mx.where(safe, beta / mx.maximum(gap, 1e-6), 0.0)
            sk = mx.sum(state * k[..., None, :], axis=-1)  # [B, Hv, Dv]
            bar = state + (coefficient[..., None] * (sk - v))[..., :, None] * k[..., None, :]
            candidate = bar / mx.where(safe, mx.maximum(g, 1e-30), 1.0)[..., None, None]

            exact = record.states[step].astype(mx.float32)
            state = mx.where(safe[..., None, None], candidate, exact)
            cumulative = mx.where(safe, running, mx.ones_like(running))

            self.stats.rewind_fallbacks += int(mx.sum((~safe).astype(mx.int32)).item())
            self.stats.rewind_steps += int(safe.size)

        return state


def begin_transaction(model, caches, mode: str = "snapshot",
                      stats: TransactionStats | None = None,
                      rewind_max_amplification: float = 100.0) -> BlockTransaction:
    return BlockTransaction(
        model, caches, mode, stats if stats is not None else TransactionStats(),
        rewind_max_amplification=rewind_max_amplification,
    )
