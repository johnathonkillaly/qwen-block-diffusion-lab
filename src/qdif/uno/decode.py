"""Decoding: the AR baseline, and Uno's draft→verify cycle.

Both decoders return the same `DecodeStats` shape so that a speed claim is always a
comparison of like with like, measured in one harness.

Three distinct achievements are tracked separately, because conflating them is the
easiest way to overclaim (see `docs/act4u_uno_source_notes.md`):

  1. `forwards`        — sequential model iterations. The *algorithmic* metric.
  2. `forward_tokens`  — total token-positions pushed through the model. The
                         compute metric; a block method does strictly more of this.
  3. `wall_seconds`    — the only thing a user experiences.

A method can win 1, lose 3, and still be scientifically interesting. It is not a
speed win unless it wins 3.

## Why the Uno cycle costs 2 forwards here, and sometimes 3

IFM roll the KV cache back to an arbitrary length between the draft and verify passes.
We cannot: 24 of our 32 layers are recurrent, and a recurrence cannot be rewound into
the middle of a forward (`cache_utils.py`). Two consequences:

  * the verify pass re-includes the seed token, so it is L+1 wide rather than L;
  * on a *partial* acceptance the cache is rewound to the last committed frontier and
    the accepted tokens are replayed in one extra forward.

On a **fully accepted** block the verify pass has already advanced the cache to
exactly the right place, so no replay is needed and the cycle costs 2 forwards. The
tax therefore falls precisely on the cycles where the adapter did badly, which is the
right place for it, but it is a real architecture-specific cost and is reported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx

from .cache_utils import cache_length, restore_caches, snapshot_caches
from .noise import build_draft_block, draft_lora_mask
from .transaction import TransactionStats, begin_transaction
from .verifier import greedy_accept


@dataclass
class DecodeStats:
    tokens: int = 0
    forwards: int = 0
    forward_tokens: int = 0
    cycles: int = 0
    accepted_specs: int = 0
    offered_specs: int = 0
    full_blocks: int = 0
    replay_forwards: int = 0
    committed_per_cycle: list[int] = field(default_factory=list)
    accepted_per_cycle: list[int] = field(default_factory=list)
    #: Verifier entropy at each block's first position — the frozen model's own
    #: uncertainty about where the continuation is going. Paired with
    #: `accepted_per_cycle` this tests the brief's §26 hypothesis directly.
    entropy_per_cycle: list[float] = field(default_factory=list)
    wall_seconds: float = 0.0
    prefill_seconds: float = 0.0
    transaction_mode: str = "replay"
    transaction: object = None

    @property
    def tokens_per_forward(self) -> float:
        return self.tokens / self.forwards if self.forwards else 0.0

    @property
    def tokens_per_forward_without_replay(self) -> float:
        """Counterfactual TPF on a backbone whose state could be rewound.

        The replay forward exists only because a Gated DeltaNet recurrence cannot be
        rewound into the middle of a pass (`cache_utils.py`). On a pure-attention
        backbone — IFM's Qwen3-8B, for instance — a partially-accepted cycle costs 2
        forwards, not 3. Reporting both separates "the algorithm does not work" from
        "this architecture cannot express the algorithm efficiently", which are very
        different findings.

        This is an accounting counterfactual, not a measurement. It is never a speed
        claim: no wall-clock number is derived from it.
        """
        usable = self.forwards - self.replay_forwards
        return self.tokens / usable if usable else 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_specs / self.offered_specs if self.offered_specs else 0.0

    @property
    def mean_committed_per_cycle(self) -> float:
        n = len(self.committed_per_cycle)
        return sum(self.committed_per_cycle) / n if n else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.wall_seconds if self.wall_seconds else 0.0

    def to_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "forwards": self.forwards,
            "forward_tokens": self.forward_tokens,
            "cycles": self.cycles,
            "tokens_per_forward": round(self.tokens_per_forward, 4),
            "tokens_per_forward_without_replay": round(
                self.tokens_per_forward_without_replay, 4
            ),
            "acceptance_rate": round(self.acceptance_rate, 4),
            "mean_committed_per_cycle": round(self.mean_committed_per_cycle, 4),
            "full_blocks": self.full_blocks,
            "replay_forwards": self.replay_forwards,
            "wall_seconds": round(self.wall_seconds, 4),
            "prefill_seconds": round(self.prefill_seconds, 4),
            "tokens_per_second": round(self.tokens_per_second, 3),
            "transaction_mode": self.transaction_mode,
            "transaction": self.transaction.to_dict() if self.transaction else None,
        }


def _stop_index(tokens: list[int], stop_ids: set[int]) -> int | None:
    for index, token in enumerate(tokens):
        if token in stop_ids:
            return index
    return None


# --------------------------------------------------------------------------- AR


def ar_greedy_generate(
    model,
    prompt_ids: list[int],
    max_tokens: int,
    stop_ids: set[int] | None = None,
    use_cache: bool = True,
) -> tuple[list[int], DecodeStats]:
    """Greedy decoding from the frozen backbone. The reference every claim is against.

    `use_cache=False` recomputes the full context each step. It is O(n²) and exists so
    that the cached path can be proven to produce identical tokens.
    """
    stop_ids = stop_ids or set()
    stats = DecodeStats()
    generated: list[int] = []
    start = time.perf_counter()

    if use_cache:
        caches = model.make_cache()
        prefill_start = time.perf_counter()
        logits = model.ar_logits(mx.array([prompt_ids], dtype=mx.int32), cache=caches)
        mx.eval(logits)
        stats.prefill_seconds = time.perf_counter() - prefill_start
        stats.forwards += 1
        stats.forward_tokens += len(prompt_ids)
        token = int(mx.argmax(logits[0, -1]).item())

        while True:
            generated.append(token)
            if token in stop_ids or len(generated) >= max_tokens:
                break
            logits = model.ar_logits(mx.array([[token]], dtype=mx.int32), cache=caches)
            stats.forwards += 1
            stats.forward_tokens += 1
            token = int(mx.argmax(logits[0, -1]).item())
    else:
        while len(generated) < max_tokens:
            ids = mx.array([prompt_ids + generated], dtype=mx.int32)
            logits = model.ar_logits(ids)
            stats.forwards += 1
            stats.forward_tokens += ids.shape[1]
            token = int(mx.argmax(logits[0, -1]).item())
            generated.append(token)
            if token in stop_ids:
                break

    stats.wall_seconds = time.perf_counter() - start
    stats.tokens = len(generated)
    stats.cycles = stats.forwards
    return generated, stats


# -------------------------------------------------------------------------- Uno


def uno_greedy_generate(
    model,
    prompt_ids: list[int],
    max_tokens: int,
    block_size: int,
    noise_mode: str = "random_uniform",
    noise_seed: int | None = None,
    stop_ids: set[int] | None = None,
    use_cache: bool = True,
    transaction_mode: str = "replay",
) -> tuple[list[int], DecodeStats]:
    """Uno draft→verify decoding. Greedy, so the output is meant to be *identical*
    to `ar_greedy_generate` — the adapter changes how fast tokens arrive, never
    which tokens arrive.

    Any divergence from the AR baseline is a bug in this function or in the verifier,
    not a property of the adapter. Gate 3 tests exactly that.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    stop_ids = stop_ids or set()
    if not use_cache:
        return _uno_greedy_uncached(
            model, prompt_ids, max_tokens, block_size, noise_mode, noise_seed, stop_ids
        )

    stats = DecodeStats()
    stats.transaction_mode = transaction_mode
    txn_stats = TransactionStats()
    generated: list[int] = []
    start = time.perf_counter()

    caches = model.make_cache()
    # Prefill everything but the final prompt token: that token becomes the first
    # seed, and the seed must stay uncached so the draft pass can read its logits.
    prefill_start = time.perf_counter()
    if len(prompt_ids) > 1:
        logits = model.ar_logits(
            mx.array([prompt_ids[:-1]], dtype=mx.int32), cache=caches
        )
        mx.eval(logits)
        stats.forwards += 1
        stats.forward_tokens += len(prompt_ids) - 1
    stats.prefill_seconds = time.perf_counter() - prefill_start
    seed = prompt_ids[-1]
    cycle = 0

    while len(generated) < max_tokens:
        cycle += 1
        snapshot = snapshot_caches(caches)
        frontier = cache_length(caches)

        # --- draft: [seed, noise...] with the adapter off on the seed row ----------
        draft_ids = build_draft_block(
            mx.array([seed], dtype=mx.int32),
            block_size,
            noise_mode,
            model.mask_token_id,
            model.vocab_size,
            seed=None if noise_seed is None else noise_seed + cycle,
        )
        mask = draft_lora_mask(1, block_size, prefix_len=0)
        draft_logits = model.draft_logits(draft_ids, mask, cache=caches)
        stats.forwards += 1
        stats.forward_tokens += block_size
        proposal = mx.argmax(draft_logits[0], axis=-1).tolist()
        proposal = [int(t) for t in proposal]  # [c, p_1, ..., p_{L-1}]
        restore_caches(caches, snapshot)

        # --- verify: [seed] + proposal, adapter fully off --------------------------
        # The leading seed is re-sent because the cache cannot be advanced by exactly
        # one token without re-running it (see module docstring).
        verify_ids = mx.array([[seed] + proposal], dtype=mx.int32)
        transaction = begin_transaction(
            model, caches, mode=transaction_mode, stats=txn_stats
        )
        verify_logits = transaction.run_block(verify_ids)
        stats.forward_tokens += int(verify_ids.shape[1])
        result = greedy_accept(proposal, verify_logits[0, 1:])

        row = verify_logits[0, 0].astype(mx.float32)
        log_probs = row - mx.logsumexp(row)
        stats.entropy_per_cycle.append(
            float(-mx.sum(mx.exp(log_probs) * log_probs).item())
        )

        stats.cycles += 1
        stats.accepted_specs += result.accepted_specs
        stats.offered_specs += result.offered_specs
        stats.full_blocks += int(result.full_block)
        committed = result.committed

        # Trim to the token budget and to any stop token.
        budget = max_tokens - len(generated)
        truncated = len(committed) > budget
        committed = committed[:budget]
        stop_at = _stop_index(committed, stop_ids)
        if stop_at is not None:
            committed = committed[: stop_at + 1]
            truncated = True

        stats.committed_per_cycle.append(len(committed))
        stats.accepted_per_cycle.append(result.accepted_specs)
        generated.extend(committed)

        if stop_at is not None or len(generated) >= max_tokens:
            transaction.commit_prefix(min(len(committed), int(verify_ids.shape[1])))
            break

        # --- advance the cache to the new committed frontier -----------------------
        # Committing `m` tokens must leave the cache at `frontier + m`: the verify
        # block's first `m` tokens are exactly `[seed] + committed[:-1]`, and the last
        # committed token becomes the next uncached seed.
        transaction.commit_prefix(len(committed))
        if cache_length(caches) != frontier + len(committed):
            raise RuntimeError(
                f"cache invariant violated after commit: "
                f"{cache_length(caches)} != {frontier + len(committed)}"
            )
        seed = committed[-1]

    # Forward accounting comes from the transaction, which is the only thing that
    # knows whether a commit needed a replay. `stats.forwards` counted the prefill and
    # each draft; the transaction counted each verify block plus any replay.
    stats.forwards += txn_stats.full_forwards
    stats.replay_forwards = txn_stats.replay_forwards
    stats.forward_tokens += txn_stats.replay_tokens
    stats.transaction = txn_stats
    stats.wall_seconds = time.perf_counter() - start
    stats.tokens = len(generated)
    return generated, stats


def _uno_greedy_uncached(
    model,
    prompt_ids: list[int],
    max_tokens: int,
    block_size: int,
    noise_mode: str,
    noise_seed: int | None,
    stop_ids: set[int],
) -> tuple[list[int], DecodeStats]:
    """The obvious, slow, obviously-correct version: no cache at all.

    Everything is recomputed from the full context on every forward, so there is no
    cache state that could be stale or mis-rewound. `tests/test_uno_cache.py` requires
    this and the cached path to emit identical tokens.
    """
    stats = DecodeStats()
    generated: list[int] = []
    start = time.perf_counter()
    cycle = 0

    while len(generated) < max_tokens:
        cycle += 1
        context = prompt_ids + generated  # seed is context[-1]
        seed = context[-1]

        draft_block = build_draft_block(
            mx.array([seed], dtype=mx.int32),
            block_size,
            noise_mode,
            model.mask_token_id,
            model.vocab_size,
            seed=None if noise_seed is None else noise_seed + cycle,
        )
        prefix = mx.array([context[:-1]], dtype=mx.int32)
        draft_ids = mx.concatenate([prefix, draft_block], axis=1)
        mask = draft_lora_mask(1, block_size, prefix_len=prefix.shape[1])
        draft_logits = model.draft_logits(draft_ids, mask)
        stats.forwards += 1
        stats.forward_tokens += draft_ids.shape[1]
        proposal = [int(t) for t in mx.argmax(draft_logits[0, -block_size:], axis=-1).tolist()]

        verify_ids = mx.array([context + proposal], dtype=mx.int32)
        verify_logits = model.ar_logits(verify_ids)
        stats.forwards += 1
        stats.forward_tokens += verify_ids.shape[1]
        result = greedy_accept(proposal, verify_logits[0, -block_size:])

        stats.cycles += 1
        stats.accepted_specs += result.accepted_specs
        stats.offered_specs += result.offered_specs
        stats.full_blocks += int(result.full_block)

        committed = result.committed[: max_tokens - len(generated)]
        stop_at = _stop_index(committed, stop_ids)
        if stop_at is not None:
            committed = committed[: stop_at + 1]
        stats.committed_per_cycle.append(len(committed))
        stats.accepted_per_cycle.append(result.accepted_specs)
        generated.extend(committed)
        if stop_at is not None:
            break

    stats.wall_seconds = time.perf_counter() - start
    stats.tokens = len(generated)
    return generated, stats
