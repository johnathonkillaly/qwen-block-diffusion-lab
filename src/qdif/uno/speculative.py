"""Decoder variants the Act IV-U work never had: refinement, adaptive K, and an
n-gram control drafter.

Everything here is *additive*. `decode.uno_greedy_generate` is untouched, so every
Act IV-U3/U4 number still replays from the same code path that produced it. Three new
capabilities, each isolated so it can be ablated:

  1. **Refinement** (`refine_steps`). The Act IV-U draft pass is single-shot: one
     forward over `[seed, noise…]` yields the whole proposal. Refinement re-runs that
     pass with the previous proposal in place of the noise, which is what makes the
     block a *diffusion* denoiser rather than a one-shot block predictor. Each extra
     step costs one more forward of width `L`, so it has to earn ~1.5 ms/slot.

  2. **Adaptive K** (`adaptive_greedy_generate`). Chooses the block width per cycle
     from a policy. The constraint that shapes every policy here: **the width must be
     fixed before the draft forward runs**, so no feature computed from the current
     cycle's draft or verify pass is available to choose it. Only history is.

  3. **An n-gram drafter** (`ngram_greedy_generate`). The brief's §17 control. Same
     verifier, same transaction, same accounting — the *only* thing that changes is
     where the proposal comes from. If this matches the adapter, diffusion is not what
     is buying the speedup.

All three commit through `verifier.greedy_accept` and the same transaction, so the
losslessness guarantee is structural rather than re-argued per decoder.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx

from .cache_utils import cache_length, restore_caches, snapshot_caches
from .decode import DecodeStats, _cycle_key, _stop_index
from .noise import build_draft_block, draft_lora_mask
from .transaction import TransactionStats, begin_transaction
from .verifier import greedy_accept


# --------------------------------------------------------------- draft proposal


def _draft_proposal(
    model,
    seed: int,
    block_size: int,
    caches,
    snapshot,
    noise_mode: str,
    key,
    refine_steps: int,
    stats: DecodeStats,
    collect_confidence: bool = False,
) -> tuple[list[int], list[float], list[float]]:
    """One draft, plus `refine_steps - 1` refinement passes.

    Returns `(proposal, top1_probs, margins)`; the confidence lists are empty unless
    `collect_confidence`.

    **Indexing, because it is the easy thing to get wrong.** The draft block is
    `[t₀, x₁ … x_{L-1}]` where `t₀` is the seed and `x_i` is a placeholder for the
    unknown token `t_i`. Output position `i` predicts `t_{i+1}`, so
    `proposal = [t₁ … t_L]` and the current estimate of `t_i` is `proposal[i-1]`.
    Refinement therefore re-runs the pass over `[seed] + proposal[:-1]`, which puts
    each estimate in the slot that estimates it. Using `proposal[1:]` instead would
    shift the canvas by one and quietly make refinement destructive.

    Each pass restores the cache first, because a draft forward advances it and the
    next pass must start from the same committed frontier.

    `collect_confidence` is off during throughput runs: a top-2 reduction over a
    151,936-wide vocabulary is not free, and charging a diagnostic to the proposal
    timer would corrupt the cost model it feeds.
    """
    proposal: list[int] = []
    top1: list[float] = []
    margin: list[float] = []
    block = build_draft_block(
        mx.array([seed], dtype=mx.int32),
        block_size,
        noise_mode,
        model.mask_token_id,
        model.vocab_size,
        seed=None,
        key=key,
    )
    mask = draft_lora_mask(1, block_size, prefix_len=0)

    for step in range(max(1, refine_steps)):
        if step > 0:
            restore_caches(caches, snapshot)
            # Feed the previous proposal back in place of the noise, aligned so that
            # slot i holds the current estimate of t_i. The seed row keeps the true
            # seed, so the adapter-off position still yields the frozen model's token.
            block = mx.array([[seed] + proposal[:-1]], dtype=mx.int32)
        stage_start = time.perf_counter()
        logits = model.draft_logits(block, mask, cache=caches)
        mx.eval(logits)
        stats.proposal_seconds += time.perf_counter() - stage_start
        stats.forwards += 1
        stats.forward_tokens += block_size

        row = logits[0].astype(mx.float32)
        proposal = [int(t) for t in mx.argmax(row, axis=-1).tolist()]

    if collect_confidence:
        probs = mx.softmax(logits[0].astype(mx.float32), axis=-1)
        top2 = mx.topk(probs, 2, axis=-1)
        mx.eval(top2)
        top1 = [float(v) for v in top2[:, -1].tolist()]
        margin = [float(v) for v in (top2[:, -1] - top2[:, -2]).tolist()]

    return proposal, top1, margin


# ----------------------------------------------------------------- adaptive K


@dataclass
class KPolicy:
    """Chooses a block width for the next cycle.

    `update()` sees the outcome of the cycle that just finished; `choose()` is called
    before the next draft. The split is the point: a policy physically cannot see the
    current cycle's draft logits, because the width decides how wide that draft is.
    """

    name: str
    allowed: tuple[int, ...] = (2, 4, 8)
    default_k: int = 4

    def reset(self) -> None:  # pragma: no cover - trivial
        pass

    def choose(self) -> int:
        return self.default_k

    def update(self, accepted: int, offered: int, entropy: float) -> None:  # pragma: no cover
        pass


@dataclass
class FixedK(KPolicy):
    def choose(self) -> int:
        return self.default_k


@dataclass
class EntropyThresholdK(KPolicy):
    """Policy B: wide block when the target was confident last cycle, narrow when not.

    The feature is the verifier's entropy at the block's first speculative position,
    which `decode` already records at zero extra cost. Used with a one-cycle lag, so
    it is causal. Local difficulty is autocorrelated; that autocorrelation is the whole
    bet, and it is measured rather than assumed (`results/speculative/`).
    """

    low: float = 0.5
    high: float = 2.0
    last_entropy: float = 0.0

    def reset(self) -> None:
        self.last_entropy = 0.0

    def choose(self) -> int:
        ordered = sorted(self.allowed)
        if self.last_entropy <= self.low:
            return ordered[-1]
        if self.last_entropy >= self.high:
            return ordered[0]
        return self.default_k

    def update(self, accepted: int, offered: int, entropy: float) -> None:
        self.last_entropy = entropy


@dataclass
class SurvivalK(KPolicy):
    """Policy C: widen while the recent accepted prefix suggests the far slots survive.

    Keeps an exponentially-weighted estimate of the accepted prefix length and picks
    the largest allowed K whose speculative slots are all within reach of it.
    """

    decay: float = 0.7
    estimate: float = 1.0

    def reset(self) -> None:
        self.estimate = 1.0

    def choose(self) -> int:
        ordered = sorted(self.allowed)
        target = self.estimate + 1.0
        pick = ordered[0]
        for k in ordered:
            if (k - 1) <= target:
                pick = k
        return pick

    def update(self, accepted: int, offered: int, entropy: float) -> None:
        self.estimate = self.decay * self.estimate + (1.0 - self.decay) * accepted


@dataclass
class ExpectedValueK(KPolicy):
    """Policy D: maximise predicted committed tokens per millisecond.

    `value(K) = E[committed | K] / measured_time(K)` with `E[committed] = E[accepted] + 2`
    (exact for this decoder, see `cost_model.committed_tokens`). The survival curve is
    estimated online per allowed K from that K's own recent history, and the cost curve
    is *measured*, passed in from the calibration run rather than assumed linear.

    Falls back to `default_k` until a K has been tried `min_samples` times, so the
    scheduler cannot commit to a width on one lucky cycle.
    """

    cost_ms: dict = field(default_factory=dict)
    min_samples: int = 3
    decay: float = 0.85
    explore_every: int = 16
    _mean_accept: dict = field(default_factory=dict)
    _counts: dict = field(default_factory=dict)
    _cycle: int = 0
    _pending: int = 0

    def reset(self) -> None:
        self._mean_accept = {k: None for k in self.allowed}
        self._counts = {k: 0 for k in self.allowed}
        self._cycle = 0
        self._pending = self.default_k

    def choose(self) -> int:
        self._cycle += 1
        if not self._mean_accept:
            self.reset()
        # Round-robin exploration keeps every arm's estimate alive; without it the
        # scheduler locks onto whichever K happened to start well.
        unseen = [k for k in self.allowed if self._counts.get(k, 0) < self.min_samples]
        if unseen:
            self._pending = unseen[0]
            return self._pending
        if self._cycle % self.explore_every == 0:
            idx = (self._cycle // self.explore_every) % len(self.allowed)
            self._pending = sorted(self.allowed)[idx]
            return self._pending

        best_k, best_value = self.default_k, -1.0
        for k in self.allowed:
            accept = self._mean_accept.get(k)
            if accept is None:
                continue
            cost = self.cost_ms.get(k)
            if not cost:
                continue
            value = (accept + 2.0) / cost
            if value > best_value:
                best_k, best_value = k, value
        self._pending = best_k
        return best_k

    def update(self, accepted: int, offered: int, entropy: float) -> None:
        k = self._pending
        prev = self._mean_accept.get(k)
        self._mean_accept[k] = (
            float(accepted) if prev is None
            else self.decay * prev + (1.0 - self.decay) * accepted
        )
        self._counts[k] = self._counts.get(k, 0) + 1


POLICIES = {
    "fixed": FixedK,
    "entropy": EntropyThresholdK,
    "survival": SurvivalK,
    "ev": ExpectedValueK,
}


# ------------------------------------------------------------------- decoders


def adaptive_greedy_generate(
    model,
    prompt_ids: list[int],
    max_tokens: int,
    policy: KPolicy,
    noise_mode: str = "random_uniform",
    stop_ids: set[int] | None = None,
    transaction_mode: str = "snapshot",
    noise_stream_seed: int | None = None,
    refine_steps: int = 1,
    collect_confidence: bool = False,
) -> tuple[list[int], DecodeStats, dict]:
    """Draft→verify with a per-cycle block width, and per-cycle confidence records.

    Structurally identical to `decode.uno_greedy_generate` — same prefill, same seed
    handling, same commit rule, same cache invariant — so losslessness is inherited
    rather than re-derived. The differences are that `block_size` comes from `policy`
    each cycle, and that the drafter's own confidence is recorded alongside the
    outcome so §10's calibration questions can be answered from one run.

    The scheduler's own cost is inside `overhead_seconds`, so it is charged to the
    wall clock like everything else.
    """
    stop_ids = stop_ids or set()
    stats = DecodeStats()
    stats.transaction_mode = transaction_mode
    txn_stats = TransactionStats()
    generated: list[int] = []
    trace: list[dict] = []
    policy.reset()
    start = time.perf_counter()

    caches = model.make_cache()
    prefill_start = time.perf_counter()
    if len(prompt_ids) > 1:
        logits = model.ar_logits(mx.array([prompt_ids[:-1]], dtype=mx.int32), cache=caches)
        mx.eval(logits)
        stats.forwards += 1
        stats.forward_tokens += len(prompt_ids) - 1
    stats.prefill_seconds = time.perf_counter() - prefill_start
    seed = prompt_ids[-1]
    cycle = 0

    while len(generated) < max_tokens:
        cycle_start = time.perf_counter()
        cycle += 1
        overhead_start = cycle_start
        block_size = int(policy.choose())
        if block_size < 1:
            raise ValueError(f"policy {policy.name} chose block_size {block_size}")
        snapshot = snapshot_caches(caches)
        frontier = cache_length(caches)
        key = None if noise_stream_seed is None else _cycle_key(noise_stream_seed, cycle)
        stats.overhead_seconds += time.perf_counter() - overhead_start

        proposal, top1, margin = _draft_proposal(
            model, seed, block_size, caches, snapshot,
            noise_mode, key, refine_steps, stats,
            collect_confidence=collect_confidence,
        )

        overhead_start = time.perf_counter()
        restore_caches(caches, snapshot)
        # Width L+1: the seed is re-sent because a Gated DeltaNet cache cannot be
        # advanced by exactly one token without re-running it. Identical to
        # `decode.uno_greedy_generate`.
        verify_ids = mx.array([[seed] + proposal], dtype=mx.int32)
        transaction = begin_transaction(model, caches, mode=transaction_mode, stats=txn_stats)
        stats.overhead_seconds += time.perf_counter() - overhead_start

        stage_start = time.perf_counter()
        verify_logits = transaction.run_block(verify_ids)
        mx.eval(verify_logits)
        stats.verify_seconds += time.perf_counter() - stage_start
        stats.forward_tokens += int(verify_ids.shape[1])

        overhead_start = time.perf_counter()
        # `verify_logits[0, 0]` is p(·|context, seed) — the target's own next token,
        # which is what the adapter-off seed row already proposed. The proposal block
        # is scored against rows 1.. onward.
        result = greedy_accept(proposal, verify_logits[0, 1:])
        row = verify_logits[0, 0].astype(mx.float32)
        log_probs = row - mx.logsumexp(row)
        entropy = float(-mx.sum(mx.exp(log_probs) * log_probs).item())
        stats.entropy_per_cycle.append(entropy)

        stats.cycles += 1
        stats.accepted_specs += result.accepted_specs
        stats.offered_specs += result.offered_specs
        stats.full_blocks += int(result.full_block)
        committed = result.committed

        budget = max_tokens - len(generated)
        committed = committed[:budget]
        stop_at = _stop_index(committed, stop_ids)
        if stop_at is not None:
            committed = committed[: stop_at + 1]

        stats.committed_per_cycle.append(len(committed))
        stats.accepted_per_cycle.append(result.accepted_specs)
        trace.append(
            {
                "cycle": cycle,
                "k": block_size,
                "accepted": result.accepted_specs,
                "offered": result.offered_specs,
                "committed": len(committed),
                "full_block": bool(result.full_block),
                "verifier_entropy": round(entropy, 5),
                "draft_top1": [round(v, 5) for v in top1[1:]],
                "draft_margin": [round(v, 5) for v in margin[1:]],
            }
        )
        policy.update(result.accepted_specs, result.offered_specs, entropy)
        generated.extend(committed)
        stats.overhead_seconds += time.perf_counter() - overhead_start

        if stop_at is not None or len(generated) >= max_tokens:
            stage_start = time.perf_counter()
            transaction.commit_prefix(min(len(committed), int(verify_ids.shape[1])))
            mx.eval([c.state for c in caches])
            stats.commit_seconds += time.perf_counter() - stage_start
            stats.cycle_seconds += time.perf_counter() - cycle_start
            break

        stage_start = time.perf_counter()
        transaction.commit_prefix(len(committed))
        mx.eval([c.state for c in caches])
        stats.commit_seconds += time.perf_counter() - stage_start
        if cache_length(caches) != frontier + len(committed):
            raise RuntimeError(
                f"cache invariant violated after commit: "
                f"{cache_length(caches)} != {frontier + len(committed)}"
            )
        seed = committed[-1]
        stats.cycle_seconds += time.perf_counter() - cycle_start

    stats.forwards += txn_stats.full_forwards
    stats.replay_forwards = txn_stats.replay_forwards
    stats.forward_tokens += txn_stats.replay_tokens
    stats.transaction = txn_stats
    stats.wall_seconds = time.perf_counter() - start
    stats.tokens = len(generated)
    return generated, stats, {"trace": trace, "policy": policy.name}


# --------------------------------------------------------------- n-gram control


def _ngram_propose(history: list[int], block_size: int, order: int) -> list[int] | None:
    """Prompt-lookup speculation: find the most recent earlier occurrence of the last
    `order` tokens and propose whatever followed it.

    This is the cheapest possible drafter — no model, no parameters, a list scan — and
    it is exactly the control the brief's §17 asks for. Returns `None` when no match
    exists, in which case the caller drafts nothing and the cycle degenerates to a
    single AR step.
    """
    need = block_size - 1
    if need <= 0 or len(history) <= order:
        return None
    pattern = history[-order:]
    for start in range(len(history) - order - 1, -1, -1):
        if history[start : start + order] == pattern:
            follow = history[start + order : start + order + need]
            if len(follow) == need:
                return [int(t) for t in follow]
    return None


def ngram_greedy_generate(
    model,
    prompt_ids: list[int],
    max_tokens: int,
    block_size: int,
    order: int = 3,
    stop_ids: set[int] | None = None,
    transaction_mode: str = "snapshot",
) -> tuple[list[int], DecodeStats]:
    """The n-gram control, through the identical verify/commit path.

    The seed token still needs the frozen model's own next-token prediction, which the
    adapter path gets free from the draft pass's adapter-off seed row. Without a
    drafter there is no such pass, so the seed's token comes from the *previous*
    cycle's verify logits — which is exactly what `greedy_accept` already commits. The
    first cycle has no predecessor, so it runs one plain AR forward to obtain it.
    """
    stop_ids = stop_ids or set()
    stats = DecodeStats()
    stats.transaction_mode = transaction_mode
    txn_stats = TransactionStats()
    generated: list[int] = []
    start = time.perf_counter()

    caches = model.make_cache()
    prefill_start = time.perf_counter()
    if len(prompt_ids) > 1:
        logits = model.ar_logits(mx.array([prompt_ids[:-1]], dtype=mx.int32), cache=caches)
        mx.eval(logits)
        stats.forwards += 1
        stats.forward_tokens += len(prompt_ids) - 1
    stats.prefill_seconds = time.perf_counter() - prefill_start
    seed = prompt_ids[-1]
    cycle = 0

    while len(generated) < max_tokens:
        cycle_start = time.perf_counter()
        cycle += 1
        overhead_start = cycle_start
        frontier = cache_length(caches)
        history = list(prompt_ids) + generated
        guess = _ngram_propose(history, block_size, order)
        proposal = guess if guess is not None else []
        verify_ids = mx.array([[seed] + proposal], dtype=mx.int32)
        transaction = begin_transaction(model, caches, mode=transaction_mode, stats=txn_stats)
        stats.overhead_seconds += time.perf_counter() - overhead_start

        stage_start = time.perf_counter()
        verify_logits = transaction.run_block(verify_ids)
        mx.eval(verify_logits)
        stats.verify_seconds += time.perf_counter() - stage_start
        stats.forward_tokens += int(verify_ids.shape[1])

        overhead_start = time.perf_counter()
        result = greedy_accept([seed] + proposal, verify_logits[0])
        row = verify_logits[0, 0].astype(mx.float32)
        log_probs = row - mx.logsumexp(row)
        stats.entropy_per_cycle.append(float(-mx.sum(mx.exp(log_probs) * log_probs).item()))

        stats.cycles += 1
        stats.accepted_specs += result.accepted_specs
        stats.offered_specs += result.offered_specs
        stats.full_blocks += int(result.full_block)
        # The seed row is not a model prediction here, so drop it: `committed[0]` is
        # the seed itself, which is already in the context.
        committed = result.committed[1:]

        budget = max_tokens - len(generated)
        committed = committed[:budget]
        stop_at = _stop_index(committed, stop_ids)
        if stop_at is not None:
            committed = committed[: stop_at + 1]

        stats.committed_per_cycle.append(len(committed))
        stats.accepted_per_cycle.append(result.accepted_specs)
        generated.extend(committed)
        stats.overhead_seconds += time.perf_counter() - overhead_start

        stage_start = time.perf_counter()
        transaction.commit_prefix(len(committed))
        mx.eval([c.state for c in caches])
        stats.commit_seconds += time.perf_counter() - stage_start
        if cache_length(caches) != frontier + len(committed):
            raise RuntimeError(
                f"cache invariant violated after commit: "
                f"{cache_length(caches)} != {frontier + len(committed)}"
            )
        seed = committed[-1]
        stats.cycle_seconds += time.perf_counter() - cycle_start
        if stop_at is not None or len(generated) >= max_tokens:
            break

    stats.forwards += txn_stats.full_forwards
    stats.replay_forwards = txn_stats.replay_forwards
    stats.forward_tokens += txn_stats.replay_tokens
    stats.transaction = txn_stats
    stats.wall_seconds = time.perf_counter() - start
    stats.tokens = len(generated)
    return generated, stats
