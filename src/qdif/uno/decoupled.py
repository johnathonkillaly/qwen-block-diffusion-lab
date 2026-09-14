"""Act IV-U6: draft width decoupled from verify width.

The coupled decoder (`decode.uno_greedy_generate`) drafts K positions and verifies the
same K. This module keeps the drafter and verifier exactly as they are and changes only
how many draft tokens each verify forward sees:

* **truncate** `T(D, V)`: draft `D`, verify `[seed, p₁ … p_V]`, discard the rest;
* **staged** `S(D, V)`: draft `D` once, verify `V` at a time on the committed cache, and
  continue only while every previous slot was accepted **and** the target's lookahead
  equals the next draft token.

Correctness is structural, not argued per variant. Every committed token is the target's
own argmax for the forward that verified it (`verifier.greedy_accept`), and every stage is
its own snapshot transaction that keeps only a verified prefix. A rejected token cannot
enter target state. Stage ≥ 2 is the same verify path the n-gram control uses, with
draft tokens as the guesses.

With `D == V` both variants are the incumbent decoder, token for token and cycle for
cycle; `tests/test_uno_decoupled.py` asserts it. `decode.py` is not modified.
"""

from __future__ import annotations

import time
from typing import Callable

import mlx.core as mx

from .cache_utils import cache_length, restore_caches, snapshot_caches
from .decode import DecodeStats, _cycle_key, _stop_index
from .speculative import _draft_proposal
from .transaction import TransactionStats, begin_transaction
from .verifier import greedy_accept

#: `proposer(context_tokens, width) -> list[int]` replaces the diffusion drafter. It exists
#: for tests: an oracle proposer exercises every stage, and an adversarial one exercises
#: none. Benchmarks never pass one.
Proposer = Callable[[list[int], int], list[int]]


def decoupled_greedy_generate(
    model,
    prompt_ids: list[int],
    max_tokens: int,
    draft_width: int,
    verify_width: int,
    staged: bool = False,
    noise_mode: str = "random_uniform",
    stop_ids: set[int] | None = None,
    transaction_mode: str = "snapshot",
    noise_stream_seed: int | None = None,
    proposer: Proposer | None = None,
) -> tuple[list[int], DecodeStats, dict]:
    """Greedy draft→verify decoding with `draft_width` ≥ `verify_width`.

    Returns `(tokens, stats, extra)`. `extra["trace"]` holds one record per cycle: the
    draft width, cycle latency, and for each stage its verify width, offered and accepted
    slots, committed tokens, whether the block was fully accepted, whether the lookahead
    matched the draft, and its verify latency.

    Accounting mirrors `decode.uno_greedy_generate`: `stats.forwards` counts the prefill,
    one draft forward per cycle and every verify forward (from the shared transaction
    stats). The verifier-entropy probe runs once per cycle, on stage 1, so the new decoder
    does exactly the incumbent's work when `D == V`.
    """
    if draft_width < 1 or verify_width < 1:
        raise ValueError(f"widths must be >= 1, got draft={draft_width} verify={verify_width}")
    if verify_width > draft_width:
        raise ValueError(f"verify_width {verify_width} exceeds draft_width {draft_width}")
    stop_ids = stop_ids or set()
    stats = DecodeStats()
    stats.transaction_mode = transaction_mode
    txn_stats = TransactionStats()
    generated: list[int] = []
    trace: list[dict] = []
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
    finished = False

    while len(generated) < max_tokens and not finished:
        cycle_start = time.perf_counter()
        cycle += 1
        overhead_start = cycle_start
        snapshot = snapshot_caches(caches)
        key = None if noise_stream_seed is None else _cycle_key(noise_stream_seed, cycle)
        stats.overhead_seconds += time.perf_counter() - overhead_start

        # ---- one draft per cycle ------------------------------------------------------
        if proposer is None:
            proposal, _, _ = _draft_proposal(
                model, seed, draft_width, caches, snapshot, noise_mode, key, 1, stats
            )
        else:
            stage_start = time.perf_counter()
            proposal = [int(t) for t in proposer(list(prompt_ids) + generated, draft_width)]
            stats.proposal_seconds += time.perf_counter() - stage_start
            if len(proposal) != draft_width:
                raise ValueError(f"proposer returned {len(proposal)} tokens, need {draft_width}")
        overhead_start = time.perf_counter()
        restore_caches(caches, snapshot)
        stats.overhead_seconds += time.perf_counter() - overhead_start

        record = {"cycle": cycle, "draft_width": draft_width, "stages": []}
        cycle_committed = cycle_accepted = 0
        block = proposal[:verify_width]
        next_index = verify_width  # proposal index of the first draft token not yet presented
        first_stage = True

        while True:
            overhead_start = time.perf_counter()
            frontier = cache_length(caches)
            verify_ids = mx.array([[seed] + block], dtype=mx.int32)
            transaction = begin_transaction(model, caches, mode=transaction_mode, stats=txn_stats)
            stats.overhead_seconds += time.perf_counter() - overhead_start

            stage_start = time.perf_counter()
            verify_logits = transaction.run_block(verify_ids)
            mx.eval(verify_logits)
            verify_elapsed = time.perf_counter() - stage_start
            stats.verify_seconds += verify_elapsed
            stats.forward_tokens += int(verify_ids.shape[1])

            overhead_start = time.perf_counter()
            if first_stage:
                # Identical to the incumbent: row 0 is the target's own next token, which
                # the adapter-off seed row already proposed as p₁.
                result = greedy_accept(block, verify_logits[0, 1:])
                committed = result.committed
                row = verify_logits[0, 0].astype(mx.float32)
                log_probs = row - mx.logsumexp(row)
                stats.entropy_per_cycle.append(
                    float(-mx.sum(mx.exp(log_probs) * log_probs).item())
                )
            else:
                # The seed is already committed, so its row is not a proposal; drop it.
                result = greedy_accept([seed] + block, verify_logits[0])
                committed = result.committed[1:]

            stats.accepted_specs += result.accepted_specs
            stats.offered_specs += result.offered_specs
            stats.full_blocks += int(result.full_block)

            committed = committed[: max_tokens - len(generated)]
            stop_at = _stop_index(committed, stop_ids)
            if stop_at is not None:
                committed = committed[: stop_at + 1]
            generated.extend(committed)
            cycle_committed += len(committed)
            cycle_accepted += result.accepted_specs
            finished = stop_at is not None or len(generated) >= max_tokens
            stats.overhead_seconds += time.perf_counter() - overhead_start

            stage_start = time.perf_counter()
            transaction.commit_prefix(len(committed))
            mx.eval([c.state for c in caches])
            stats.commit_seconds += time.perf_counter() - stage_start

            stage_record = {
                "stage": len(record["stages"]) + 1,
                "verify_tokens": int(verify_ids.shape[1]),
                "offered": result.offered_specs,
                "accepted": result.accepted_specs,
                "committed": len(committed),
                "full_block": bool(result.full_block),
                "verify_ms": round(1000 * verify_elapsed, 4),
                "continues": False,
                "lookahead_matched_draft": False,
            }
            record["stages"].append(stage_record)
            if finished:
                break
            if cache_length(caches) != frontier + len(committed):
                raise RuntimeError(
                    f"cache invariant violated after stage commit: "
                    f"{cache_length(caches)} != {frontier + len(committed)}"
                )
            seed = committed[-1]

            # ---- bounded continuation ------------------------------------------------
            matched = next_index < draft_width and seed == proposal[next_index]
            stage_record["lookahead_matched_draft"] = bool(result.full_block and matched)
            if not (staged and result.full_block and matched):
                break
            guesses = proposal[next_index + 1 : next_index + 1 + verify_width]
            if not guesses:
                break
            stage_record["continues"] = True
            block = guesses
            next_index = next_index + 1 + len(guesses)
            first_stage = False

        stats.cycles += 1
        stats.committed_per_cycle.append(cycle_committed)
        stats.accepted_per_cycle.append(cycle_accepted)
        elapsed = time.perf_counter() - cycle_start
        stats.cycle_seconds += elapsed
        record["cycle_ms"] = round(1000 * elapsed, 4)
        record["committed"] = cycle_committed
        record["accepted"] = cycle_accepted
        record["unused_draft_tokens"] = draft_width - min(next_index, draft_width)
        trace.append(record)

    stats.forwards += txn_stats.full_forwards
    stats.replay_forwards = txn_stats.replay_forwards
    stats.forward_tokens += txn_stats.replay_tokens
    stats.transaction = txn_stats
    stats.wall_seconds = time.perf_counter() - start
    stats.tokens = len(generated)

    stage_counts: dict[int, int] = {}
    for r in trace:
        stage_counts[len(r["stages"])] = stage_counts.get(len(r["stages"]), 0) + 1
    return generated, stats, {
        "trace": trace,
        "draft_width": draft_width,
        "verify_width": verify_width,
        "staged": staged,
        "stages_per_cycle_histogram": {str(k): v for k, v in sorted(stage_counts.items())},
        "verify_forwards": txn_stats.full_forwards,
    }
