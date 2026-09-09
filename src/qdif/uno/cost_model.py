"""An empirical cost model for the transactional Uno decoder.

Act IV-U2 removed the replay forward, so decode cost is now a clean function of the
accepted-prefix distribution. This module makes that function explicit, fits it from
*measured* per-stage timings, and inverts it to answer the question U3 actually needs:

    what accepted-prefix distribution is required to reach 1.10x / 1.25x / 1.5x AR?

Two things it deliberately does **not** do:

* It does not assume TPF maps linearly to wall-clock throughput. A Uno cycle pushes
  more token-positions through the model than an AR step, so a forward is not a
  forward; the per-stage times are measured and used directly.
* It does not replace measurement. `predicted_tokens_per_second` is checked against
  the measured value at every checkpoint, and a divergence is reported as a failure to
  understand the decoder rather than smoothed over.

## The cycle, in the snapshot transaction mode

Per cycle at block size `L` the decoder performs exactly two full forwards — a draft
over `L` positions and a verify over `L+1` — plus a commit that touches only cache
state. Committed tokens per cycle:

* every proposal accepted  ->  `L + 1` tokens (the clean token, `L-1` proposals, and
  the free lookahead)
* rejected at speculative slot `j` (0-based)  ->  `j + 2` tokens

so a cycle always commits at least 2 tokens and never stalls.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CycleCost:
    """Measured per-cycle stage costs, in milliseconds.

    ## v2: the overhead term

    Act IV-U3 modelled a cycle as `proposal + verify + commit` and over-predicted
    throughput at **every** checkpoint, by +3.9% to +10.2% (mean +7.7%). A residual
    that never changes sign is not noise; it is a cost the model does not have a term
    for. The missing work is the part of the cycle that is neither a forward nor a
    commit: snapshotting and restoring the caches, building the draft block, pulling
    the proposal back to the host, running the greedy accept test, and the entropy
    probe.

    `overhead_ms` is that term, and it is **measured**, not fitted. `decode.py` times
    those regions directly and reports them as `overhead_ms_per_cycle`. Fitting a
    correction factor to U3's residual would have produced a model that predicts U3
    perfectly and explains nothing; this one can be wrong, which is the point.

    `overhead_ms` defaults to 0.0 so that U3's stage timings still evaluate under the
    v1 model and its bias stays visible for comparison.
    """

    proposal_ms: float
    verify_ms: float
    commit_ms: float
    overhead_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.proposal_ms + self.verify_ms + self.commit_ms + self.overhead_ms

    def to_dict(self) -> dict:
        return {
            "proposal_ms": round(self.proposal_ms, 4),
            "verify_ms": round(self.verify_ms, 4),
            "commit_ms": round(self.commit_ms, 4),
            "overhead_ms": round(self.overhead_ms, 4),
            "total_ms": round(self.total_ms, 4),
        }


def committed_tokens(block_size: int, accepted_specs: int) -> int:
    """Tokens committed by one cycle that accepted `accepted_specs` proposals."""
    offered = block_size - 1
    if not 0 <= accepted_specs <= offered:
        raise ValueError(f"accepted_specs={accepted_specs} outside [0, {offered}]")
    if accepted_specs == offered:
        return block_size + 1
    return accepted_specs + 2


def expected_committed(block_size: int, distribution: dict[int, float]) -> float:
    """E[tokens per cycle] under a distribution over accepted-spec counts."""
    total = sum(distribution.values())
    if total <= 0:
        raise ValueError("acceptance distribution must have positive mass")
    return sum(
        weight * committed_tokens(block_size, accepted)
        for accepted, weight in distribution.items()
    ) / total


def geometric_distribution(block_size: int, per_slot_accept: float) -> dict[int, float]:
    """Accepted-spec distribution if each slot is accepted i.i.d. with probability p.

    A modelling convenience for the *inverse* question only ("what acceptance would be
    needed?"). Real acceptance is not i.i.d. across slots — later slots are harder, as
    the per-slot agreement curves show — so measured distributions are always preferred
    when available. Used for threshold estimation, never for reporting a result.
    """
    if not 0.0 <= per_slot_accept <= 1.0:
        raise ValueError("per_slot_accept must be in [0, 1]")
    offered = block_size - 1
    distribution: dict[int, float] = {}
    for accepted in range(offered):
        distribution[accepted] = (per_slot_accept**accepted) * (1.0 - per_slot_accept)
    distribution[offered] = per_slot_accept**offered
    return distribution


def predicted_tokens_per_second(
    block_size: int, distribution: dict[int, float], cost: CycleCost
) -> float:
    """Steady-state E[tokens] / E[seconds] for one cycle. Excludes prefill."""
    tokens = expected_committed(block_size, distribution)
    return tokens / (cost.total_ms / 1000.0)


def predicted_end_to_end_tokens_per_second(
    block_size: int,
    distribution: dict[int, float],
    cost: CycleCost,
    prefill_ms: float,
    tokens: int,
) -> float:
    """Throughput as the benchmark actually reports it: prefill included.

    The second half of the v1 bias. `tokens_per_second` in `DecodeStats` is
    `tokens / wall_seconds`, and `wall_seconds` contains the prompt prefill — one full
    forward over the whole prompt, before any cycle runs. The v1 model predicted a
    pure steady-state rate and was compared against that end-to-end number, so it
    over-predicted by roughly the prefill's share of the run: at 48 generated tokens
    and a ~15-token prompt that is a few percent, in the same direction, every time.

    Separating the two is what makes the residual diagnostic. If the steady-state
    prediction is accurate and only the end-to-end one is off, the cycle model is fine
    and the harness is amortising a fixed cost; if both are off, the cycle model is
    missing work. Both are reported.
    """
    per_cycle = expected_committed(block_size, distribution)
    cycles = tokens / per_cycle
    total_ms = prefill_ms + cycles * cost.total_ms
    return tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0


def predicted_tpf(block_size: int, distribution: dict[int, float]) -> float:
    """Tokens per full forward. Snapshot mode is exactly 2 forwards per cycle."""
    return expected_committed(block_size, distribution) / 2.0


def distribution_from_counts(counts: dict[int, int]) -> dict[int, float]:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("empty acceptance histogram")
    return {accepted: n / total for accepted, n in counts.items()}


def required_per_slot_accept(
    block_size: int,
    cost: CycleCost,
    ar_tokens_per_second: float,
    target_speedup: float,
    tolerance: float = 1e-5,
) -> float | None:
    """Smallest i.i.d. per-slot acceptance reaching `target_speedup` x AR.

    Returns `None` when the target is unreachable at this block size even with every
    proposal accepted — which is itself a useful answer, and is reported rather than
    silently clamped.
    """
    target = target_speedup * ar_tokens_per_second
    best = predicted_tokens_per_second(
        block_size, geometric_distribution(block_size, 1.0), cost
    )
    if best < target:
        return None

    low, high = 0.0, 1.0
    while high - low > tolerance:
        mid = 0.5 * (low + high)
        rate = predicted_tokens_per_second(
            block_size, geometric_distribution(block_size, mid), cost
        )
        if rate >= target:
            high = mid
        else:
            low = mid
    return high


def speedup_table(
    block_size: int,
    cost: CycleCost,
    ar_tokens_per_second: float,
    targets: tuple[float, ...] = (1.10, 1.20, 1.30, 1.50, 2.00),
) -> list[dict]:
    """What acceptance each speedup target demands, or that it is unreachable."""
    rows = []
    for target in targets:
        per_slot = required_per_slot_accept(
            block_size, cost, ar_tokens_per_second, target
        )
        if per_slot is None:
            rows.append({
                "target_speedup": target,
                "reachable": False,
                "required_per_slot_accept": None,
                "required_mean_committed": None,
                "required_full_block_rate": None,
                "required_tpf": None,
            })
            continue
        distribution = geometric_distribution(block_size, per_slot)
        rows.append({
            "target_speedup": target,
            "reachable": True,
            "required_per_slot_accept": round(per_slot, 4),
            "required_mean_committed": round(
                expected_committed(block_size, distribution), 4
            ),
            "required_full_block_rate": round(distribution[block_size - 1], 4),
            "required_tpf": round(predicted_tpf(block_size, distribution), 4),
        })
    return rows


# ----------------------------------------------------------------- U4 analyses


def survival_curve(block_size: int, distribution: dict[int, float]) -> dict:
    """`P(accepted >= j)` — the horizon survival curve (Act IV-U4 §21).

    Two curves, because Act IV-U4 §19 is right that the naming matters:

    * `accepted_specs` — `P(a >= j)` over speculative slots, `j = 1 … L-1`.
    * `committed_tokens` — `P(c >= j)` over tokens actually committed, `j = 1 … L+1`.

    They are one substitution apart. A cycle that accepts `a` speculative tokens
    commits `a + 2` (the clean token, the accepted run, and the free lookahead) — and
    that holds at `a = L-1` too, where the full-block rule gives `L + 1 = (L-1) + 2`.
    So `E[c] = E[a] + 2` exactly, and since `E[X] = Σ_{j>=1} P(X >= j)` for a
    non-negative integer variable, the area under either curve *is* the corresponding
    expectation rather than merely relating to it.

    The consequence used in `marginal_slot_value`: the marginal contribution of
    speculative slot `j` to committed tokens per cycle is exactly `P(a >= j)`.
    """
    total = sum(distribution.values())
    if total <= 0:
        raise ValueError("acceptance distribution must have positive mass")
    offered = block_size - 1
    accepted = {
        j: sum(w for a, w in distribution.items() if a >= j) / total
        for j in range(1, offered + 1)
    }
    committed = {
        j: sum(
            w
            for a, w in distribution.items()
            if committed_tokens(block_size, a) >= j
        )
        / total
        for j in range(1, block_size + 2)
    }
    mean_accepted = sum(accepted.values())
    return {
        "block_size": block_size,
        "accepted_specs": {str(j): round(p, 5) for j, p in accepted.items()},
        "committed_tokens": {str(j): round(p, 5) for j, p in committed.items()},
        "mean_accepted_specs": round(mean_accepted, 5),
        "mean_committed_tokens": round(mean_accepted + 2.0, 5),
        "area_check": round(
            abs(mean_accepted + 2.0 - expected_committed(block_size, distribution)), 9
        ),
    }


def marginal_slot_value(
    block_size: int,
    distribution: dict[int, float],
    cost: CycleCost,
    ms_per_extra_slot: float,
) -> list[dict]:
    """When does one more draft position stop paying for itself? (§22)

    Extending a block from `L` to `L+1` adds `P(a >= L)` expected committed tokens and
    `ms_per_extra_slot` milliseconds. Throughput improves iff

        P(a >= L)  >  E[committed at L] * (delta_ms / total_ms at L)

    i.e. the extra slot must contribute a larger *fraction* of the tokens than it does
    of the time. `ms_per_extra_slot` is estimated by regressing measured cycle cost on
    block size across the K we actually ran — it is not a guess, and it is reported.

    This is measurement, not routing. Act IV-U4 §23 keeps dynamic K out of scope; the
    point here is to locate the frontier, not to schedule around it.
    """
    survival = survival_curve(block_size, distribution)["accepted_specs"]
    total = cost.total_ms
    rows = []
    running = 2.0
    for slot in range(1, block_size):
        gain = survival[str(slot)]
        # Cost/tokens at the block size for which this slot is the *last* one.
        cost_at = total - (block_size - slot) * ms_per_extra_slot
        break_even = running * (ms_per_extra_slot / cost_at) if cost_at > 0 else float("inf")
        rows.append({
            "slot": slot,
            "marginal_tokens": round(gain, 5),
            "break_even_tokens": round(break_even, 5),
            "pays_for_itself": bool(gain > break_even),
            "cycle_ms_at_this_block": round(cost_at, 4),
        })
        running += gain
    return rows


def slot_cost_regression(samples: list[tuple[int, float]]) -> dict:
    """Least-squares `cycle_ms = intercept + slope * block_size` over measured points.

    `slope` is the millisecond price of one more draft position — the quantity
    `marginal_slot_value` needs and the one U3's model had no way to see, because it
    only ever looked at one block size at a time.
    """
    if len(samples) < 2:
        raise ValueError("need at least two block sizes to estimate a slope")
    n = len(samples)
    mean_x = sum(x for x, _ in samples) / n
    mean_y = sum(y for _, y in samples) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in samples)
    if sxx == 0:
        raise ValueError("all samples share one block size")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in samples)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    predicted = [intercept + slope * x for x, _ in samples]
    ss_res = sum((y - p) ** 2 for (_, y), p in zip(samples, predicted))
    ss_tot = sum((y - mean_y) ** 2 for _, y in samples)
    return {
        "intercept_ms": round(intercept, 4),
        "ms_per_slot": round(slope, 4),
        "r_squared": round(1.0 - ss_res / ss_tot, 5) if ss_tot > 0 else None,
        "samples": [[x, round(y, 4)] for x, y in samples],
    }


def ceiling(block_size: int, cost: CycleCost, ar_tokens_per_second: float) -> dict:
    """The best this block size can do if every proposal is always accepted."""
    distribution = geometric_distribution(block_size, 1.0)
    rate = predicted_tokens_per_second(block_size, distribution, cost)
    return {
        "block_size": block_size,
        "max_tokens_per_cycle": block_size + 1,
        "max_tpf": (block_size + 1) / 2.0,
        "max_tokens_per_second": round(rate, 3),
        "max_speedup_vs_ar": round(rate / ar_tokens_per_second, 4),
    }
