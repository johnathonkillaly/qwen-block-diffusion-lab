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
    """Measured per-cycle stage costs, in milliseconds."""

    proposal_ms: float
    verify_ms: float
    commit_ms: float

    @property
    def total_ms(self) -> float:
        return self.proposal_ms + self.verify_ms + self.commit_ms

    def to_dict(self) -> dict:
        return {
            "proposal_ms": round(self.proposal_ms, 4),
            "verify_ms": round(self.verify_ms, 4),
            "commit_ms": round(self.commit_ms, 4),
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
    """E[tokens] / E[seconds] for one cycle."""
    tokens = expected_committed(block_size, distribution)
    return tokens / (cost.total_ms / 1000.0)


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
