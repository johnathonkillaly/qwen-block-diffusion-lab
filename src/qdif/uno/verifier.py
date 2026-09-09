"""Block verification against the frozen AR model.

The whole speedup rests on one fact: a causal forward over `context + [t₀ … t_{L-1}]`
exposes, in a *single* pass, the model's next-token distribution after every prefix of
that block. So verifying L speculative tokens costs one forward, not L.

Indexing is the part that is easy to get wrong, so it is spelled out once here and
then tested against a deliberately slow reference implementation
(`sequential_reference_accept_length`) that really does perform L separate AR calls.

    block      = [ t₀ , t₁ , … , t_{L-1} ]
    logits[i]  = p(· | context, t₀ … t_i)
    so logits[i] predicts t_{i+1}, for i = 0 … L-2
    and logits[L-1] predicts one token *beyond* the block  → the lookahead

`t₀` is the "clean" token: it comes from the draft pass's seed row, which runs with
the adapter switched **off**, so it is already the frozen model's own next token. It
needs no verification — but it does need **committing**. Dropping it shifts the whole
output by one; that bug survived the tiny-model tests because a degenerate model emits
a constant sequence in which an off-by-one is invisible, and was caught only on the
real 4B backbone. `t₁ … t_{L-1}` are the speculative proposals and are verified.

The commit rule is copied from `nano_vllm_uno/engine/two_pass_decoding.py`
(`_build_committed_payload`) and has two details worth not losing:

  * a rejected position is **replaced** by the verifier's own token rather than
    discarded, so a cycle that accepts nothing still commits one correct token and
    never stalls;
  * if every proposal is accepted, the lookahead token is committed too, so a
    fully-accepted block of L yields **L+1** tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx


@dataclass
class VerifyResult:
    """Outcome of verifying one proposal block for one sequence."""

    committed: list[int]
    #: How many *speculative* proposals (t₁ … t_{L-1}) matched the verifier.
    accepted_specs: int
    #: Total speculative slots offered, i.e. L-1.
    offered_specs: int
    #: True when the whole block matched and the lookahead token was taken.
    full_block: bool
    #: The verifier's own argmax at each speculative slot; useful for diagnostics.
    target_tokens: list[int] = field(default_factory=list)

    @property
    def num_committed(self) -> int:
        return len(self.committed)


def greedy_accept(
    block: list[int] | mx.array,
    verify_logits: mx.array,
) -> VerifyResult:
    """Greedy (temperature-0) acceptance for a single sequence.

    `block` is `[t₀ … t_{L-1}]` including the seed. `verify_logits` is `[L, V]`, the
    output of one causal forward over that block on top of the committed context.

    Note t₀ is *not* verified: at decode time it is produced by the seed row with the
    adapter switched off, so it is already the frozen model's own token.
    """
    tokens = [int(t) for t in (block.tolist() if isinstance(block, mx.array) else block)]
    length = len(tokens)
    if verify_logits.ndim != 2 or verify_logits.shape[0] != length:
        raise ValueError(
            f"verify_logits must be [L, V] with L={length}, got {tuple(verify_logits.shape)}"
        )

    targets = [int(t) for t in mx.argmax(verify_logits, axis=-1).tolist()]
    num_specs = length - 1
    spec_targets = targets[:num_specs]

    accepted = 0
    while accepted < num_specs and tokens[accepted + 1] == spec_targets[accepted]:
        accepted += 1

    if accepted == num_specs:
        # Everything matched. Commit the clean token, every proposal, and the
        # lookahead the verify pass produced for free. L+1 tokens from one cycle.
        return VerifyResult(
            committed=tokens + [targets[num_specs]],
            accepted_specs=accepted,
            offered_specs=num_specs,
            full_block=True,
            target_tokens=spec_targets,
        )

    # Rejected at `accepted`: keep the clean token and the accepted prefix, then
    # substitute the verifier's own token for the mismatch. That token is correct by
    # construction, so a cycle that accepts nothing still commits two tokens and the
    # decoder can never stall.
    return VerifyResult(
        committed=tokens[: accepted + 1] + [spec_targets[accepted]],
        accepted_specs=accepted,
        offered_specs=num_specs,
        full_block=False,
        target_tokens=spec_targets,
    )


def greedy_accept_batch(
    blocks: mx.array, verify_logits: mx.array
) -> list[VerifyResult]:
    """`blocks` is `[B, L]`, `verify_logits` is `[B, L, V]`."""
    if verify_logits.shape[:2] != blocks.shape:
        raise ValueError(
            f"shape mismatch: blocks {tuple(blocks.shape)} vs "
            f"logits {tuple(verify_logits.shape[:2])}"
        )
    return [
        greedy_accept(blocks[row], verify_logits[row]) for row in range(blocks.shape[0])
    ]


def sequential_reference_accept_length(
    model,
    context_ids: mx.array,
    block: list[int],
) -> int:
    """Deliberately slow ground truth: one full AR call per speculative position.

    Used only by `tests/test_uno_verifier.py` to prove the single-forward verifier
    agrees with the obvious-but-expensive implementation. If these two ever disagree,
    the fast path is wrong and every acceptance number in the results is void.
    """
    if context_ids.ndim != 1:
        raise ValueError(f"context_ids must be 1-D, got {context_ids.shape}")
    accepted = 0
    for slot in range(len(block) - 1):
        prefix = mx.concatenate(
            [context_ids, mx.array(block[: slot + 1], dtype=context_ids.dtype)]
        )
        logits = model.ar_logits(prefix[None])
        predicted = int(mx.argmax(logits[0, -1]).item())
        if predicted != block[slot + 1]:
            break
        accepted += 1
    return accepted
