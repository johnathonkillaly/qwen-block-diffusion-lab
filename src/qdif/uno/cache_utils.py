"""Snapshot and restore for mlx_lm's hybrid cache list.

Speculative decoding needs to *undo* a forward: the draft block and any rejected
proposals must leave no trace in the cache. For a pure attention stack that is a
one-line offset rewind. Qwen3.5 is 24/32 Gated DeltaNet, whose `ArraysCache` holds a
running convolution state and a recurrent state -- neither of which can be truncated,
because a recurrence has no notion of "the state 3 steps ago".

Snapshotting works anyway, and cheaply, because those states are small and
fixed-size (they do not grow with sequence length): saving them is a handful of
references to immutable MLX arrays, not a copy of the KV history.

What is *not* recoverable is an intermediate state from inside a multi-token forward.
That is why `decode.py` replays accepted tokens after a partial acceptance instead of
rewinding into the middle of the verify pass -- see the note there.
"""

from __future__ import annotations

from typing import Any


def snapshot_caches(caches: list[Any]) -> list[dict]:
    """Capture enough state to rewind every cache to this exact point."""
    snapshot = []
    for cache in caches:
        entry: dict[str, Any] = {}
        if hasattr(cache, "offset"):
            entry["offset"] = cache.offset
        if hasattr(cache, "cache") and isinstance(getattr(cache, "cache"), list):
            # ArraysCache: the list holds MLX arrays, which are values, so copying the
            # list is a genuine snapshot rather than an alias.
            entry["arrays"] = list(cache.cache)
        snapshot.append(entry)
    return snapshot


def restore_caches(caches: list[Any], snapshot: list[dict]) -> None:
    """Rewind in place. Must be paired with a `snapshot_caches` from the same list."""
    if len(caches) != len(snapshot):
        raise ValueError(
            f"cache/snapshot length mismatch: {len(caches)} vs {len(snapshot)}"
        )
    for cache, entry in zip(caches, snapshot):
        if "offset" in entry:
            cache.offset = entry["offset"]
        if "arrays" in entry:
            cache.cache = list(entry["arrays"])


def cache_length(caches: list[Any]) -> int:
    """The attention caches' common offset — the number of tokens the cache holds.

    DeltaNet caches carry no length of their own, so the KV caches are the only
    readable clock. They are advanced by the same forwards, so this is exact.
    """
    for cache in caches:
        if hasattr(cache, "offset") and hasattr(cache, "keys"):
            return int(cache.offset)
    return 0
