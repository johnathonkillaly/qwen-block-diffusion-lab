"""Flat token windows, and the held-out decoding prompt suite.

Act IV-U needs plainer data than Acts I-IV-N: no prefix/canvas split, just contiguous
windows of width `W`, because the block is always the suffix (`teacher.py`).

Train and evaluation are separated by *dataset split*, not by slicing one stream, so
there is no way for a training window to leak into validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


def pack_windows(
    texts, tokenizer, width: int, max_windows: int, eos_id: int | None = None
) -> np.ndarray:
    """`[N, width]` contiguous token windows from a packed stream."""
    if width < 2:
        raise ValueError("width must be >= 2")
    need = max_windows * width + width
    chunks: list[np.ndarray] = []
    total = 0
    for text in texts:
        if not text or not text.strip():
            continue
        ids = np.asarray(tokenizer.encode(text), dtype=np.int64)
        if ids.size == 0:
            continue
        chunks.append(ids)
        total += ids.size
        if eos_id is not None:
            chunks.append(np.array([eos_id], dtype=np.int64))
            total += 1
        if total >= need:
            break
    if not chunks:
        raise ValueError("no text to pack")
    stream = np.concatenate(chunks)
    count = min(stream.size // width, max_windows)
    if count == 0:
        raise ValueError(
            f"packed stream has {stream.size} tokens, fewer than one window of {width}"
        )
    return stream[: count * width].reshape(count, width)


def load_wikitext(split: str, min_chars: int = 200):
    """Texts from the locally cached WikiText-103 — the corpus Acts II/III used."""
    from datasets import load_dataset

    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    return (row["text"] for row in dataset if len(row["text"].strip()) >= min_chars)


@dataclass
class WindowSource:
    """A reproducible, reshuffling iterator over a fixed pool of windows."""

    windows: np.ndarray
    batch_size: int
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._order = self._rng.permutation(len(self.windows))
        self._cursor = 0
        self._draws = 0

    def __iter__(self):
        return self

    def __next__(self) -> mx.array:
        if self._cursor + self.batch_size > len(self._order):
            self._order = self._rng.permutation(len(self.windows))
            self._cursor = 0
        picked = self._order[self._cursor : self._cursor + self.batch_size]
        self._cursor += self.batch_size
        self._draws += 1
        return mx.array(self.windows[picked].astype(np.int32))

    @property
    def num_windows(self) -> int:
        return len(self.windows)

    @property
    def num_tokens(self) -> int:
        return int(self.windows.size)

    # ------------------------------------------------------------ resumability

    @property
    def draws(self) -> int:
        """How many batches have been handed out. The data-position state."""
        return self._draws

    def fast_forward(self, draws: int) -> "WindowSource":
        """Advance by `draws` batches without materialising them.

        This is how Act IV-U3's checkpoints are made resumable after the fact. They
        recorded the training step but not the data cursor -- and because the source is
        a pure function of `(seed, number of draws)`, replaying `step` draws recovers
        the exact position rather than approximating it. Reshuffles are reproduced
        because the reshuffle happens inside this loop, on the same `_rng`.
        """
        if draws < 0:
            raise ValueError(f"cannot fast-forward by {draws} draws")
        for _ in range(draws):
            if self._cursor + self.batch_size > len(self._order):
                self._order = self._rng.permutation(len(self.windows))
                self._cursor = 0
            self._cursor += self.batch_size
            self._draws += 1
        return self

    def state_dict(self) -> dict:
        """Everything needed to resume this iterator exactly."""
        return {
            "seed": int(self.seed),
            "batch_size": int(self.batch_size),
            "num_windows": int(len(self.windows)),
            "draws": int(self._draws),
            "cursor": int(self._cursor),
            "order_digest": int(self._order[:8].sum()) if len(self._order) else 0,
        }

    def load_state_dict(self, state: dict) -> "WindowSource":
        """Restore by replaying `draws` batches from a fresh iterator.

        Replay rather than pickling numpy's bit generator: it is verifiable (the cursor
        and the order digest must both land where they were saved) and it survives a
        numpy version change.
        """
        if int(state["num_windows"]) != len(self.windows):
            raise ValueError(
                f"checkpoint was taken over {state['num_windows']} windows, "
                f"this source has {len(self.windows)}"
            )
        if int(state["batch_size"]) != int(self.batch_size):
            raise ValueError(
                f"checkpoint batch size {state['batch_size']} != {self.batch_size}"
            )
        self.__post_init__()
        self.fast_forward(int(state["draws"]))
        restored = self.state_dict()
        if restored["cursor"] != int(state["cursor"]) or (
            restored["order_digest"] != int(state["order_digest"])
        ):
            raise ValueError(
                f"data-position restore did not land where it was saved: "
                f"{restored} != {state}"
            )
        return self


def build_sources(
    tokenizer,
    width: int,
    batch_size: int,
    train_windows: int,
    val_windows: int,
    seed: int = 0,
    eos_id: int | None = None,
) -> tuple[WindowSource, WindowSource]:
    train = pack_windows(
        load_wikitext("train"), tokenizer, width, train_windows, eos_id=eos_id
    )
    val = pack_windows(
        load_wikitext("validation"), tokenizer, width, val_windows, eos_id=eos_id
    )
    return (
        WindowSource(train, batch_size, seed=seed),
        WindowSource(val, batch_size, seed=seed + 1),
    )


#: Hand-written held-out decoding prompts, five domains.
#:
#: These exist because the quantitative corpus is WikiText only, and the brief asks
#: whether the parallel horizon depends on how constrained the continuation is (§26).
#: Code and structured text should be the easy cases -- much of their next-token
#: distribution is syntax -- and open-ended prose the hard one. Nothing here is
#: trained on.
PROMPT_SUITE: dict[str, list[str]] = {
    "prose": [
        "The history of the printing press begins in the fifteenth century, when",
        "He looked across the room and",
        "In the years following the war, the city",
    ],
    "factual": [
        "The capital of France is",
        "Water boils at a temperature of",
        "The largest planet in the solar system is",
    ],
    "code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "import numpy as np\n\ndef normalize(x):\n    \"\"\"Scale x to zero mean and unit variance.\"\"\"\n    return",
        "for i in range(10):\n    print(",
    ],
    "math": [
        "To solve the equation 2x + 5 = 13, we first subtract 5 from both sides, giving",
        "The derivative of x^3 with respect to x is",
        "If a triangle has sides of length 3, 4 and 5, then",
    ],
    "structured": [
        '{"name": "Ada Lovelace", "born": 1815, "known_for":',
        "| Country | Capital |\n| --- | --- |\n| France | Paris |\n| Japan |",
        "Ingredients:\n- 2 cups flour\n- 1 cup sugar\n-",
    ],
}


def prompt_suite_tokens(tokenizer) -> list[tuple[str, str, list[int]]]:
    """`(domain, text, token_ids)` for every prompt in the suite."""
    out = []
    for domain, prompts in PROMPT_SUITE.items():
        for text in prompts:
            out.append((domain, text, list(tokenizer.encode(text))))
    return out
