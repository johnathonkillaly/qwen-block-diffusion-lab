"""Dataset interface: every example exposes a clean prefix and a clean target canvas.

    CanvasExample(prefix_ids=[P], canvas_ids=[C], text=...)

The canvas is what the diffusion process corrupts; the prefix is the causal
conditioning context and is never corrupted. Corruption is *not* part of the
dataset -- see `qdif.data.collator`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from .dev_corpus import DEV_TEXTS


def encode(tokenizer, text: str) -> np.ndarray:
    """Tokenize to a numpy int64 array, for any of the tokenizer flavours in play.

    The torch backend uses a Hugging Face fast tokenizer; the MLX backend gets an
    `mlx_lm` `TokenizerWrapper`, which is not callable and takes different kwargs.
    Both backends must produce *identical* token ids, or a v0.1/v0.2 comparison is
    comparing datasets rather than models -- hence one shared entry point.
    """
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    if hasattr(ids, "ids"):  # raw `tokenizers` Encoding
        ids = ids.ids
    return np.asarray(ids, dtype=np.int64).reshape(-1)


def decode(tokenizer, ids) -> str:
    return tokenizer.decode([int(i) for i in np.asarray(ids).reshape(-1)])


@dataclass
class CanvasExample:
    prefix_ids: np.ndarray  # [P] int64
    canvas_ids: np.ndarray  # [C] int64
    text: str = ""

    def decoded(self, tokenizer) -> tuple[str, str]:
        return decode(tokenizer, self.prefix_ids), decode(tokenizer, self.canvas_ids)


class CanvasDataset:
    """Turns a list of texts into fixed-size (prefix, canvas) windows."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        prefix_length: int,
        canvas_length: int,
        max_examples: int | None = None,
        stride: int | None = None,
    ):
        if prefix_length < 0 or canvas_length < 1:
            raise ValueError("need prefix_length >= 0 and canvas_length >= 1")
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length
        self.canvas_length = canvas_length
        window = prefix_length + canvas_length
        stride = stride or window

        self.examples: list[CanvasExample] = []
        for text in texts:
            ids = encode(tokenizer, text)
            for start in range(0, max(len(ids) - window + 1, 0), stride):
                chunk = ids[start : start + window]
                self.examples.append(
                    CanvasExample(
                        prefix_ids=chunk[:prefix_length].copy(),
                        canvas_ids=chunk[prefix_length:].copy(),
                        text=decode(tokenizer, chunk),
                    )
                )
                if max_examples and len(self.examples) >= max_examples:
                    break
            if max_examples and len(self.examples) >= max_examples:
                break

        if not self.examples:
            raise ValueError(
                f"no examples produced: every text was shorter than "
                f"prefix_length + canvas_length = {window} tokens"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> CanvasExample:
        return self.examples[i]

    def __iter__(self) -> Iterator[CanvasExample]:
        return iter(self.examples)


def load_texts(cfg, split: str | None = None) -> list[str]:
    """Raw texts for a split. `hf_dataset` may be `repo` or `repo:config`."""
    if cfg.source == "dev":
        return list(DEV_TEXTS)
    if cfg.source != "hf":
        raise ValueError(f"unknown data.source {cfg.source!r}, expected 'dev' or 'hf'")
    if not cfg.hf_dataset:
        raise ValueError("data.source='hf' requires data.hf_dataset")

    from datasets import load_dataset

    repo, _, name = cfg.hf_dataset.partition(":")
    ds = load_dataset(repo, name or None, split=split or cfg.hf_split)
    return [
        r[cfg.text_field]
        for r in ds
        if len(r[cfg.text_field].strip()) >= cfg.min_chars
    ]


class PackedCanvasDataset(CanvasDataset):
    """Token-packed windows: paragraphs are concatenated with an EOS separator and
    then cut into contiguous (prefix, canvas) windows.

    Act I/II used per-paragraph windows, which throws away every paragraph shorter
    than the window and wastes the tail of every longer one. Packing matters at Act
    III scale: the diffusion term only supervises canvas positions, so token
    efficiency directly determines how much denoising signal a step carries.
    """

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        prefix_length: int,
        canvas_length: int,
        max_examples: int | None = None,
        eos_id: int | None = None,
        max_source_texts: int | None = None,
    ):
        window = prefix_length + canvas_length
        if window < 2:
            raise ValueError("prefix_length + canvas_length must be >= 2")
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length
        self.canvas_length = canvas_length

        need = (max_examples or 1_000_000) * window + window
        buf: list[np.ndarray] = []
        total = 0
        for i, text in enumerate(texts):
            if max_source_texts is not None and i >= max_source_texts:
                break
            ids = encode(tokenizer, text)
            if ids.size == 0:
                continue
            buf.append(ids)
            total += ids.size
            if eos_id is not None:
                buf.append(np.array([eos_id], dtype=np.int64))
                total += 1
            if total >= need:
                break

        if not buf:
            raise ValueError("no text to pack")
        stream = np.concatenate(buf)
        n_windows = stream.size // window
        if max_examples:
            n_windows = min(n_windows, max_examples)
        if n_windows == 0:
            raise ValueError(
                f"packed stream has {stream.size} tokens, fewer than one window of {window}"
            )

        self.total_tokens = int(n_windows * window)
        self.examples = []
        for w in range(n_windows):
            chunk = stream[w * window : (w + 1) * window]
            self.examples.append(
                CanvasExample(
                    prefix_ids=chunk[:prefix_length].copy(),
                    canvas_ids=chunk[prefix_length:].copy(),
                    text="",
                )
            )


def build_packed_dataset(
    cfg, tokenizer, canvas_length: int, split: str, max_examples: int, eos_id: int | None = None
) -> PackedCanvasDataset:
    texts = load_texts(cfg, split=split)
    return PackedCanvasDataset(
        texts=texts,
        tokenizer=tokenizer,
        prefix_length=cfg.prefix_length,
        canvas_length=canvas_length,
        max_examples=max_examples,
        eos_id=eos_id,
    )


def build_dataset(
    cfg, tokenizer, canvas_length: int, split: str | None = None, max_examples: int | None = None
) -> CanvasDataset:
    """Construct the dataset described by a `DataConfig`."""
    texts = load_texts(cfg, split=split)
    limit = max_examples if max_examples is not None else cfg.max_examples
    return CanvasDataset(
        texts=texts,
        tokenizer=tokenizer,
        prefix_length=cfg.prefix_length,
        canvas_length=canvas_length,
        max_examples=limit,
    )
