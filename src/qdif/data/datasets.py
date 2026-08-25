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


def build_dataset(cfg, tokenizer, canvas_length: int) -> CanvasDataset:
    """Construct the dataset described by a `DataConfig`."""
    if cfg.source == "dev":
        texts = DEV_TEXTS
    elif cfg.source == "hf":
        if not cfg.hf_dataset:
            raise ValueError("data.source='hf' requires data.hf_dataset")
        from datasets import load_dataset

        ds = load_dataset(cfg.hf_dataset, split=cfg.hf_split)
        texts = [r[cfg.text_field] for r in ds.select(range(min(len(ds), cfg.max_examples * 4)))]
    else:
        raise ValueError(f"unknown data.source {cfg.source!r}, expected 'dev' or 'hf'")

    return CanvasDataset(
        texts=texts,
        tokenizer=tokenizer,
        prefix_length=cfg.prefix_length,
        canvas_length=canvas_length,
        max_examples=cfg.max_examples,
    )
