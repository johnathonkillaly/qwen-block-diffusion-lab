"""Dataset interface: every example exposes a clean prefix and a clean target canvas.

    CanvasExample(prefix_ids=[P], canvas_ids=[C], text=...)

The canvas is what the diffusion process corrupts; the prefix is the causal
conditioning context and is never corrupted. Corruption is *not* part of the
dataset -- see `qdif.data.collator`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch

from .dev_corpus import DEV_TEXTS


@dataclass
class CanvasExample:
    prefix_ids: torch.Tensor  # [P] int64
    canvas_ids: torch.Tensor  # [C] int64
    text: str = ""

    def decoded(self, tokenizer) -> tuple[str, str]:
        return (
            tokenizer.decode(self.prefix_ids, skip_special_tokens=False),
            tokenizer.decode(self.canvas_ids, skip_special_tokens=False),
        )


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
            ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            for start in range(0, max(len(ids) - window + 1, 0), stride):
                chunk = ids[start : start + window]
                self.examples.append(
                    CanvasExample(
                        prefix_ids=chunk[:prefix_length].clone(),
                        canvas_ids=chunk[prefix_length:].clone(),
                        text=tokenizer.decode(chunk),
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
