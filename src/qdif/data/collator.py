"""Batching and dynamic corruption.

The collator is where a clean batch becomes a noisy training example. It draws a
fresh timestep per example on every call, so the same dataset row is seen at many
noise levels across training -- which is the whole point of a diffusion objective
and the reason no pre-corrupted corpus is ever written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ..config import DiffusionConfig
from ..diffusion.corruption import CorruptionResult, corrupt_canvas
from ..diffusion.schedule import sample_timesteps
from .datasets import CanvasExample


@dataclass
class DiffusionBatch:
    prefix_ids: torch.Tensor  # [B, P]
    canvas_x0: torch.Tensor  # [B, C] clean target
    canvas_xt: torch.Tensor  # [B, C] corrupted input
    t: torch.Tensor  # [B]
    corrupted_mask: torch.Tensor  # [B, C] bool
    texts: list[str]

    @property
    def batch_size(self) -> int:
        return self.canvas_x0.shape[0]

    def to(self, device) -> "DiffusionBatch":
        return DiffusionBatch(
            prefix_ids=self.prefix_ids.to(device),
            canvas_x0=self.canvas_x0.to(device),
            canvas_xt=self.canvas_xt.to(device),
            t=self.t.to(device),
            corrupted_mask=self.corrupted_mask.to(device),
            texts=self.texts,
        )


class DiffusionCollator:
    """Stacks `CanvasExample`s and applies the forward noising process."""

    def __init__(
        self,
        cfg: DiffusionConfig,
        vocab_size: int,
        mask_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ):
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.mask_token_id = cfg.mask_token_id if cfg.mask_token_id is not None else mask_token_id
        self.generator = generator

    def corrupt(self, x0: torch.Tensor, t: torch.Tensor) -> CorruptionResult:
        return corrupt_canvas(
            x0,
            t,
            vocab_size=self.vocab_size,
            mode=self.cfg.corruption,
            schedule=self.cfg.schedule,
            mask_token_id=self.mask_token_id,
            generator=self.generator,
        )

    def __call__(
        self, examples: Sequence[CanvasExample], t: torch.Tensor | float | None = None
    ) -> DiffusionBatch:
        """Args:
            examples: batch of clean windows.
            t: fixed noise level(s) for evaluation. None -> sample from the schedule.
        """
        # Datasets store numpy so both backends share one data path; the torch
        # collator converts at the boundary.
        prefix = torch.from_numpy(np.stack([e.prefix_ids for e in examples]))
        x0 = torch.from_numpy(np.stack([e.canvas_ids for e in examples]))
        B = x0.shape[0]

        if t is None:
            t_vec = sample_timesteps(
                B,
                schedule=self.cfg.schedule,
                t_min=self.cfg.t_min,
                t_max=self.cfg.t_max,
                generator=self.generator,
            )
        elif isinstance(t, (int, float)):
            t_vec = torch.full((B,), float(t), dtype=torch.float32)
        else:
            t_vec = t.reshape(-1).float()
            if t_vec.numel() == 1:
                t_vec = t_vec.expand(B).clone()

        result = self.corrupt(x0, t_vec)
        return DiffusionBatch(
            prefix_ids=prefix,
            canvas_x0=result.x0,
            canvas_xt=result.xt,
            t=result.t,
            corrupted_mask=result.corrupted_mask,
            texts=[e.text for e in examples],
        )


def batches(dataset, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Infinite batch iterator over dataset indices."""
    n = len(dataset)
    while True:
        order = (
            torch.randperm(n, generator=generator).tolist() if shuffle else list(range(n))
        )
        for i in range(0, n - batch_size + 1, batch_size):
            yield [dataset[j] for j in order[i : i + batch_size]]
