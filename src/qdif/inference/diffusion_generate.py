"""Diffusion generation entry point (block-autoregressive)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from ..config import ExperimentConfig
from ..diffusion.sampler import SamplerResult, block_generate
from ..training.diagnostics import memory_label, peak_memory_bytes


@dataclass
class DiffusionGenerationResult:
    prompt: str
    output_text: str
    full_text: str
    output_tokens: int
    steps_per_block: int
    num_blocks: int
    forwards: int
    seconds: float
    tokens_per_sec: float
    tokens_per_forward: float
    peak_memory_gb: float
    memory_kind: str
    blocks: list[SamplerResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[diffusion] {self.output_tokens} tok in {self.seconds:.2f}s "
            f"({self.tokens_per_sec:.1f} tok/s) | {self.forwards} forwards | "
            f"{self.tokens_per_forward:.2f} tok/forward | "
            f"{self.peak_memory_gb:.2f} GB {self.memory_kind}"
        )


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    cfg: ExperimentConfig,
    steps: int | None = None,
    num_blocks: int | None = None,
    seed: int | None = None,
) -> DiffusionGenerationResult:
    steps = steps if steps is not None else cfg.sampler.steps
    num_blocks = num_blocks if num_blocks is not None else cfg.sampler.num_blocks
    gen = torch.Generator().manual_seed(seed if seed is not None else cfg.training.seed)

    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

    t0 = time.time()
    full_ids, blocks = block_generate(
        model,
        prompt_ids=prompt_ids,
        canvas_length=cfg.diffusion.canvas_length,
        num_blocks=num_blocks,
        steps=steps,
        strategy=cfg.sampler.strategy,
        temperature=cfg.sampler.temperature,
        adaptive=cfg.sampler.adaptive,
        confidence_threshold=cfg.sampler.confidence_threshold,
        corruption=cfg.diffusion.corruption,
        schedule=cfg.diffusion.schedule,
        mask_token_id=cfg.diffusion.mask_token_id,
        generator=gen,
        self_conditioning=cfg.diffusion.self_conditioning,
    )
    if model.device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.time() - t0

    generated = full_ids[0, prompt_ids.shape[1] :]
    forwards = sum(b.forwards for b in blocks)

    return DiffusionGenerationResult(
        prompt=prompt,
        output_text=tokenizer.decode(generated, skip_special_tokens=False),
        full_text=tokenizer.decode(full_ids[0], skip_special_tokens=False),
        output_tokens=int(generated.numel()),
        steps_per_block=steps,
        num_blocks=num_blocks,
        forwards=forwards,
        seconds=elapsed,
        tokens_per_sec=generated.numel() / elapsed if elapsed else 0.0,
        tokens_per_forward=generated.numel() / max(forwards, 1),
        peak_memory_gb=peak_memory_bytes(model.device) / 1e9,
        memory_kind=memory_label(model.device),
        blocks=blocks,
    )
