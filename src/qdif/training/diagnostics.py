"""Training diagnostics: metric accumulation, memory, and readable reconstructions.

The human-readable CLEAN / NOISY / PREDICTED triple is treated as a first-class
output, not a debug print. At this stage of the project it carries more information
than any scalar: it shows immediately whether the model is denoising, copying its
input, or still emitting next-token continuations.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..diffusion.schedule import noise_bucket


def peak_memory_bytes(device: torch.device | str) -> int:
    """Peak allocated memory. On Apple silicon this is *unified* memory, shared with
    the CPU -- it is not VRAM and must not be reported as such."""
    dev = torch.device(device)
    if dev.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    if dev.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    import psutil

    return int(psutil.Process().memory_info().rss)


def memory_label(device: torch.device | str) -> str:
    return "unified" if torch.device(device).type == "mps" else (
        "vram" if torch.device(device).type == "cuda" else "rss"
    )


def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().float().norm() ** 2)
    return total**0.5


def param_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        total += float(p.detach().float().norm() ** 2)
    return total**0.5


@dataclass
class StepRecord:
    step: int
    loss: float
    learning_rate: float
    t_mean: float
    t_values: list[float]
    corruption_fraction: float
    changed_fraction: float
    grad_norm: float
    param_norm: float
    tokens_per_sec: float
    examples_per_sec: float
    peak_memory_gb: float
    memory_kind: str
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    def summary(self) -> str:
        m = self.metrics
        return (
            f"step {self.step:>4d} | loss {self.loss:8.4f} | lr {self.learning_rate:.2e} "
            f"| t {self.t_mean:.3f} | corrupt {self.corruption_fraction:5.1%} "
            f"| id-acc {m.get('identity_accuracy', float('nan')):5.1%} "
            f"| corr-acc {m.get('corrupted_accuracy', float('nan')):5.1%} "
            f"| lift {m.get('lift_over_copy', float('nan')):+6.1%} "
            f"| copy {m.get('copy_rate', float('nan')):5.1%} "
            f"| next-tok {m.get('next_token_accuracy', float('nan')):5.1%} "
            f"| gnorm {self.grad_norm:7.3f} | {self.tokens_per_sec:7.1f} tok/s "
            f"| {self.peak_memory_gb:.2f} GB {self.memory_kind}"
        )


class NoiseBucketAccumulator:
    """Reconstruction accuracy grouped by noise level, aggregated across steps."""

    def __init__(self, num_buckets: int = 5):
        self.num_buckets = num_buckets
        self._sum: dict[str, float] = defaultdict(float)
        self._n: dict[str, int] = defaultdict(int)

    def update(self, t_values: list[float], accuracy: float) -> None:
        for t in t_values:
            b = noise_bucket(t, self.num_buckets)
            self._sum[b] += accuracy
            self._n[b] += 1

    def summary(self) -> dict[str, float]:
        return {b: self._sum[b] / self._n[b] for b in sorted(self._sum) if self._n[b]}


class RunLogger:
    """Writes JSONL metrics + reconstruction samples into the run directory."""

    def __init__(self, run_dir: Path, echo=print):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.samples_path = self.run_dir / "reconstructions.txt"
        self.echo = echo
        self._t0 = time.time()

    def log_step(self, record: StepRecord, echo: bool = True) -> None:
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        if echo:
            self.echo(record.summary())

    def log_reconstruction(self, block: str, echo: bool = True) -> None:
        with self.samples_path.open("a") as f:
            f.write(block + "\n")
        if echo:
            self.echo(block)

    def log_json(self, name: str, payload: dict) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0


def format_reconstruction(
    tokenizer,
    x0: torch.Tensor,
    xt: torch.Tensor,
    pred: torch.Tensor,
    prefix: torch.Tensor | None = None,
    step: int | None = None,
    t: float | None = None,
    max_chars: int = 400,
) -> str:
    """Render one CLEAN / NOISY / PREDICTED triple for a single example."""

    def dec(ids: torch.Tensor) -> str:
        text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
        text = text.replace("\n", "\\n")
        return text[:max_chars] + ("..." if len(text) > max_chars else "")

    header = "--- reconstruction"
    if step is not None:
        header += f" @ step {step}"
    if t is not None:
        header += f" (t = {t:.3f})"
    header += " ---"

    lines = [header]
    if prefix is not None and prefix.numel():
        lines += ["PREFIX:", dec(prefix), ""]
    matched = int((pred == x0).sum())
    lines += [
        "CLEAN:",
        dec(x0),
        "",
        "NOISY:",
        dec(xt),
        "",
        "PREDICTED:",
        dec(pred),
        "",
        f"exact-token match: {matched}/{x0.numel()} ({matched / x0.numel():.1%})",
        "",
    ]
    return "\n".join(lines)
