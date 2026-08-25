"""Autoregressive control runs.

Without these the experiment says nothing: "the diffusion loss went down" is only
interesting relative to what the same weights do in their original mode.

Three baselines, in decreasing order of scientific value:

  transformers  -- the *same* torch model object used for diffusion, with adapters
                   disabled. This is the only baseline that isolates the effect of
                   the adapters, because nothing else differs.
  mlx           -- mlx-lm running the same checkpoint. Independent implementation;
                   useful as a speed reference and as a cross-check that our AR
                   numbers are not an artefact of our loading path.
  server        -- LM Studio / OMLX OpenAI-compatible endpoint. Convenience only:
                   the server cannot expose the internal forward passes the
                   diffusion sampler needs, so it is never used for diffusion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class ARBaselineResult:
    backend: str
    prompt: str
    output_text: str
    output_tokens: int
    seconds: float
    tokens_per_sec: float
    peak_memory_gb: float = 0.0
    memory_kind: str = "n/a"
    nll: float | None = None
    perplexity: float | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        ppl = f"{self.perplexity:.3f}" if self.perplexity is not None else "n/a"
        return (
            f"[{self.backend}] {self.output_tokens} tok in {self.seconds:.2f}s "
            f"({self.tokens_per_sec:.1f} tok/s) | ppl {ppl} | "
            f"{self.peak_memory_gb:.2f} GB {self.memory_kind}"
        )


@torch.no_grad()
def run_transformers_baseline(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    disable_adapters: bool = True,
) -> ARBaselineResult:
    """Greedy AR decode through the same model object, adapters optionally off."""
    from ..training.diagnostics import memory_label, peak_memory_bytes

    base = getattr(model, "base", model)
    device = next(base.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    import contextlib

    ctx = model.no_diffusion() if (disable_adapters and hasattr(model, "no_diffusion")) else contextlib.nullcontext()
    with ctx:
        t0 = time.time()
        out = base.generate(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.time() - t0
        new_ids = out[0, ids.shape[1] :]

        # Teacher-forced NLL of the model's own continuation, for a perplexity number
        # comparable to the diffusion loss (both are token cross entropy in nats).
        full = out[:, : ids.shape[1] + len(new_ids)]
        logits = base(input_ids=full, use_cache=False).logits
        nll = torch.nn.functional.cross_entropy(
            logits[0, ids.shape[1] - 1 : -1].float(), new_ids
        )

    return ARBaselineResult(
        backend="transformers" + ("" if disable_adapters else "+adapters"),
        prompt=prompt,
        output_text=tokenizer.decode(new_ids, skip_special_tokens=True),
        output_tokens=int(len(new_ids)),
        seconds=elapsed,
        tokens_per_sec=len(new_ids) / elapsed if elapsed else 0.0,
        peak_memory_gb=peak_memory_bytes(device) / 1e9,
        memory_kind=memory_label(device),
        nll=float(nll),
        perplexity=float(nll.exp()),
    )


def run_mlx_baseline(
    model_path: str, prompt: str, max_new_tokens: int = 64
) -> ARBaselineResult:
    """AR decode via mlx-lm. Independent implementation of the same architecture."""
    t0 = time.time()
    import mlx.core as mx
    from mlx_lm import generate, load

    model, tokenizer = load(str(model_path))
    load_time = time.time() - t0

    t1 = time.time()
    text = generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens, verbose=False)
    elapsed = time.time() - t1
    n = len(tokenizer.encode(text))

    return ARBaselineResult(
        backend="mlx-lm",
        prompt=prompt,
        output_text=text,
        output_tokens=n,
        seconds=elapsed,
        tokens_per_sec=n / elapsed if elapsed else 0.0,
        peak_memory_gb=mx.get_peak_memory() / 1e9,
        memory_kind="unified",
        notes=[f"model load took {load_time:.1f}s (excluded from tokens/sec)"],
    )
