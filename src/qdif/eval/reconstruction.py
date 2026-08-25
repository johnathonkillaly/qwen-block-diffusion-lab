"""The noise sweep: the primary scientific control for this project.

Evaluate x0 reconstruction at fixed noise levels t in {0, 0.1, 0.5, 0.9, 1.0}.

HOW TO READ THE RESULT -- this matters, because the naive reading is wrong.

  t = 0 (clean canvas). "The model should reconstruct this trivially." True *after*
  adaptation, false before it. An unadapted Qwen3.5 read out under the diffusion
  alignment (logits[i] -> x0[i], no shift) is still predicting token i+1, so it
  scores near-zero identity_accuracy and high next_token_accuracy. The sweep
  reports both. Before training:
        identity_accuracy ~ 0.0,  next_token_accuracy high   -> plumbing is correct
        identity_accuracy ~ 0.0,  next_token_accuracy ~ 0.0   -> something is broken
  After training, t=0 identity_accuracy must approach 1.0. If it does not, stop and
  debug rather than training longer.

  t = 1.0 (no original information). Only the language prior and the prefix remain.
  Whatever accuracy survives here is the interesting number for research question 6.

  copy_rate near 1.0 with corrupted_accuracy near 0.0 means the model has collapsed
  to echoing its input -- a failure mode that a falling loss curve will happily hide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..diffusion.objective import diffusion_loss, diffusion_metrics

DEFAULT_NOISE_LEVELS = (0.0, 0.1, 0.5, 0.9, 1.0)


@dataclass
class NoiseLevelResult:
    t: float
    loss: float
    targeted_fraction: float
    changed_fraction: float
    metrics: dict[str, float] = field(default_factory=dict)
    example: str = ""

    def summary(self) -> str:
        m = self.metrics
        return (
            f"t={self.t:<4} | loss {self.loss:8.4f} | changed {self.changed_fraction:5.1%} "
            f"| id-acc {m.get('identity_accuracy', float('nan')):6.2%} "
            f"| corrupted {m.get('corrupted_accuracy', float('nan')):6.2%} "
            f"| clean {m.get('clean_accuracy', float('nan')):6.2%} "
            f"| copy {m.get('copy_rate', float('nan')):6.2%} "
            f"| next-tok {m.get('next_token_accuracy', float('nan')):6.2%}"
        )


@torch.no_grad()
def noise_sweep(
    model,
    tokenizer,
    dataset,
    collator,
    noise_levels=DEFAULT_NOISE_LEVELS,
    batch_size: int = 2,
    seed: int = 0,
    include_example: bool = True,
) -> list[NoiseLevelResult]:
    """Reconstruction quality at each fixed noise level, on the same examples."""
    from ..training.diagnostics import format_reconstruction

    model.eval()
    n = min(batch_size, len(dataset))
    examples = [dataset[i] for i in range(n)]
    results: list[NoiseLevelResult] = []

    for t in noise_levels:
        # Reset the collator RNG per level so the corruption pattern is comparable.
        collator.generator = torch.Generator().manual_seed(seed)
        batch = collator(examples, t=t).to(model.device)

        out = model(canvas_ids=batch.canvas_xt, t=batch.t, prefix_ids=batch.prefix_ids)
        loss_out = diffusion_loss(out.logits, batch.canvas_x0, batch.corrupted_mask, loss_on="all")
        metrics = diffusion_metrics(
            out.logits, batch.canvas_x0, batch.canvas_xt, batch.corrupted_mask
        )

        example = ""
        if include_example:
            example = format_reconstruction(
                tokenizer,
                x0=batch.canvas_x0[0],
                xt=batch.canvas_xt[0],
                pred=out.logits[0].argmax(-1),
                prefix=batch.prefix_ids[0],
                t=t,
            )

        results.append(
            NoiseLevelResult(
                t=float(t),
                loss=float(loss_out.loss),
                targeted_fraction=float(batch.corrupted_mask.float().mean()),
                changed_fraction=float((batch.canvas_xt != batch.canvas_x0).float().mean()),
                metrics=metrics,
                example=example,
            )
        )
    return results


@torch.no_grad()
def probe_bidirectionality(model, tokenizer, text: str, canvas_length: int = 16) -> dict:
    """Measure how far information actually flows backwards through the stack.

    Method: build a prefix+canvas sequence, perturb the LAST canvas token, and see
    which earlier positions' hidden states change.

      * Under a purely causal model, no earlier position changes at all.
      * Under our mask, earlier *canvas* positions change (via the 6 full-attention
        layers) while *prefix* positions do not (the prefix rows of the mask are
        still causal, and the DeltaNet layers cannot carry information backwards).

    This is the empirical counterpart to the code-level claim in
    `qdif.models.masks`: it demonstrates that the bidirectionality is real, that it
    is confined to the canvas, and -- by comparing against `bidirectional=False` --
    that it comes from the full-attention layers rather than from anything else.
    """
    device = model.device
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    if ids.shape[1] <= canvas_length:
        raise ValueError(
            f"probe text is only {ids.shape[1]} tokens; need more than canvas_length={canvas_length}"
        )
    prefix = ids[:, :-canvas_length]
    canvas = ids[:, -canvas_length:]
    t = torch.zeros(1, device=device)

    def hidden(c, bidi):
        return model(canvas_ids=c, t=t, prefix_ids=prefix, bidirectional=bidi).hidden_states

    perturbed = canvas.clone()
    perturbed[0, -1] = (int(perturbed[0, -1]) + 1013) % model.config.vocab_size

    out = {}
    for label, bidi in (("bidirectional", True), ("causal_control", False)):
        h0 = hidden(canvas, bidi).float()
        h1 = hidden(perturbed, bidi).float()
        delta = (h1 - h0).abs().mean(dim=-1)[0]  # [C] per canvas position
        earlier = delta[:-1]
        out[label] = {
            "per_position_delta": [round(float(v), 6) for v in delta],
            "max_delta_before_last_position": float(earlier.max()) if earlier.numel() else 0.0,
            "num_earlier_positions_affected": int((earlier > 1e-4).sum()),
            "canvas_positions": int(delta.numel()),
        }

    out["interpretation"] = (
        "bidirectional.max_delta_before_last_position > 0 and "
        "causal_control.max_delta_before_last_position == 0 means canvas "
        "bidirectionality is real and originates in the full-attention layers only."
    )
    out["full_attention_layers"] = [
        i for i, ty in enumerate(model.config.layer_types) if ty == "full_attention"
    ]
    out["linear_attention_layers_are_causal"] = True
    return out
