"""Zero-training evaluation of the fusion strategies.

WHY THIS EXISTS
---------------
At 4B the one-batch overfit test saturates: ablations A (causal) and B
(bidirectional) both reach loss 1e-4 and 100% corrupted-position accuracy, so it
cannot discriminate between them. Held-out training is the real answer, but it is
expensive and it is Phase 3.

This is the cheap discriminator available now. With **no training at all** -- LoRA
zero-initialised, timestep conditioner zero-initialised -- the only thing that
differs between conditions is the DeltaNet fusion. So the diffusion loss measured
here answers a clean question:

    does the reverse recurrence give the PRETRAINED model information it can
    already use for x0 prediction, before anything is learned?

Read it carefully, though. Two effects are confounded in a raw A-vs-B comparison:

    (1) genuinely new information from the reverse direction, and
    (2) attenuation of the forward path -- mean fusion outputs (H_f + H_r)/2, which
        also halves the pretrained forward contribution.

`scalar_gate` at g -> 1.0 is exactly the causal path, so sweeping the gate separates
the two: if the loss improves as g moves off 1.0, the reverse direction is carrying
usable signal; if it only improves at g = 0.5, the effect is attenuation.
That sweep is `gate_sweep()` below, and it is the honest version of this measurement.
"""

from __future__ import annotations

import numpy as np

from .deltanet import build_fusion, deltanet_modules, set_bidirectional


def _eval_loss(setup, batches, noise_levels, seed=0):
    import mlx.core as mx

    from .ops import corrupt_canvas, cross_entropy, diffusion_metrics

    model = setup.model
    model.eval()
    vocab = setup.load_report.vocab_size
    out = {}
    for t_val in noise_levels:
        rng = np.random.default_rng(seed)  # identical corruption at every condition
        losses, lifts, corr = [], [], []
        for x0, prefix in batches:
            res = corrupt_canvas(x0, np.full(x0.shape[0], t_val, np.float32), vocab, rng)
            o = model(
                canvas_ids=mx.array(res.xt),
                t=mx.array(res.t),
                prefix_ids=mx.array(prefix),
            )
            mx.eval(o.logits)
            losses.append(float(cross_entropy(o.logits, mx.array(res.x0))))
            m = diffusion_metrics(o.logits, res.x0, res.xt, res.corrupted_mask)
            lifts.append(m["lift_over_copy"])
            corr.append(m["corrupted_accuracy"])
        out[t_val] = {
            "loss": float(np.mean(losses)),
            "lift_over_copy": float(np.mean(lifts)),
            "corrupted_accuracy": float(np.nanmean(corr)),
        }
    return out


def _batches(setup, cfg, num_batches=4):
    from ..data.datasets import build_dataset

    ds = build_dataset(cfg.data, setup.tokenizer, cfg.diffusion.canvas_length)
    B = cfg.training.batch_size
    out = []
    for i in range(0, min(num_batches * B, len(ds)), B):
        ex = [ds[j] for j in range(i, min(i + B, len(ds)))]
        if len(ex) < B:
            break
        out.append(
            (
                np.stack([e.canvas_ids for e in ex]),
                np.stack([e.prefix_ids for e in ex]),
            )
        )
    if not out:
        raise ValueError("dataset produced no full batches")
    return out


def _install_fusion(setup, strategy: str, gate_init: float):
    import mlx.core as mx

    for _, mod in deltanet_modules(setup.model.base_model):
        mod.fusion = build_fusion(
            strategy, mod.base.num_v_heads, mod.base.head_v_dim, gate_init
        )
    mx.eval(setup.model.parameters())


def compare_fusions(setup, cfg, noise_levels=(0.1, 0.25, 0.5, 0.75, 0.9), num_batches=4):
    """Untrained diffusion loss for the causal baseline and each fusion strategy."""
    batches = _batches(setup, cfg, num_batches)
    results = {}

    set_bidirectional(setup.model.base_model, False)
    results["A_causal"] = _eval_loss(setup, batches, noise_levels)

    set_bidirectional(setup.model.base_model, True)
    for strategy in ("mean", "scalar_gate", "token_gate", "concat_proj"):
        _install_fusion(setup, strategy, cfg.bidirectional_deltanet.gate_init)
        results[f"B_{strategy}"] = _eval_loss(setup, batches, noise_levels)

    set_bidirectional(setup.model.base_model, False)
    return {
        "noise_levels": list(noise_levels),
        "num_batches": len(batches),
        "batch_size": cfg.training.batch_size,
        "canvas_length": cfg.diffusion.canvas_length,
        "gate_init": cfg.bidirectional_deltanet.gate_init,
        "results": results,
    }


def control_sweep(setup, cfg, gates=(1.0, 0.75, 0.5, 0.25, 0.0), t_val=0.5, num_batches=4):
    """The de-confounding controls, swept over the same gate values.

        scalar_gate               g*H_f + (1-g)*H_r        -- the real thing
        forward_scaled_control    g*H_f                    -- attenuation only
        shuffled_reverse_control  g*H_f + (1-g)*shuffle(H_r) -- reverse magnitude,
                                                              destroyed alignment

    If `scalar_gate` beats both controls at the same g, the reverse recurrence is
    contributing genuine position-aligned information. If `forward_scaled_control`
    matches it, the effect was attenuation of the pretrained forward path all along.
    """
    batches = _batches(setup, cfg, num_batches)
    set_bidirectional(setup.model.base_model, True)
    out: dict[str, dict] = {}
    for strategy in ("scalar_gate", "forward_scaled_control", "shuffled_reverse_control"):
        rows = {}
        for g in gates:
            _install_fusion(setup, strategy, float(np.clip(g, 1e-4, 1 - 1e-4)))
            rows[g] = _eval_loss(setup, batches, [t_val])[t_val]
        out[strategy] = rows
    set_bidirectional(setup.model.base_model, False)
    return {"t": t_val, "gates": list(gates), "conditions": out, "num_batches": len(batches)}


def gate_sweep(setup, cfg, gates=(1.0, 0.95, 0.9, 0.75, 0.5, 0.25, 0.0), t_val=0.5, num_batches=4):
    """Sweep the scalar fusion gate from pure-forward (g=1) to pure-reverse (g=0).

    This is the measurement that separates "the reverse direction carries usable
    information" from "attenuating the forward path happens to help".

        g = 1.0  -> exactly the causal baseline
        g = 0.5  -> exactly mean fusion
        g = 0.0  -> reverse direction only
    """
    batches = _batches(setup, cfg, num_batches)
    set_bidirectional(setup.model.base_model, True)
    out = {}
    for g in gates:
        _install_fusion(setup, "scalar_gate", float(np.clip(g, 1e-4, 1 - 1e-4)))
        out[g] = _eval_loss(setup, batches, [t_val])[t_val]
    set_bidirectional(setup.model.base_model, False)
    return {"t": t_val, "gates": out, "num_batches": len(batches)}
