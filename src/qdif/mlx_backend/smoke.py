"""The v0.2 architecture smoke test.

Produces, in order, the twelve things that must hold before any training run:

     1  Qwen3.5-4B Base loaded through the Unsloth/MLX path
     2  the architecture actually discovered at runtime (never hard-coded)
     3  forward DeltaNet output
     4  reversed DeltaNet output
     5  position alignment verified (reverse-of-reverse is the identity)
     6  fused output
     7  shared parameter identity verified (4B must not become 8B)
     8  bidirectionality probe: future-token influence at earlier fused positions
     9  one finite diffusion loss
    10  nonzero gradients only where expected
    11  peak unified memory
    12  runtime ratio, bidirectional vs causal, for one forward and one forward+backward

Any failure stops the milestone. This module returns structured results; the CLI
renders them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        return f"  [{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def peak_unified_memory_gb() -> float:
    """MLX peak memory. On Apple silicon this is UNIFIED memory shared with the CPU,
    not VRAM."""
    return mx.get_peak_memory() / 1e9


def active_memory_gb() -> float:
    return mx.get_active_memory() / 1e9


def _flat(tree) -> dict:
    from mlx.utils import tree_flatten

    return dict(tree_flatten(tree))


def to_numpy(a: mx.array) -> np.ndarray:
    """numpy has no bfloat16 dtype, so cast through float32 before crossing over."""
    if a.dtype in (mx.bfloat16, mx.float16):
        a = a.astype(mx.float32)
    return np.asarray(a)


# ------------------------------------------------------------------ 3/4/5/6/7


def check_directional_outputs(setup, canvas_length: int = 16, prefix_length: int = 8) -> list[Check]:
    """Forward, reverse, alignment, fusion and weight sharing, on one real layer."""
    from .deltanet import (
        BidirectionalGatedDeltaNet,
        deltanet_front_end,
        deltanet_recurrence,
    )

    model = setup.model
    checks: list[Check] = []

    layer_idx = next(i for i, l in enumerate(model.layers) if l.is_linear)
    mod = model.layers[layer_idx].linear_attn
    assert isinstance(mod, BidirectionalGatedDeltaNet)
    base = mod.base

    T = prefix_length + canvas_length
    hidden = mx.random.normal((1, T, setup.load_report.hidden_size)).astype(mx.bfloat16)

    # (3) forward direction
    q, k, v, a, b, z = deltanet_front_end(base, hidden, None)
    h_f = deltanet_recurrence(base, q, k, v, a, b, None, use_kernel=False)
    mx.eval(h_f)
    checks.append(
        Check(
            "3 forward DeltaNet output",
            bool(mx.all(mx.isfinite(h_f.astype(mx.float32)))),
            f"shape {tuple(h_f.shape)} dtype {h_f.dtype}, finite, "
            f"mean|h| {float(mx.abs(h_f.astype(mx.float32)).mean()):.4e}",
            {"shape": list(h_f.shape)},
        )
    )

    # (4) reverse direction over the canvas only
    canvas = hidden[:, prefix_length:]
    canvas_rev = mx.flip(canvas, axis=1)
    qr, kr, vr, ar, br, _ = deltanet_front_end(base, canvas_rev, None)
    h_r_rev = deltanet_recurrence(base, qr, kr, vr, ar, br, None, use_kernel=False)
    h_r = mx.flip(h_r_rev, axis=1)
    mx.eval(h_r)
    differs = float(
        mx.abs(h_r.astype(mx.float32) - h_f[:, prefix_length:].astype(mx.float32)).max()
    )
    checks.append(
        Check(
            "4 reverse DeltaNet output",
            bool(mx.all(mx.isfinite(h_r.astype(mx.float32)))) and differs > 0,
            f"shape {tuple(h_r.shape)}, finite, max|H_r - H_f| = {differs:.4e} "
            f"(must be > 0, else the reverse pass is not doing anything)",
            {"max_abs_diff_vs_forward": differs},
        )
    )

    # (5) position alignment: flipping twice is the identity, and the LAST canvas
    # position of the reverse pass sees only itself (it is the reverse pass's t=0).
    round_trip = mx.flip(mx.flip(canvas, axis=1), axis=1)
    aligned = bool(mx.array_equal(round_trip, canvas))
    # Reverse recurrence at the final canvas position depends on no other position:
    # perturb the FIRST canvas position and confirm H_r[-1] is unchanged.
    pert = to_numpy(canvas).copy()
    pert[:, 0, :] += 1.0
    pert_mx = mx.array(pert).astype(canvas.dtype)
    qp, kp, vp, ap, bp, _ = deltanet_front_end(base, mx.flip(pert_mx, axis=1), None)
    h_rp = mx.flip(deltanet_recurrence(base, qp, kp, vp, ap, bp, None, use_kernel=False), axis=1)
    mx.eval(h_rp)
    last_unchanged = float(
        mx.abs(h_rp[:, -1].astype(mx.float32) - h_r[:, -1].astype(mx.float32)).max()
    )
    checks.append(
        Check(
            "5 position alignment",
            aligned and last_unchanged == 0.0,
            f"flip(flip(x)) == x: {aligned}; perturbing canvas[0] leaves H_r[-1] "
            f"unchanged (max delta {last_unchanged:.3e}) -- confirms H_r is realigned "
            f"to original positions and the reverse recurrence starts at the canvas end",
            {"round_trip_exact": aligned, "last_position_delta": last_unchanged},
        )
    )

    # (6) fused output
    mod.set_canvas(prefix_length)
    mod.bidirectional = True
    fused_full = mod(hidden)
    mod.bidirectional = False
    causal_full = mod(hidden)
    mx.eval(fused_full, causal_full)
    prefix_delta = float(
        mx.abs(fused_full[:, :prefix_length].astype(mx.float32)
               - causal_full[:, :prefix_length].astype(mx.float32)).max()
    )
    canvas_delta = float(
        mx.abs(fused_full[:, prefix_length:].astype(mx.float32)
               - causal_full[:, prefix_length:].astype(mx.float32)).max()
    )
    checks.append(
        Check(
            "6 fused output",
            canvas_delta > 0 and prefix_delta == 0.0,
            f"fusion={setup.bidi_report['fusion']}, canvas changed (max delta "
            f"{canvas_delta:.4e}), prefix EXACTLY unchanged (max delta {prefix_delta:.3e}) "
            f"-- the leakage boundary holds at layer level",
            {"canvas_delta": canvas_delta, "prefix_delta": prefix_delta},
        )
    )

    # (7) weight sharing: object identity, not value equality
    ids_now = {i: id(m.base.in_proj_qkv.weight) for i, m in _bidi(model)}
    recorded = setup.bidi_report["shared_weight_ids"]
    same_object = all(ids_now.get(i) == recorded.get(i) for i in recorded)
    # Second, structural: the wrapper must expose the SAME array object as the layer.
    probe_mod = model.layers[layer_idx].linear_attn
    identity_ok = probe_mod.base is base and id(probe_mod.base.out_proj.weight) == id(
        base.out_proj.weight
    )
    total_now = sum(v.size for v in _flat(model.parameters()).values())
    expected = setup.load_report.total_params + setup.bidi_report["fusion_params"]
    lora_p = setup.lora_report.lora_params if setup.lora_report else 0
    cond_p = setup.params["trainable_by_family"].get("timestep_conditioner", 0)
    budget_ok = total_now <= expected + lora_p + cond_p + 8
    checks.append(
        Check(
            "7 shared parameter identity",
            same_object and identity_ok and budget_ok,
            f"forward and reverse reference the SAME tensors (id match across "
            f"{len(recorded)} wrapped layers: {same_object}); total parameters "
            f"{total_now:,} vs base {setup.load_report.total_params:,} + fusion "
            f"{setup.bidi_report['fusion_params']:,} + lora {lora_p:,} + cond {cond_p:,} "
            f"-- no DeltaNet weight was duplicated",
            {
                "total_params_now": total_now,
                "base_params": setup.load_report.total_params,
                "fusion_params": setup.bidi_report["fusion_params"],
                "id_match": same_object,
            },
        )
    )
    return checks


def _bidi(model):
    from .deltanet import BidirectionalGatedDeltaNet

    return [
        (i, l.linear_attn)
        for i, l in enumerate(model.layers)
        if isinstance(getattr(l, "linear_attn", None), BidirectionalGatedDeltaNet)
    ]


# ------------------------------------------------------------------- 9/10/12


def check_loss_and_gradients(setup, cfg, canvas_length: int = 16, prefix_length: int = 8):
    """One finite diffusion loss, and gradients only where expected."""
    import mlx.nn as nn

    from .ops import corrupt_canvas, cross_entropy, diffusion_metrics

    model = setup.model
    model.train()
    vocab = setup.load_report.vocab_size
    rng = np.random.default_rng(cfg.training.seed)

    x0 = rng.integers(0, 100000, size=(1, canvas_length), dtype=np.int64)
    prefix = rng.integers(0, 100000, size=(1, prefix_length), dtype=np.int64)
    res = corrupt_canvas(x0, np.array([0.5]), vocab, rng)

    xt = mx.array(res.xt)
    pre = mx.array(prefix)
    t = mx.array(res.t)
    target = mx.array(res.x0)

    def loss_fn(m):
        out = m(canvas_ids=xt, t=t, prefix_ids=pre)
        return cross_entropy(out.logits, target)

    t0 = time.time()
    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, grads)
    grad_seconds = time.time() - t0

    loss_val = float(loss)
    finite = bool(np.isfinite(loss_val))

    flat_grads = _flat(grads)
    nonzero, zero = [], []
    for path, g in flat_grads.items():
        n = float(mx.linalg.norm(g.astype(mx.float32)))
        (nonzero if n > 0 else zero).append((path, n))

    trainable_paths = set(_flat(model.trainable_parameters()))
    unexpected = [p for p, _ in nonzero if p not in trainable_paths]

    def fam(p):
        if p.endswith(("lora_a", "lora_b")):
            return "lora"
        if ".fusion." in p:
            return "fusion"
        if p.startswith("timestep_conditioner"):
            return "timestep_conditioner"
        return "base"

    by_family: dict[str, int] = {}
    for p, _ in nonzero:
        by_family[fam(p)] = by_family.get(fam(p), 0) + 1

    out = model(canvas_ids=xt, t=t, prefix_ids=pre)
    mx.eval(out.logits)
    metrics = diffusion_metrics(out.logits, res.x0, res.xt, res.corrupted_mask)

    return {
        "loss": loss_val,
        "finite": finite,
        "grad_seconds": grad_seconds,
        "num_grad_tensors": len(flat_grads),
        "nonzero_grads": len(nonzero),
        "zero_grads": len(zero),
        "nonzero_by_family": by_family,
        "unexpected_base_grads": unexpected[:5],
        "frozen_clean": not unexpected,
        "zero_grad_paths": [p for p, _ in zero][:8],
        "metrics": metrics,
        "grad_norms": {p: round(n, 6) for p, n in sorted(nonzero, key=lambda x: -x[1])[:6]},
    }


def benchmark_directions(setup, canvas_length: int = 32, prefix_length: int = 16, repeats: int = 3):
    """Runtime of one forward and one forward+backward, bidirectional vs causal.

    v0.2 intentionally buys backward information flow with compute. This measures the
    price rather than assuming it.
    """
    import mlx.nn as nn

    from .deltanet import set_bidirectional
    from .ops import cross_entropy

    model = setup.model
    model.train()
    vocab = setup.load_report.vocab_size
    rng = np.random.default_rng(0)
    xt = mx.array(rng.integers(0, vocab, size=(1, canvas_length), dtype=np.int64))
    pre = mx.array(rng.integers(0, vocab, size=(1, prefix_length), dtype=np.int64))
    target = mx.array(rng.integers(0, vocab, size=(1, canvas_length), dtype=np.int64))
    t = mx.array([0.5])

    def loss_fn(m):
        return cross_entropy(m(canvas_ids=xt, t=t, prefix_ids=pre).logits, target)

    results = {}
    for label, enabled in (("causal", False), ("bidirectional", True)):
        set_bidirectional(model.base_model, enabled)

        # warm-up (kernel compilation / lazy graph build)
        mx.eval(model(canvas_ids=xt, t=t, prefix_ids=pre).logits)

        fwd = []
        for _ in range(repeats):
            t0 = time.time()
            mx.eval(model(canvas_ids=xt, t=t, prefix_ids=pre).logits)
            fwd.append(time.time() - t0)

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss, grads)
        bwd = []
        for _ in range(repeats):
            t0 = time.time()
            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            mx.eval(loss, grads)
            bwd.append(time.time() - t0)

        results[label] = {
            "forward_s": float(np.median(fwd)),
            "fwd_bwd_s": float(np.median(bwd)),
            "peak_unified_gb": peak_unified_memory_gb(),
        }

    c, b = results["causal"], results["bidirectional"]
    results["ratio"] = {
        "forward": b["forward_s"] / c["forward_s"] if c["forward_s"] else float("nan"),
        "fwd_bwd": b["fwd_bwd_s"] / c["fwd_bwd_s"] if c["fwd_bwd_s"] else float("nan"),
    }
    results["canvas_length"] = canvas_length
    results["prefix_length"] = prefix_length
    return results
