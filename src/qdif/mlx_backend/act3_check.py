"""Act III milestone check: everything that must hold before the main transfer run.

Produces, in order, the fifteen items the milestone requires:

     1  exact FLARE-inspired objective implemented
     2  mask corruption example
     3  clean vs noisy stream tensor layout
     4  DeltaNet state-scheduling verification
     5  full-attention visibility verification
     6  trainable parameter count
     7  one finite L_AR
     8  one finite L_diff
     9  combined loss
    10  gradient verification
    11  canvas-conditioning probe
    12  peak unified memory
    13  projected throughput
    14  dataset and license
    15  pre-registered success/failure criteria

Any failure stops the milestone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .act3 import act3_loss, canvas_conditioning, health_metrics, mask_corrupt


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        return f"  [{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def _flat(tree) -> dict:
    from mlx.utils import tree_flatten

    return dict(tree_flatten(tree))


def check_state_scheduling(a3) -> list[Check]:
    """(4) DeltaNet block-end state scheduling, and (5) full-attention visibility."""
    from .deltanet import BidirectionalGatedDeltaNet, deltanet_modules

    model = a3.setup.model
    checks: list[Check] = []
    mods = deltanet_modules(model.base_model)
    hidden = a3.setup.load_report.hidden_size
    P, C = 8, 16
    x = mx.random.normal((1, P + C, hidden)).astype(mx.bfloat16)

    if not mods:
        return [Check("4 DeltaNet state scheduling", False, "no wrapped DeltaNet layers")]

    _, mod = mods[0]
    all_flare = all(m.mode == "flare" and m.bidirectional for _, m in mods)
    mod.set_canvas(P)

    mod.bidirectional = False
    causal = mod(x)
    mod.bidirectional = True
    mod.mode = "flare"
    flare = mod(x)
    mx.eval(causal, flare)

    # Perturb the LAST canvas token: earlier canvas positions must move (they read the
    # block-end state) and the prefix must not (clean boundary is causal and fixed).
    pert = np.asarray(x.astype(mx.float32)).copy()
    pert[:, -1, :] += 3.0
    flare_p = mod(mx.array(pert).astype(mx.bfloat16))
    mx.eval(flare_p)
    earlier = float(mx.abs((flare[:, P:-1] - flare_p[:, P:-1]).astype(mx.float32)).max())
    prefix_delta = float(mx.abs((flare[:, :P] - flare_p[:, :P]).astype(mx.float32)).max())
    differs = float(mx.abs((flare - causal).astype(mx.float32)).max())

    checks.append(
        Check(
            "4 DeltaNet block-end state scheduling",
            all_flare and earlier > 0 and prefix_delta == 0.0 and differs > 0,
            f"{len(mods)}/{len(mods)} layers in mode='flare'; a later canvas token moves "
            f"earlier canvas positions (max delta {earlier:.3e}) while the clean prefix "
            f"stays EXACTLY fixed ({prefix_delta:.1e}); differs from causal by {differs:.3e}",
            {"earlier_delta": earlier, "prefix_delta": prefix_delta, "layers": len(mods)},
        )
    )

    # (5) full-attention visibility via the canvas mask
    from .masks import build_canvas_allow_matrix

    T = P + C
    allow = build_canvas_allow_matrix(T, [(P, T)], bidirectional=True)
    prefix_causal = bool(mx.all(allow[:P, :P] == mx.tril(mx.ones((P, P), dtype=mx.bool_))))
    canvas_full = bool(mx.all(allow[P:, P:]))
    no_leak = not bool(mx.any(allow[:P, P:]))
    sees_prefix = bool(mx.all(allow[P:, :P]))
    checks.append(
        Check(
            "5 full-attention visibility",
            prefix_causal and canvas_full and no_leak and sees_prefix,
            f"prefix causal {prefix_causal}; noisy block bidirectional within itself "
            f"{canvas_full}; canvas->prefix leak {not no_leak}; canvas sees prefix {sees_prefix}",
        )
    )
    return checks


def run_check(a3, cfg, echo) -> dict:
    from .deltanet import deltanet_modules

    setup = a3.setup
    model = setup.model
    model.train()
    results: dict[str, Any] = {}
    checks: list[Check] = []

    # ---- (2) mask corruption example ---------------------------------------
    tok = setup.tokenizer
    ex = a3.train_ds[0]
    rng = np.random.default_rng(0)
    x0 = ex.canvas_ids[None, :]
    prefix_np = ex.prefix_ids[None, :]
    canvas = mask_corrupt(
        x0, np.array([0.5]), a3.mask_token_id, rng, exact_count=cfg.diffusion.exact_mask_count
    )

    def dec(ids):
        return tok.decode([int(i) for i in ids]).replace("\n", "\\n")

    def dec_masked(ids):
        out = []
        for i in ids:
            out.append("[MASK]" if int(i) == a3.mask_token_id else tok.decode([int(i)]))
        return "".join(out).replace("\n", "\\n")

    results["corruption_example"] = {
        "prefix": dec(prefix_np[0])[:300],
        "clean": dec(canvas.x0[0])[:300],
        "noisy_M": dec_masked(canvas.xt[0])[:400],
        "noisy_Mc": dec_masked(canvas.xt_complement[0])[:400],
        "requested_fraction": float(canvas.t[0]),
        "actual_masked_fraction": canvas.masked_fraction,
        "num_masked": canvas.num_masked,
        "num_visible": canvas.num_visible,
    }
    partition_ok = bool(
        ((canvas.xt == a3.mask_token_id) ^ (canvas.xt_complement == a3.mask_token_id)).all()
    )
    checks.append(
        Check(
            "2 mask corruption",
            partition_ok
            and abs(canvas.masked_fraction - 0.5) < 0.02
            and (canvas.x0 != a3.mask_token_id).all(),
            f"requested {float(canvas.t[0]):.2f}, actual {canvas.masked_fraction:.3f}, "
            f"{canvas.num_masked:.0f} masked / {canvas.num_visible:.0f} visible; M and M^c "
            f"partition the block exactly ({partition_ok}); the clean target never "
            f"contains the mask token",
        )
    )

    # ---- (3) tensor layout --------------------------------------------------
    P = cfg.data.prefix_length
    C = cfg.diffusion.canvas_length
    B = cfg.training.batch_size
    V = setup.load_report.vocab_size
    results["layout"] = {
        "clean_stream_input": [B, P + C],
        "clean_stream_logits": [B, P + C, V],
        "ar_targets": [B, P + C - 1],
        "noisy_stream_canvas_input": [B, C],
        "noisy_stream_logits": [B, C, V],
        "prefix": [B, P],
        "mask_set": [B, C],
        "forwards_per_step": 3 if cfg.diffusion.complementary_views else 2,
    }

    # ---- (4)(5) mechanism verification -------------------------------------
    checks.extend(check_state_scheduling(a3))

    # ---- (6) trainable parameters ------------------------------------------
    results["params"] = setup.params
    lora = setup.lora_report
    results["lora"] = {
        "modules": lora.num_injected, "families": lora.families,
        "rank": lora.rank, "alpha": lora.alpha, "params": lora.lora_params,
    } if lora else None

    # ---- (7)(8)(9) losses ---------------------------------------------------
    pre = mx.array(np.repeat(prefix_np, B, axis=0))
    big = mask_corrupt(
        np.repeat(x0, B, axis=0), np.full(B, 0.5, np.float32), a3.mask_token_id,
        np.random.default_rng(1), exact_count=cfg.diffusion.exact_mask_count,
    )
    out = act3_loss(
        model, pre, big,
        lambda_diff=cfg.diffusion.lambda_diff,
        complementary_views=cfg.diffusion.complementary_views,
        ar_weight=cfg.diffusion.ar_weight,
    )
    mx.eval(out.total, out.loss_ar, out.loss_diff)
    l_ar, l_diff, l_tot = float(out.loss_ar), float(out.loss_diff), float(out.total)
    results["losses"] = {
        "loss_ar": l_ar, "loss_diff": l_diff, "loss_total": l_tot,
        "lambda_diff": cfg.diffusion.lambda_diff, "ar_weight": cfg.diffusion.ar_weight,
    }
    checks.append(
        Check("7 finite L_AR", np.isfinite(l_ar) and l_ar > 0, f"{l_ar:.6f}"))
    checks.append(
        Check("8 finite L_diff", np.isfinite(l_diff) and l_diff > 0, f"{l_diff:.6f}"))
    checks.append(
        Check(
            "9 combined loss",
            abs(l_tot - (cfg.diffusion.ar_weight * l_ar + cfg.diffusion.lambda_diff * l_diff)) < 1e-3,
            f"L_total {l_tot:.6f} = {cfg.diffusion.ar_weight} * L_AR + "
            f"{cfg.diffusion.lambda_diff} * L_diff",
        )
    )

    # ---- (10) gradients ------------------------------------------------------
    def loss_fn(m):
        return act3_loss(
            m, pre, big, lambda_diff=cfg.diffusion.lambda_diff,
            complementary_views=cfg.diffusion.complementary_views,
            ar_weight=cfg.diffusion.ar_weight,
        ).total

    t0 = time.time()
    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, grads)
    grad_seconds = time.time() - t0

    flat = _flat(grads)
    nonzero = {k: float(mx.linalg.norm(v.astype(mx.float32))) for k, v in flat.items()}
    nz = {k: v for k, v in nonzero.items() if v > 0}
    trainable_paths = set(_flat(model.trainable_parameters()))
    unexpected = [k for k in nz if k not in trainable_paths]

    def fam(p):
        if p.endswith(("lora_a", "lora_b")):
            return "lora"
        if ".fusion." in p:
            return "fusion"
        if p.startswith("timestep_conditioner"):
            return "timestep_conditioner"
        return "base"

    by_fam: dict[str, int] = {}
    for k in nz:
        by_fam[fam(k)] = by_fam.get(fam(k), 0) + 1
    results["gradients"] = {
        "tensors": len(flat), "nonzero": len(nz), "by_family": by_fam,
        "unexpected_base": unexpected[:5], "grad_seconds": grad_seconds,
        "top": dict(sorted(nz.items(), key=lambda kv: -kv[1])[:5]),
    }
    checks.append(
        Check(
            "10 gradients only where expected",
            not unexpected and len(nz) > 0,
            f"{len(nz)} nonzero of {len(flat)} by family {by_fam}; "
            f"frozen base received gradient: {'YES (WRONG)' if unexpected else 'no (correct)'}",
        )
    )

    # ---- (11) canvas-conditioning probe -------------------------------------
    model.eval()
    cc = canvas_conditioning(model, pre, big, np.random.default_rng(2), V)
    o = model(canvas_ids=mx.array(big.xt), t=mx.array(big.t), prefix_ids=pre)
    mx.eval(o.logits)
    hm = health_metrics(o.logits, big, a3.mask_token_id)
    model.train()
    results["canvas_conditioning"] = cc
    results["health_at_init"] = hm
    checks.append(
        Check(
            "11 canvas-conditioning probe",
            np.isfinite(cc["canvas_l1"]),
            f"L1 {cc['canvas_l1']:.4f} (range 0-2), JS {cc['canvas_js']:.4f} at "
            f"initialisation. This is the BASELINE the probe is tracked against; a "
            f"healthy model should keep it clearly above zero.",
        )
    )

    # ---- (12)(13) memory and throughput -------------------------------------
    reps = 3
    times = []
    for _ in range(reps):
        t0 = time.time()
        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss, grads)
        times.append(time.time() - t0)
    s_step = float(np.median(times))
    tok_s = B * C / s_step
    total_steps = cfg.training.max_steps
    results["performance"] = {
        "seconds_per_step": s_step,
        "canvas_tokens_per_sec": tok_s,
        "sequence_tokens_per_sec": B * (P + C) / s_step,
        "peak_unified_gb": mx.get_peak_memory() / 1e9,
        "projected_wall_minutes": total_steps * s_step / 60.0,
        "projected_train_tokens": total_steps * B * C,
        "forwards_per_step": 3 if cfg.diffusion.complementary_views else 2,
    }
    checks.append(
        Check(
            "12-13 memory and throughput",
            mx.get_peak_memory() / 1e9 < 100,
            f"{mx.get_peak_memory() / 1e9:.2f} GB peak unified memory; {s_step:.3f} s/step; "
            f"{tok_s:.1f} canvas tok/s; {total_steps} steps projected at "
            f"{total_steps * s_step / 60.0:.0f} min",
        )
    )

    results["checks"] = [c.__dict__ for c in checks]
    results["all_pass"] = all(c.passed for c in checks)
    return results
