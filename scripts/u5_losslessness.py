#!/usr/bin/env python
"""Act IV-U5 gate U5-5, clause 1: cacheless losslessness, measured after the fact.

U5-5 requires verified decoding to "remain bit-exact lossless at `K_decode = 4` in the
cacheless reference regime for every arm". The frozen launcher (`u5_launch.sh`) never
runs that measurement, so no U5 artifact contains it. This script measures it after the
primary run, with the method `uno.py stage0` and U4-B4 use: cacheless AR greedy against
cacheless Uno greedy, identical token IDs required.

It cannot change U5-5's verdict, because clause 3 already fails by the letter (C_k8's
accepted-prefix shortfall). It exists so the clause is measured rather than silently
absent. First run 2026-09-14, after U5's results had been read.

Every divergence, if any, is audited the way Act IV-S audits them: the target's
full-context logits at that position, and the contested gap in bfloat16 ULPs. A gap
within one ULP is arithmetic (a tie broken differently by two forward widths); anything
larger is a defect.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

ARMS = ("A_k4", "B_k6", "C_k8", "D_curr")


def bf16_ulp(value: float) -> float:
    """Distance to the next representable bfloat16 at this magnitude (8-bit mantissa)."""
    if value == 0 or not math.isfinite(value):
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(abs(value))) - 7)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", type=int, default=16000)
    parser.add_argument("--tokens", type=int, default=48, help="u5-eval's generation length")
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--noise-seed", type=int, default=20260905)
    parser.add_argument("--out", default="results/act4u5/u5_losslessness.json")
    args = parser.parse_args()

    import mlx.core as mx
    from uno import load_model
    from qdif.uno.data import prompt_suite_tokens
    from qdif.uno.decode import ar_greedy_generate, uno_greedy_generate
    from qdif.uno.trainer import load_adapter

    model, _ = load_model(rank=16, full_attention=True, mlp=True,
                          purpose="Act IV-U5 U5-5 cacheless losslessness")
    fingerprint = model.fingerprint(full=True).digest
    prompts = prompt_suite_tokens(model.tokenizer)
    started = time.perf_counter()

    arms, references = {}, {}
    for arm in ARMS:
        path = REPO / "runs" / "u5" / arm / f"step-{args.step}" / "adapter.safetensors"
        n = load_adapter(model, str(path))
        rows = []
        for domain, text, ids in prompts:
            # The AR reference is recomputed per arm, with that arm's adapter loaded, so the
            # check also shows the adapter never reaches the AR path.
            reference, _ = ar_greedy_generate(model, ids, max_tokens=args.tokens,
                                              use_cache=False)
            tokens, stats = uno_greedy_generate(
                model, ids, max_tokens=args.tokens, block_size=args.block_size,
                noise_mode="deterministic_uniform", noise_seed=args.noise_seed,
                use_cache=False)
            cached, _ = ar_greedy_generate(model, ids, max_tokens=args.tokens)
            divergence = next((i for i, (a, b) in enumerate(zip(tokens, reference))
                               if a != b), None)
            row = {
                "domain": domain, "prompt": text,
                "identical": tokens == reference,
                "first_divergence": divergence,
                "acceptance_rate": round(stats.acceptance_rate, 4),
                "cached_ar_agreement": round(
                    sum(a == b for a, b in zip(cached, reference)) / max(len(reference), 1), 4),
            }
            if divergence is not None:
                ctx = mx.array([list(ids) + list(reference[:divergence])], dtype=mx.int32)
                logits = model.ar_logits(ctx)[0, -1].astype(mx.float32)
                a_tok, u_tok = reference[divergence], tokens[divergence]
                a_logit, u_logit = float(logits[a_tok].item()), float(logits[u_tok].item())
                ulp = bf16_ulp(max(abs(a_logit), abs(u_logit)))
                row["audit"] = {"ar_token": a_tok, "uno_token": u_tok,
                                "contested_gap": abs(a_logit - u_logit),
                                "gap_in_ulps": round(abs(a_logit - u_logit) / ulp, 3),
                                "within_one_ulp": abs(a_logit - u_logit) <= ulp * 1.0001}
            rows.append(row)
            references.setdefault(text, reference)
            row["ar_reference_same_as_first_arm"] = reference == references[text]
        identical = sum(r["identical"] for r in rows)
        arms[arm] = {
            "adapter": str(path.relative_to(REPO)), "adapter_tensors": n,
            "prompts": len(rows), "identical": identical,
            "lossless": identical == len(rows),
            "divergences_within_one_ulp": sum(
                1 for r in rows if "audit" in r and r["audit"]["within_one_ulp"]),
            "divergences_beyond_one_ulp": sum(
                1 for r in rows if "audit" in r and not r["audit"]["within_one_ulp"]),
            "ar_reference_identical_across_arms": all(
                r["ar_reference_same_as_first_arm"] for r in rows),
            "mean_cached_ar_agreement": round(
                sum(r["cached_ar_agreement"] for r in rows) / len(rows), 4),
            "rows": rows,
        }
        print(f"[u5-lossless] {arm:7s} {identical}/{len(rows)} identical "
              f"(K={args.block_size}, cacheless)  cached-AR agreement "
              f"{arms[arm]['mean_cached_ar_agreement']:.4f}", flush=True)

    backbone_unchanged = model.fingerprint(full=True).digest == fingerprint
    payload = {
        "gate": "U5-5 clause 1 (cacheless losslessness at K_decode=4, every arm)",
        "measured": "after the primary run; the frozen launcher omits this measurement",
        "method": "uno.py stage0 / U4-B4: cacheless AR greedy vs cacheless Uno greedy",
        "harness": {"step": args.step, "tokens": args.tokens, "block_size": args.block_size,
                    "noise_mode": "deterministic_uniform", "noise_seed": args.noise_seed,
                    "prompts": len(prompts)},
        "backbone_digest": fingerprint, "backbone_unchanged": backbone_unchanged,
        "lossless_every_arm": all(a["lossless"] for a in arms.values()),
        "unexplained_divergences": sum(a["divergences_beyond_one_ulp"] for a in arms.values()),
        "seconds": round(time.perf_counter() - started, 1),
        "arms": arms,
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[u5-lossless] lossless every arm: {payload['lossless_every_arm']}  "
          f"unexplained divergences: {payload['unexplained_divergences']}  "
          f"backbone unchanged: {backbone_unchanged}")
    print(f"[u5-lossless] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
