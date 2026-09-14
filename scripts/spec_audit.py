#!/usr/bin/env python
"""Two integrity checks that decide how the Act IV-S numbers may be read.

**1. Divergence audit.** Every position where speculative output differs from native AR
is re-examined against the target's own full-context logits, and classified by how far
apart the two contested tokens actually were *in the target's native dtype*. A
divergence where the target had a clear preference is a bug. A divergence where the two
tokens are adjacent bfloat16 values is arithmetic, not logic.

The classification is stated in ULPs rather than in an absolute epsilon, because the
meaningful question is "could these two logits be distinguished in bf16 at this
magnitude?" — and bf16 has 8 mantissa bits, so one ULP at logit ~25 is 0.125.

**2. Degeneration check.** Greedy decoding from a *base* model loops. A loop is trivially
draftable, so any speculative speedup measured on long greedy continuations is partly
measuring degeneration rather than drafting. The baseline run showed acceptance *rising*
with generation length and the `high_entropy` prompts scoring highest of all, which is
the signature. This quantifies it so the headline can be stated against non-degenerate
text instead of being quietly inflated by loops.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
RESULTS = ROOT / "results" / "speculative"


def bf16_ulp(value: float) -> float:
    """Distance to the next representable bfloat16 at this magnitude (8-bit mantissa)."""
    if value == 0 or not math.isfinite(value):
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(abs(value))) - 7)


def repetition_stats(tokens: list[int], n: int = 8) -> dict:
    """How much of this continuation is a loop?

    `looped_fraction` is the share of positions that begin an n-gram which has already
    appeared earlier in the same continuation. A healthy continuation scores near 0; a
    model stuck in a cycle scores near 1.
    """
    if len(tokens) < 2 * n:
        return {"looped_fraction": 0.0, "distinct_ratio": 1.0, "n": n}
    seen: set[tuple] = set()
    repeats = 0
    total = 0
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        total += 1
        if gram in seen:
            repeats += 1
        seen.add(gram)
    return {
        "looped_fraction": round(repeats / total, 4) if total else 0.0,
        "distinct_ratio": round(len(set(tokens)) / len(tokens), 4),
        "n": n,
    }


def cmd_divergence(args) -> int:
    import mlx.core as mx
    from uno import load_model
    from qdif.uno.trainer import load_adapter

    payload = json.loads((RESULTS / args.file).read_text())
    rows = [r for r in payload["rows"] if r.get("first_divergence") is not None]
    print(f"[audit] {len(rows)} divergent rows in {args.file}")
    if not rows:
        return 0

    model, _ = load_model(rank=16, full_attention=True, mlp=True)
    if args.adapter:
        load_adapter(model, args.adapter)

    audited = []
    for i, row in enumerate(rows):
        ctx = mx.array([row["divergence_context"]], dtype=mx.int32)
        logits = model.ar_logits(ctx)[0, -1]
        dtype = str(logits.dtype)
        f32 = logits.astype(mx.float32)
        ar_token, spec_token = row["divergence_tokens"]
        ar_logit = float(f32[ar_token].item())
        spec_logit = float(f32[spec_token].item())
        gap = abs(ar_logit - spec_logit)
        ulp = bf16_ulp(max(abs(ar_logit), abs(spec_logit)))
        audited.append(
            {
                "arm": row["arm"],
                "category": row["category"],
                "max_tokens": row.get("max_tokens"),
                "position": row["first_divergence"],
                "ar_token": ar_token,
                "spec_token": spec_token,
                "ar_logit": ar_logit,
                "spec_logit": spec_logit,
                "contested_gap": gap,
                "ulp": ulp,
                "gap_in_ulps": round(gap / ulp, 3) if ulp else None,
                "logit_dtype": dtype,
                "exact_tie": gap == 0.0,
                "within_one_ulp": gap <= ulp * 1.0001,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(rows)}]", flush=True)

    ulps = [a["gap_in_ulps"] for a in audited if a["gap_in_ulps"] is not None]
    summary = {
        "divergences": len(audited),
        "exact_ties": sum(a["exact_tie"] for a in audited),
        "within_one_ulp": sum(a["within_one_ulp"] for a in audited),
        "beyond_one_ulp": sum(not a["within_one_ulp"] for a in audited),
        "max_gap_in_ulps": round(max(ulps), 3) if ulps else None,
        "median_gap_in_ulps": round(statistics.median(ulps), 3) if ulps else None,
        "verdict": (
            "ARITHMETIC — every divergence is within one bf16 ULP of a tie"
            if all(a["within_one_ulp"] for a in audited)
            else "DEFECT — at least one divergence has a clear target preference"
        ),
    }
    out = RESULTS / args.file.replace(".json", "_divergence_audit.json")
    out.write_text(json.dumps({"summary": summary, "audit": audited}, indent=2) + "\n")
    print(f"\n[audit] wrote {out.name}")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_degeneration(args) -> int:
    """Does greedy decoding loop, and does acceptance track the looping?"""
    from uno import load_model
    from qdif.uno.trainer import load_adapter
    from qdif.uno.decode import ar_greedy_generate, uno_greedy_generate
    from spec_decode import suite_prompts, NOISE_STREAM_SEED

    model, _ = load_model(rank=16, full_attention=True, mlp=True)
    if args.adapter:
        load_adapter(model, args.adapter)
    prompts = suite_prompts(model.tokenizer)

    rows = []
    for length in [int(x) for x in args.lengths.split(",")]:
        for category, text, ids in prompts:
            out, _ = ar_greedy_generate(model, ids, max_tokens=length)
            _, stats = uno_greedy_generate(
                model, ids, max_tokens=length, block_size=4,
                transaction_mode="snapshot", noise_stream_seed=NOISE_STREAM_SEED,
            )
            rep = repetition_stats(out, n=args.ngram)
            rows.append(
                {
                    "max_tokens": length,
                    "category": category,
                    "prompt": text[:60],
                    "looped_fraction": rep["looped_fraction"],
                    "distinct_ratio": rep["distinct_ratio"],
                    "acceptance_rate": round(stats.acceptance_rate, 4),
                    "tokens_per_second": round(stats.tokens_per_second, 3),
                    "mean_accepted": round(
                        statistics.fmean(stats.accepted_per_cycle), 4)
                    if stats.accepted_per_cycle else 0.0,
                }
            )
        print(f"[audit] length {length} done", flush=True)

    # The decisive number: correlation between looping and acceptance.
    def pearson(xs, ys):
        n = len(xs)
        if n < 3:
            return 0.0
        mx_, my = statistics.fmean(xs), statistics.fmean(ys)
        num = sum((x - mx_) * (y - my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x - mx_) ** 2 for x in xs))
        dy = math.sqrt(sum((y - my) ** 2 for y in ys))
        return round(num / (dx * dy), 4) if dx and dy else 0.0

    by_length = {}
    for length in sorted({r["max_tokens"] for r in rows}):
        at = [r for r in rows if r["max_tokens"] == length]
        by_length[str(length)] = {
            "mean_looped_fraction": round(
                statistics.fmean(r["looped_fraction"] for r in at), 4),
            "mean_acceptance": round(
                statistics.fmean(r["acceptance_rate"] for r in at), 4),
            "r_loop_vs_acceptance": pearson(
                [r["looped_fraction"] for r in at], [r["acceptance_rate"] for r in at]),
            "clean_rows": sum(1 for r in at if r["looped_fraction"] < 0.05),
            "clean_mean_acceptance": round(
                statistics.fmean(
                    [r["acceptance_rate"] for r in at if r["looped_fraction"] < 0.05]
                    or [0.0]), 4),
            "looped_mean_acceptance": round(
                statistics.fmean(
                    [r["acceptance_rate"] for r in at if r["looped_fraction"] >= 0.05]
                    or [0.0]), 4),
        }

    out = RESULTS / f"{args.tag}_degeneration.json"
    out.write_text(json.dumps(
        {"by_length": by_length, "rows": rows,
         "pooled_r_loop_vs_acceptance": pearson(
             [r["looped_fraction"] for r in rows],
             [r["acceptance_rate"] for r in rows])},
        indent=2) + "\n")
    print(f"[audit] wrote {out.name}")
    print(json.dumps(by_length, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="runs/u4b1/step-16000/adapter.safetensors")
    ap.add_argument("--tag", default="main")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("divergence")
    p.add_argument("--file", default="main_baseline.json")
    p.set_defaults(func=cmd_divergence)

    p = sub.add_parser("degeneration")
    p.add_argument("--lengths", default="128,512")
    p.add_argument("--ngram", type=int, default=8)
    p.set_defaults(func=cmd_degeneration)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
