#!/usr/bin/env python
"""Act IV-U6 gate U6-2: the M4 Max cost curve for verify width and draft width.

Measured through the decoder's own code path: a snapshot transaction around the verify
forward, and the gated draft forward on a restored cache. Each context is the suite
prompt plus 64 greedy tokens, so the cache resembles mid-generation. Widths are
interleaved in a fresh shuffled order on every repeat, one warm-up pass is discarded, and
medians are reported. A fit `ms = F + m·width` separates the fixed cost of a forward from
its marginal cost per token.

The question it answers directly: is verifying 2 twice cheaper or dearer than verifying 4
once?
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

VERIFY_WIDTHS = (2, 3, 4, 5, 6, 7, 8, 9, 13, 17)
DRAFT_WIDTHS = (2, 4, 6, 8, 12, 16)


def fit(points: dict[int, float], widths) -> dict:
    xs = [w for w in widths if w in points]
    ys = [points[w] for w in xs]
    n = len(xs)
    mx_, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx_) ** 2 for x in xs)
    slope = sum((x - mx_) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx_
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return {"fixed_ms": round(intercept, 4), "marginal_ms_per_token": round(slope, 4),
            "r2": round(1 - ss_res / ss_tot, 5) if ss_tot else None, "widths": xs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", default="runs/u4b1/step-16000/adapter.safetensors")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--context-tokens", type=int, default=64)
    ap.add_argument("--out", default="results/act4u6/cost_curve.json")
    args = ap.parse_args()

    import mlx.core as mx

    from qdif.uno.cache_utils import restore_caches, snapshot_caches
    from qdif.uno.decode import ar_greedy_generate
    from qdif.uno.noise import build_draft_block, draft_lora_mask
    from qdif.uno.rng import BENCH_NOISE, stream_key
    from qdif.uno.transaction import begin_transaction
    from spec_decode import NOISE_STREAM_SEED, load, provenance, suite_prompts
    from u6_wide_draft_diagnostic import check_adapter

    sha = check_adapter(args.adapter)
    model = load(args.adapter)
    digest_before = model.fingerprint(full=True).digest
    prompts = suite_prompts(model.tokenizer)

    contexts = []
    for _cat, _text, ids in prompts:
        cont, _ = ar_greedy_generate(model, ids, max_tokens=args.context_tokens)
        contexts.append(list(ids) + list(cont))

    samples = {("verify", w): [] for w in VERIFY_WIDTHS}
    samples.update({("commit", w): [] for w in VERIFY_WIDTHS})
    samples.update({("draft", w): [] for w in DRAFT_WIDTHS})
    snapshot_ms = []
    rng = random.Random(20260914)

    def measure(ctx_index, ctx, record):
        caches = model.make_cache()
        mx.eval(model.ar_logits(mx.array([ctx[:-1]], dtype=mx.int32), cache=caches))
        t0 = time.perf_counter()
        snap = snapshot_caches(caches)
        restore_caches(caches, snap)
        if record:
            snapshot_ms.append(1000 * (time.perf_counter() - t0))
        jobs = [("verify", w) for w in VERIFY_WIDTHS] + [("draft", w) for w in DRAFT_WIDTHS]
        rng.shuffle(jobs)
        seed = ctx[-1]
        for kind, w in jobs:
            restore_caches(caches, snap)
            if kind == "verify":
                ids = mx.array([[seed] + ctx[-(w - 1):]], dtype=mx.int32) if w > 1 else \
                    mx.array([[seed]], dtype=mx.int32)
                txn = begin_transaction(model, caches, mode="snapshot")
                t0 = time.perf_counter()
                logits = txn.run_block(ids)
                mx.eval(logits)
                t1 = time.perf_counter()
                txn.commit_prefix(max(1, w // 2))
                mx.eval([c.state for c in caches])
                t2 = time.perf_counter()
                if record:
                    samples[("verify", w)].append(1000 * (t1 - t0))
                    samples[("commit", w)].append(1000 * (t2 - t1))
            else:
                block = build_draft_block(mx.array([seed], dtype=mx.int32), w, "random_uniform",
                                          model.mask_token_id, model.vocab_size,
                                          key=stream_key(NOISE_STREAM_SEED, BENCH_NOISE, ctx_index))
                mask = draft_lora_mask(1, w)
                t0 = time.perf_counter()
                logits = model.draft_logits(block, mask, cache=caches)
                mx.eval(logits)
                if record:
                    samples[("draft", w)].append(1000 * (time.perf_counter() - t0))

    print("[u6-cost] warm-up", flush=True)
    for i, ctx in enumerate(contexts[:3]):
        measure(i, ctx, record=False)
    started = time.perf_counter()
    for r in range(args.repeats):
        for i, ctx in enumerate(contexts):
            measure(i, ctx, record=True)
        print(f"  repeat {r + 1}/{args.repeats} done ({time.perf_counter() - started:.0f}s)", flush=True)

    def summ(values):
        v = sorted(values)
        return {"n": len(v), "median": round(statistics.median(v), 4),
                "p25": round(v[len(v) // 4], 4), "p75": round(v[(3 * len(v)) // 4], 4),
                "mean": round(statistics.fmean(v), 4)}

    verify = {w: summ(samples[("verify", w)]) for w in VERIFY_WIDTHS}
    commit = {w: summ(samples[("commit", w)]) for w in VERIFY_WIDTHS}
    draft = {w: summ(samples[("draft", w)]) for w in DRAFT_WIDTHS}
    vmed = {w: verify[w]["median"] for w in VERIFY_WIDTHS}
    dmed = {w: draft[w]["median"] for w in DRAFT_WIDTHS}
    comparisons = {
        "verify(3)+verify(3) vs verify(5)  [2 speculative twice vs 4 once]":
            [round(2 * vmed[3], 3), vmed[5]],
        "verify(5)+verify(4) vs verify(9)  [S(8,4) both stages vs coupled K=8]":
            [round(vmed[5] + vmed[4], 3), vmed[9]],
        "verify(3)+verify(3)+verify(3) vs verify(9)  [S(8,2) three stages vs K=8]":
            [round(3 * vmed[3], 3), vmed[9]],
        "draft(8)-draft(4)  [extra cost of drafting 8 instead of 4]":
            [round(dmed[8] - dmed[4], 3)],
        "draft(6)-draft(4)":
            [round(dmed[6] - dmed[4], 3)],
    }
    payload = {
        "gate": "U6-2", "provenance": provenance(), "adapter_sha256": sha,
        "backbone_unchanged": digest_before == model.fingerprint(full=True).digest,
        "contexts": len(contexts), "context_tokens_after_prompt": args.context_tokens,
        "repeats": args.repeats,
        "verify_ms_by_width": verify, "commit_ms_by_verify_width": commit,
        "draft_ms_by_width": draft, "snapshot_restore_ms": summ(snapshot_ms),
        "verify_fit_2_to_9": fit(vmed, range(2, 10)),
        "draft_fit": fit(dmed, DRAFT_WIDTHS),
        "comparisons_ms": comparisons,
        "seconds": round(time.perf_counter() - started, 1),
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print("\nverify ms (median) by width:", vmed)
    print("draft  ms (median) by width:", dmed)
    print("verify fit:", payload["verify_fit_2_to_9"], "| draft fit:", payload["draft_fit"])
    for k, v in comparisons.items():
        print(f"  {k}: {v}")
    print(f"[u6-cost] backbone unchanged: {payload['backbone_unchanged']}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
