#!/usr/bin/env python
"""Act IV-U6 wall-clock benchmark: calibration, pilot and decisive runs.

Uses the Act IV-S harness unchanged (`spec_decode.interleaved_benchmark`): the same 27
prompts, 128 tokens, arms interleaved inside each prompt inside each repeat, keyed draft
noise and snapshot transactions. Adds only the decoupled arm kind and U6's bookkeeping:
* an adapter digest check against the manifest;
* backbone digests before and after the session;
* a bf16-ULP audit of every divergence from native AR.

    u6_bench.py calibrate   A, B, B2                 3 repeats  -> floor (criteria §6)
    u6_bench.py pilot       A, B, B_dec, staged [+T] 1 repeat   -> gate U6-3
    u6_bench.py decisive    A, B, B_dec, staged [+T] 3 repeats  -> gates U6-4/U6-5

Truncate arms are included only if `results/act4u6/diagnostic_wide_draft.json` says gate
U6-1 kept Variant 1 alive.
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

RESULTS = REPO / "results" / "act4u6"
STAGED = [(8, 4), (8, 2), (6, 4), (6, 3)]


def bf16_ulp(value: float) -> float:
    if value == 0 or not math.isfinite(value):
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(abs(value))) - 7)


def arms_for(mode: str, include_truncate: bool) -> list[dict]:
    arms = [{"name": "ar", "kind": "ar"}, {"name": "uno_k4", "kind": "uno", "k": 4}]
    if mode == "calibrate":
        return arms + [{"name": "uno_k4_b2", "kind": "uno", "k": 4}]
    arms.append({"name": "dec_d4v4", "kind": "decoupled", "draft": 4, "verify": 4, "staged": False})
    arms += [{"name": f"staged_d{d}v{v}", "kind": "decoupled", "draft": d, "verify": v, "staged": True}
             for d, v in STAGED]
    if include_truncate:
        arms += [{"name": f"trunc_d{d}v{v}", "kind": "decoupled", "draft": d, "verify": v, "staged": False}
                 for d, v in STAGED]
    return arms


def ulp_audit(model, rows: list[dict]) -> dict:
    import mlx.core as mx

    audited = []
    for row in rows:
        if row.get("first_divergence") is None or not row.get("divergence_context"):
            continue
        logits = model.ar_logits(mx.array([row["divergence_context"]], dtype=mx.int32))[0, -1]
        f32 = logits.astype(mx.float32)
        a, s = row["divergence_tokens"]
        la, ls = float(f32[a].item()), float(f32[s].item())
        ulp = bf16_ulp(max(abs(la), abs(ls)))
        audited.append({"arm": row["arm"], "prompt": row["prompt"][:60], "repeat": row["repeat"],
                        "position": row["first_divergence"], "gap": abs(la - ls),
                        "gap_in_ulps": round(abs(la - ls) / ulp, 3),
                        "within_one_ulp": abs(la - ls) <= ulp * 1.0001})
    by_arm: dict[str, dict] = {}
    for rec in audited:
        slot = by_arm.setdefault(rec["arm"], {"divergences": 0, "beyond_one_ulp": 0})
        slot["divergences"] += 1
        slot["beyond_one_ulp"] += int(not rec["within_one_ulp"])
    return {"divergences": len(audited),
            "beyond_one_ulp": sum(not r["within_one_ulp"] for r in audited),
            "by_arm": by_arm, "audit": audited}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("calibrate", "pilot", "decisive"))
    ap.add_argument("--adapter", default="runs/u4b1/step-16000/adapter.safetensors")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()
    repeats = args.repeats or (1 if args.mode == "pilot" else 3)

    import spec_decode as sd
    from u6_wide_draft_diagnostic import check_adapter

    include_truncate = False
    if args.mode != "calibrate":
        diag = RESULTS / "diagnostic_wide_draft.json"
        if not diag.is_file():
            raise SystemExit("run the U6-1 diagnostic first: gate U6-1 decides the truncate arms")
        include_truncate = json.loads(diag.read_text())["gate_result"]["variant1_alive"]

    sha = check_adapter(args.adapter)
    model = sd.load(args.adapter)
    digest_before = model.fingerprint(full=True).digest
    prompts = sd.suite_prompts(model.tokenizer)
    arms = arms_for(args.mode, include_truncate)
    print(f"[u6] {args.mode}: {len(prompts)} prompts x {len(arms)} arms x {repeats} repeats "
          f"@ {args.tokens} tokens (truncate arms: {include_truncate})", flush=True)

    started = time.perf_counter()
    rows = sd.interleaved_benchmark(model, arms, prompts, args.tokens, repeats, args.warmup,
                                    keep_traces=True)
    digest_after = model.fingerprint(full=True).digest
    audit = ulp_audit(model, rows)

    payload = {
        "act": "IV-U6", "mode": args.mode, "provenance": sd.provenance(model),
        "adapter": args.adapter, "adapter_sha256": sha,
        "backbone_digest_before": digest_before, "backbone_digest_after": digest_after,
        "backbone_unchanged": digest_before == digest_after,
        "config": {"tokens": args.tokens, "repeats": repeats, "warmup": args.warmup,
                   "noise_stream_seed": sd.NOISE_STREAM_SEED, "transaction_mode": "snapshot",
                   "include_truncate": include_truncate},
        "arms": arms, "ulp_audit": audit,
        "summary": sd.speedup_table(sd.aggregate(rows, ("arm",))),
        "summary_clean": sd.speedup_table(sd.aggregate(sd.clean_rows(rows), ("arm",))),
        "seconds": round(time.perf_counter() - started, 1),
        "rows": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{args.mode}.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    sd.report_summary(payload["summary"], sd.check_losslessness(rows))
    print(f"\n[u6] divergences {audit['divergences']}, beyond one bf16 ULP {audit['beyond_one_ulp']}; "
          f"backbone unchanged {payload['backbone_unchanged']}; {payload['seconds']:.0f}s; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
