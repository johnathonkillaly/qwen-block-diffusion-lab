#!/usr/bin/env python
"""Act IV-U5: score every checkpoint against the *matched* control, from recorded data.

U5-1, U5-3 and U5-4 are defined against `A_k4`, with the §3 test: a paired bootstrap CI
on `mean(arm − control)` that excludes zero, plus a floor. The frozen launcher's curve
session (`runs/u5/eval_curve.json`) paired every checkpoint against `A_k4@12800`, the
shared start, so its transfer table does not score the intermediate checkpoints against
the matched control. The endpoint session does, but only at step 16000.

The per-prompt data needed for the matched comparison was already recorded in the curve
session, same session and same noise streams. This re-pairs it, `arm@step` against
`A_k4@step`, using the harness's own bootstrap (`uno._paired_bootstrap`, fixed seed) and
the §3 floors. It loads no model and changes no recorded result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

FLOOR = {"prefix": 0.0254, "tpf": 0.0062, "tok_s": 0.22}
ARMS = ("B_k6", "C_k8", "D_curr")
STEPS = (13200, 13600, 14400, 15200, 16000)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval", default="runs/u5/eval_curve.json")
    parser.add_argument("--out", default="results/act4u5/u5_matched_step_pairing.json")
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()

    from uno import _paired_bootstrap as boot

    data = json.loads((REPO / args.eval).read_text())
    by = {e["arm"]: e["per_block"]["K4"]["free_running"] for e in data["arms"]}
    rows = []
    for arm in ARMS:
        for step in STEPS:
            control, treated = by[f"A_k4@{step}"], by[f"{arm}@{step}"]
            pc, pt = control["per_prompt"], treated["per_prompt"]
            if [p["prompt"] for p in pc] != [p["prompt"] for p in pt]:
                raise SystemExit(f"prompt order differs for {arm}@{step}")
            row = {"arm": arm, "step": step}
            for label, key, pooled in (
                ("prefix", "mean_accepted_specs",
                 treated["mean_accepted_specs"] - control["mean_accepted_specs"]),
                ("tpf", "tokens_per_forward",
                 treated["tokens_per_forward"] - control["tokens_per_forward"]),
                ("tok_s", "tokens_per_second", None),
            ):
                b = boot([(x[key], y[key]) for x, y in zip(pc, pt)], resamples=args.resamples)
                # Point estimates follow the frozen transfer table: pooled deltas for prefix
                # and TPF, the paired per-prompt mean for tok/s (U5-4).
                point = pooled if pooled is not None else b["mean_diff"]
                row[label] = {"point": point, "ci": [b["ci_low"], b["ci_high"]],
                              "significant": b["significant"],
                              "passes": b["ci_low"] > 0 and point >= FLOOR[label]}
            row["U5-1"], row["U5-3"], row["U5-4"] = (row["prefix"]["passes"],
                                                    row["tpf"]["passes"],
                                                    row["tok_s"]["passes"])
            rows.append(row)
            print(f"{arm + '@' + str(step):>14}  prefix {row['prefix']['point']:+.4f} "
                  f"[{row['prefix']['ci'][0]:+.4f},{row['prefix']['ci'][1]:+.4f}]  "
                  f"TPF {row['tpf']['point']:+.4f}  tok/s {row['tok_s']['point']:+.3f} "
                  f"[{row['tok_s']['ci'][0]:+.3f},{row['tok_s']['ci'][1]:+.3f}]  "
                  f"U5-1 {'PASS' if row['U5-1'] else 'fail'}  U5-3 {'PASS' if row['U5-3'] else 'fail'}  "
                  f"U5-4 {'PASS' if row['U5-4'] else 'fail'}")

    payload = {
        "session": args.eval, "control": "A_k4 at the same step",
        "test": "criteria §3: paired bootstrap (10,000 resamples, harness implementation) + floor",
        "note": ("The frozen curve session paired against A_k4@12800 (the shared start); this "
                 "re-pairs the recorded per-prompt data against the matched control, which is "
                 "what U5-1/3/4 specify."),
        "any_gate_passed": any(r["U5-1"] or r["U5-3"] or r["U5-4"] for r in rows),
        "significant_anywhere": [
            {"arm": r["arm"], "step": r["step"],
             "metrics": [m for m in ("prefix", "tpf", "tok_s") if r[m]["significant"]]}
            for r in rows if any(r[m]["significant"] for m in ("prefix", "tpf", "tok_s"))],
        "rows": rows,
    }
    out = REPO / args.out
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"any gate passed at any matched step: {payload['any_gate_passed']}")
    print(f"significant anywhere: {payload['significant_anywhere']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
