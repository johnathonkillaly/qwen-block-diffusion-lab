#!/usr/bin/env python
"""Act IV-U6 scoring: the practical floor, the pilot kill rule, the decisive gates and the
verdict, exactly as `docs/act4u6_preregistered_criteria.md` defines them.

Runs on the recorded JSON in `results/act4u6/`; no model needed.

    u6_report.py floor      calibration.json            -> criteria §6 floor
    u6_report.py pilot      pilot.json + floor          -> gate U6-3
    u6_report.py final      decisive.json (or pilot.json if killed) + diagnostic
                            + cost curve                -> gates, verdict, plots

Pairing unit throughout: one (prompt, repeat) cell. Arms were interleaved inside it, so
the pairing removes prompt difficulty and slow drift. The bootstrap is the harness's own
(`uno._paired_bootstrap`, 10,000 resamples, fixed seed).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

RESULTS = REPO / "results" / "act4u6"
PLOTS = REPO / "plots" / "act4u6"
INCUMBENT = "uno_k4"
CALIBRATION_TWIN = "uno_k4_b2"
CONTROL = "dec_d4v4"
FLOOR_MIN = 1.0
CLEAN_THRESHOLD = 0.05
#: Act IV-S measured per-cycle overhead outside the forwards (snapshot, block build, accept
#: test), used only for the perfect-drafter bound in gate U6-2.
OVERHEAD_MS_PER_STAGE = 0.83


def _boot(pairs):
    from uno import _paired_bootstrap
    return _paired_bootstrap(pairs)


# ------------------------------------------------------------------ primitives


def floor_from_differences(control: list[float], twin: list[float], boot=None) -> dict:
    """Criteria §6: the largest mean difference an identical-code pair produces at 95%,
    as a percentage of the incumbent's mean tok/s, rounded up to 0.5, never below 1.0."""
    boot = boot or _boot
    b = boot(list(zip(control, twin)))
    mean_control = statistics.fmean(control)
    raw = 100.0 * max(abs(b["ci_low"]), abs(b["ci_high"])) / mean_control
    return {"units": len(control), "mean_incumbent_tok_s": mean_control,
            "mean_diff_tok_s": b["mean_diff"], "ci_tok_s": [b["ci_low"], b["ci_high"]],
            "raw_percent": raw, "floor_percent": max(FLOOR_MIN, math.ceil(raw * 2) / 2)}


def units(payload: dict, arm: str, clean_only: bool = False) -> dict:
    out = {}
    for row in payload["rows"]:
        if row["arm"] != arm:
            continue
        if clean_only and row.get("ar_looped_fraction", 0.0) >= CLEAN_THRESHOLD:
            continue
        out[(row["prompt"], row["repeat"])] = row
    return out


def paired_percent(payload: dict, arm: str, clean_only: bool = False, boot=None) -> dict:
    """Mean tok/s difference vs the incumbent, as a percentage with a 95% CI."""
    boot = boot or _boot
    base, other = units(payload, INCUMBENT, clean_only), units(payload, arm, clean_only)
    keys = sorted(set(base) & set(other))
    control = [base[k]["tokens_per_second"] for k in keys]
    treated = [other[k]["tokens_per_second"] for k in keys]
    if not keys:
        return {"units": 0}
    b = boot(list(zip(control, treated)))
    m = statistics.fmean(control)
    return {"units": len(keys), "mean_incumbent_tok_s": m, "mean_arm_tok_s": statistics.fmean(treated),
            "percent": 100 * b["mean_diff"] / m, "ci_percent": [100 * b["ci_low"] / m, 100 * b["ci_high"] / m]}


def paired_metric(payload: dict, arm: str, fn, boot=None) -> dict:
    boot = boot or _boot
    base, other = units(payload, INCUMBENT), units(payload, arm)
    keys = sorted(set(base) & set(other))
    b = boot([(fn(base[k]), fn(other[k])) for k in keys])
    return {"mean_diff": b["mean_diff"], "ci": [b["ci_low"], b["ci_high"]], "units": len(keys),
            "incumbent_mean": statistics.fmean(fn(base[k]) for k in keys),
            "arm_mean": statistics.fmean(fn(other[k]) for k in keys)}


def committed_per_cycle(row):
    c = row["committed_per_cycle"]
    return statistics.fmean(c) if c else 0.0


def accepted_per_cycle(row):
    a = row["accepted_per_cycle"]
    return statistics.fmean(a) if a else 0.0


def arm_kind(name: str) -> str:
    if name.startswith("staged_"):
        return "staged"
    if name.startswith("trunc_"):
        return "trunc"
    return "control" if name == CONTROL else "other"


# ------------------------------------------------------------------------ gates


def pilot_kill(arms: dict, floor: float) -> dict:
    """Gate U6-3: kill the decisive run only if *every* decoupled arm's CI upper bound
    lies below −floor."""
    decoupled = {n: a for n, a in arms.items() if arm_kind(n) in ("staged", "trunc")}
    below = {n: a["tok_s"]["ci_percent"][1] < -floor for n, a in decoupled.items()}
    return {"floor_percent": floor, "clearly_below": below,
            "kill": bool(decoupled) and all(below.values())}


def score_arm(arm: dict, floor: float) -> dict:
    """U6-4 pass, robustness and the U6-5 losslessness requirement."""
    full, clean = arm["tok_s"], arm.get("tok_s_clean", {})
    passes = full["ci_percent"][0] > 0 and full["percent"] >= floor
    robust = passes and clean.get("units", 0) > 0 and clean["percent"] > 0
    return {"passes": passes, "robust": robust, "lossless": arm["beyond_one_ulp"] == 0,
            "wins": passes and robust and arm["beyond_one_ulp"] == 0}


def verdict(arms: dict, variant1_alive: bool, floor: float) -> dict:
    """Criteria §8: the first rule that applies."""
    scored = {n: {**a, **score_arm(a, floor)} for n, a in arms.items()
              if arm_kind(n) in ("staged", "trunc")}
    staged = {n: a for n, a in scored.items() if arm_kind(n) == "staged"}
    trunc = {n: a for n, a in scored.items() if arm_kind(n) == "trunc"}

    def more_committed(a):
        return a["committed_per_cycle"]["ci"][0] > 0

    def more_accepted(a):
        return a["accepted_per_cycle"]["ci"][0] > 0

    if any(a["wins"] for a in staged.values()):
        return {"verdict": "STAGED VERIFICATION WINS", "rule": 1,
                "arms": [n for n, a in staged.items() if a["wins"]]}
    if any(a["wins"] for a in trunc.values()):
        return {"verdict": "DECOUPLED WIDTH WINS", "rule": 2,
                "arms": [n for n, a in trunc.items() if a["wins"]]}
    verify_dominated = [n for n, a in staged.items()
                        if more_committed(a) and a["added_verify_ms_per_cycle"] > a["added_draft_ms_per_cycle"]]
    if verify_dominated:
        return {"verdict": "VERIFY COST DOMINATES", "rule": 3, "arms": verify_dominated}
    accepted_more = [n for n, a in scored.items() if more_accepted(a)]
    if variant1_alive or accepted_more:
        return {"verdict": "WIDER DRAFT HELPS, BUT NOT WALL CLOCK", "rule": 4, "arms": accepted_more}
    if not variant1_alive and not any(more_committed(a) for a in staged.values()):
        return {"verdict": "DRAFT WIDTH DOES NOT AFFECT USEFUL PREFIX", "rule": 5, "arms": []}
    return {"verdict": "NO BENEFIT FROM DECOUPLING", "rule": 6, "arms": []}


def perfect_drafter_bound(cost: dict) -> dict:
    """Gate U6-2: committed tokens per ms if every stage were fully accepted, from the
    measured cost curve. Staged verification is killed only if no staged arm beats an
    equally perfect incumbent."""
    verify = {int(w): v["median"] for w, v in cost["verify_ms_by_width"].items()}
    commit = {int(w): v["median"] for w, v in cost["commit_ms_by_verify_width"].items()}
    draft = {int(w): v["median"] for w, v in cost["draft_ms_by_width"].items()}

    def arm(d, v, staged):
        widths, next_index = [v + 1], v
        while staged and next_index < d:
            guesses = min(v, d - next_index - 1)
            if guesses <= 0:
                break
            widths.append(1 + guesses)
            next_index += 1 + guesses
        tokens = sum(widths)  # a perfect stage commits every presented token plus one
        ms = draft[d] + sum(verify[w] + commit[w] + OVERHEAD_MS_PER_STAGE for w in widths)
        return {"stage_widths": widths, "tokens": tokens, "ms": round(ms, 3),
                "tokens_per_ms": tokens / ms}

    arms = {"incumbent_d4v4": arm(4, 4, False)}
    for d, v in ((8, 4), (8, 2), (6, 4), (6, 3)):
        arms[f"staged_d{d}v{v}"] = arm(d, v, True)
    base = arms["incumbent_d4v4"]["tokens_per_ms"]
    beats = {n: a["tokens_per_ms"] > base for n, a in arms.items() if n.startswith("staged")}
    return {"arms": arms, "staged_beats_perfect_incumbent": beats,
            "staged_killed": not any(beats.values())}


# ---------------------------------------------------------------------- assembly


def assemble(payload: dict, boot=None) -> dict:
    audit = payload["ulp_audit"]["by_arm"]
    arms = {}
    for spec in payload["arms"]:
        name = spec["name"]
        if name in ("ar", INCUMBENT, CALIBRATION_TWIN):
            continue
        tok = paired_percent(payload, name, boot=boot)
        arms[name] = {
            "spec": spec, "tok_s": tok, "tok_s_clean": paired_percent(payload, name, True, boot=boot),
            "committed_per_cycle": paired_metric(payload, name, committed_per_cycle, boot=boot),
            "accepted_per_cycle": paired_metric(payload, name, accepted_per_cycle, boot=boot),
            "added_verify_ms_per_cycle": paired_metric(
                payload, name, lambda r: r["verify_ms_per_cycle"], boot=boot)["mean_diff"],
            "added_draft_ms_per_cycle": paired_metric(
                payload, name, lambda r: r["proposal_ms_per_cycle"], boot=boot)["mean_diff"],
            "beyond_one_ulp": audit.get(name, {}).get("beyond_one_ulp", 0),
            "divergences": audit.get(name, {}).get("divergences", 0),
        }
    return arms


def stage_profile(payload: dict) -> dict:
    """Per arm: stages per cycle, verify tokens per committed token, unused draft tokens."""
    out = {}
    for spec in payload["arms"]:
        if spec.get("kind") != "decoupled":
            continue
        traces = [r for row in payload["rows"] if row["arm"] == spec["name"] for r in row.get("trace", [])]
        if not traces:
            continue
        stages = [len(t["stages"]) for t in traces]
        verify_tokens = sum(s["verify_tokens"] for t in traces for s in t["stages"])
        committed = sum(t["committed"] for t in traces)
        hist = {}
        for s in stages:
            hist[str(s)] = hist.get(str(s), 0) + 1
        out[spec["name"]] = {
            "cycles": len(traces), "stages_per_cycle_histogram": hist,
            "mean_stages_per_cycle": statistics.fmean(stages),
            "second_stage_rate": sum(1 for s in stages if s > 1) / len(stages),
            "verify_tokens_per_committed_token": verify_tokens / committed,
            "mean_unused_draft_tokens": statistics.fmean(t["unused_draft_tokens"] for t in traces),
            "mean_rejected_draft_tokens": statistics.fmean(
                sum(s["offered"] - s["accepted"] for s in t["stages"]) for t in traces),
            "mean_cycle_ms": statistics.fmean(t["cycle_ms"] for t in traces),
            "sd_cycle_ms": statistics.stdev([t["cycle_ms"] for t in traces]) if len(traces) > 1 else 0.0,
        }
    return out


def plots(payload: dict, arms: dict, cost: dict, profile: dict, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    summary = {s["arm"]: s for s in payload["summary"]}
    names = [n for n in (INCUMBENT, CONTROL, "staged_d6v4", "staged_d8v4", "staged_d6v3", "staged_d8v2",
                         "trunc_d6v4", "trunc_d8v4", "trunc_d6v3", "trunc_d8v2") if n in summary]
    written = []

    def save(fig, name):
        fig.tight_layout(); fig.savefig(out / name, dpi=140); plt.close(fig); written.append(name)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["native AR"] + names, [summary["ar"]["tokens_per_second"]["median"]] +
           [summary[n]["tokens_per_second"]["median"] for n in names],
           color=["#888", "#2b7bba", "#7fb3d5"] + ["#d95f02"] * (len(names) - 2))
    ax.set_ylabel("committed tok/s (median)"); ax.tick_params(axis="x", rotation=30)
    ax.set_title("1. Throughput by (K_draft, K_verify)")
    save(fig, "01_throughput_by_arm.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label in (("mean_accepted_prefix", "accepted / cycle"), ("mean_committed_per_cycle", "committed / cycle")):
        ax.plot(names, [summary[n][key] for n in names], "o-", label=label)
    ax.set_ylabel("tokens per cycle"); ax.tick_params(axis="x", rotation=30); ax.legend()
    ax.set_title("2 & 6. Accepted and committed tokens per cycle")
    save(fig, "02_06_accepted_committed_per_cycle.png")

    vw = sorted(int(w) for w in cost["verify_ms_by_width"])
    dw = sorted(int(w) for w in cost["draft_ms_by_width"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(vw, [cost["verify_ms_by_width"][str(w)]["median"] for w in vw], "o-", label="verify forward")
    ax.plot(dw, [cost["draft_ms_by_width"][str(w)]["median"] for w in dw], "s-", label="draft forward")
    f = cost["verify_fit_2_to_9"]
    ax.plot([2, 9], [f["fixed_ms"] + 2 * f["marginal_ms_per_token"], f["fixed_ms"] + 9 * f["marginal_ms_per_token"]],
            "--", c="k", lw=0.8, label=f"verify fit {f['fixed_ms']:.1f} + {f['marginal_ms_per_token']:.2f}·w")
    ax.set_xlabel("forward width (tokens)"); ax.set_ylabel("ms (median)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("3 & 4. Verifier and drafter cost vs width (M4 Max)")
    save(fig, "03_04_cost_vs_width.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    parts = ("proposal_ms_per_cycle", "verify_ms_per_cycle", "commit_ms_per_cycle", "overhead_ms_per_cycle")
    bottom = [0.0] * len(names)
    for part in parts:
        vals = [summary[n][part]["median"] for n in names]
        ax.bar(names, vals, bottom=bottom, label=part.replace("_ms_per_cycle", ""))
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("ms per cycle (median)"); ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=8)
    ax.set_title("5. Cycle breakdown")
    save(fig, "05_cycle_breakdown.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    shown = [n for n in names if n in arms]
    pct = [arms[n]["tok_s"]["percent"] for n in shown]
    lo = [arms[n]["tok_s"]["percent"] - arms[n]["tok_s"]["ci_percent"][0] for n in shown]
    hi = [arms[n]["tok_s"]["ci_percent"][1] - arms[n]["tok_s"]["percent"] for n in shown]
    ax.bar(shown, pct, yerr=[lo, hi], capsize=3, color=["#7fb3d5"] + ["#d95f02"] * (len(shown) - 1))
    ax.axhline(0, c="k", lw=0.8)
    ax.set_ylabel("tok/s vs incumbent K=4 (%)"); ax.tick_params(axis="x", rotation=30)
    ax.set_title("7. Speedup vs incumbent K=4 (paired 95% CI)")
    save(fig, "07_speedup_vs_incumbent.png")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("floor", "pilot", "final"))
    ap.add_argument("--floor", type=float, default=None,
                    help="floor from the preregistered calibration; defaults to recomputing it")
    ap.add_argument("--source", default=None, help="final: decisive.json (default) or pilot.json")
    args = ap.parse_args()

    cal = json.loads((RESULTS / "calibrate.json").read_text())
    control = units(cal, INCUMBENT); twin = units(cal, CALIBRATION_TWIN)
    keys = sorted(set(control) & set(twin))
    floor_rec = floor_from_differences([control[k]["tokens_per_second"] for k in keys],
                                       [twin[k]["tokens_per_second"] for k in keys])
    floor = args.floor if args.floor is not None else floor_rec["floor_percent"]

    if args.mode == "floor":
        (RESULTS / "floor.json").write_text(json.dumps(floor_rec, indent=2) + "\n")
        print(json.dumps(floor_rec, indent=2))
        return 0

    if args.mode == "pilot":
        pilot = json.loads((RESULTS / "pilot.json").read_text())
        arms = assemble(pilot)
        kill = pilot_kill(arms, floor)
        rec = {"floor_percent": floor, "gate": "U6-3", **kill,
               "arms": {n: {"percent": a["tok_s"]["percent"], "ci_percent": a["tok_s"]["ci_percent"]}
                        for n, a in arms.items()}}
        (RESULTS / "pilot_gate.json").write_text(json.dumps(rec, indent=2) + "\n")
        for n, a in rec["arms"].items():
            print(f"  {n:14s} {a['percent']:+6.2f}% [{a['ci_percent'][0]:+6.2f}, {a['ci_percent'][1]:+6.2f}]"
                  f"  clearly below: {kill['clearly_below'].get(n, '-')}")
        print(f"[u6-report] floor {floor}%  pilot kill: {kill['kill']}")
        return 0

    source = args.source or ("decisive.json" if (RESULTS / "decisive.json").is_file() else "pilot.json")
    payload = json.loads((RESULTS / source).read_text())
    diag = json.loads((RESULTS / "diagnostic_wide_draft.json").read_text())
    cost = json.loads((RESULTS / "cost_curve.json").read_text())
    arms = assemble(payload)
    alive = diag["gate_result"]["variant1_alive"]
    result = {
        "source": source, "floor_percent": floor, "floor_record": floor_rec,
        "u6_0": {"adapter_sha256": payload["adapter_sha256"], "backbone_unchanged": payload["backbone_unchanged"],
                 "divergences": payload["ulp_audit"]["divergences"],
                 "beyond_one_ulp": payload["ulp_audit"]["beyond_one_ulp"]},
        "u6_1": diag["gate_result"], "u6_2": perfect_drafter_bound(cost),
        "arms": {n: {**a, **score_arm(a, floor)} for n, a in arms.items()},
        "b_dec_not_slower_than_floor": (arms[CONTROL]["tok_s"]["percent"] >= -floor) if CONTROL in arms else None,
        "stage_profile": stage_profile(payload),
        "verdict": verdict(arms, alive, floor),
    }
    (RESULTS / "u6_scores.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    written = plots(payload, arms, cost, result["stage_profile"], PLOTS)
    for n, a in result["arms"].items():
        t, c = a["tok_s"], a["tok_s_clean"]
        print(f"  {n:14s} {t['percent']:+6.2f}% [{t['ci_percent'][0]:+6.2f}, {t['ci_percent'][1]:+6.2f}] "
              f"clean {c.get('percent', float('nan')):+6.2f}%  committed/cycle "
              f"{a['committed_per_cycle']['mean_diff']:+.3f} [{a['committed_per_cycle']['ci'][0]:+.3f}, "
              f"{a['committed_per_cycle']['ci'][1]:+.3f}]  +verify {a['added_verify_ms_per_cycle']:+.2f} ms "
              f"+draft {a['added_draft_ms_per_cycle']:+.2f} ms  passes={a['passes']} lossless={a['lossless']}")
    print(f"[u6-report] verdict: {result['verdict']}")
    print(f"[u6-report] wrote u6_scores.json and {len(written)} plots to {PLOTS.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
