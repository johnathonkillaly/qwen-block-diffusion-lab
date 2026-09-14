#!/usr/bin/env python
"""Act IV-U5 supplementary training-launch robustness (criteria §11 A3 and A4).

Reads the single supplementary paired session, `runs/u5/eval_replicates.json`, in which
three launches of the control, the original `C_k8`, one independent `C_k8_r2`, and the
`B_k6` and `D_curr` endpoints were all evaluated together at `K_decode = 4`.

**This script does not score the frozen gates, and cannot.** The preregistered scorecard
comes from `u5_report.py` over the primary sessions, using the original `C_k8` only. What
this produces is a separate section, *Supplementary training-launch robustness*, which
may qualify how that scorecard is read and may not change it:

* the original `C_k8` is never replaced by `C_k8_r2`;
* the two C launches are never averaged, and the better one is never chosen;
* no threshold is derived from anything here.

It is a new script rather than a section added to `u5_report.py` for a mechanical reason:
`u5_launch.sh` invokes `u5_report.py` at the end of the primary run, so editing it while
that run is in progress would change the primary run's own output.

Every rule below was fixed in A4 before any U5 endpoint was evaluated.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

A_ARMS = ("A_k4@16000", "A_k4_r2@16000", "A_k4_r3@16000")
C_ORIGINAL = "C_k8@16000"
C_REPLICATE = "C_k8_r2@16000"
OTHER_ARMS = ("B_k6@16000", "D_curr@16000")
PRIMARY_BLOCK = 4

PRINCIPAL = "mean_accepted_prefix"
METRICS = ("mean_accepted_prefix", "tpf", "tok_s")
FIRST_SLOT = "+2"
DEEP_SLOTS = ("+3", "+4")
#: U5-2's frozen floor. Used here only as a descriptive reference for the mechanism
#: profile; nothing in this script scores U5-2.
U5_2_FLOOR = 0.022

CASE_TEXT = {
    1: "The C_k8 effect replicated across two independent training launches and both "
       "exceeded the measured A_k4 launch spread.",
    2: "The C_k8 effect is launch-sensitive; the original frozen U5 result may be real but "
       "is not robust to an independent C training launch.",
    3: "The assumption that A_k4 launch spread is representative of C_k8 training "
       "variability is not supported by this check.",
    4: "Independent C_k8 replication provides no evidence that the U4/U5 effect exceeds "
       "measured launch-to-launch training noise.",
    5: "C_k8 is highly launch-sensitive under this training setup; no stable positive "
       "horizon-training effect is established.",
}

LIMITS = (
    "Two C_k8 launches do not establish C's standard deviation, a confidence interval "
    "across training launches, heteroskedasticity, or a distribution of K=8 outcomes. "
    "This check answers only whether the C effect survives one independent rerun, and "
    "whether the difference between the two launches looks roughly compatible with the "
    "A_k4 launch spread."
)

TOK_S_CAVEAT = (
    "u5-eval records each prompt's tok/s as the best of three keyed repeats. Repeats "
    "decode identical tokens, so the selection is on timing jitter only, and it applies "
    "identically to every arm in this session."
)


# ----------------------------------------------------------------- extraction


def arm_metrics(entry: dict, block: int = PRIMARY_BLOCK) -> dict:
    """The A4 metrics for one arm, at `K_decode = block`, from the harness's own keys."""
    blob = entry["per_block"][f"K{block}"]
    free_running, teacher_forced = blob["free_running"], blob["teacher_forced"]
    per_prompt = [row["tokens_per_second"] for row in free_running["per_prompt"]]
    return {
        # U5-1's metric: the free-running mean accepted prefix the transfer table uses.
        "mean_accepted_prefix": float(free_running["mean_accepted_specs"]),
        "tpf": float(free_running["tokens_per_forward"]),
        # U5-4 is scored on the per-prompt mean, not the median.
        "tok_s": statistics.fmean(per_prompt),
        "agree_per_offset": {k: float(v) for k, v in teacher_forced["agree_per_offset"].items()},
    }


def band(values: list[float]) -> float:
    return max(values) - min(values)


# -------------------------------------------------------------- classification


def classify_c_replication(c1: float, c2: float, a_band: float) -> dict:
    """A4's five cases, each evaluated independently, plus the headline rule.

    "Exceeds the band" is strictly greater than `a_band`, in the positive direction:
    the claim under test is a positive transfer effect, so a large negative effect does
    not "exceed" anything.
    """
    exceeds1, exceeds2 = c1 > a_band, c2 > a_band
    conditions = {
        1: c1 > 0 and c2 > 0 and exceeds1 and exceeds2,
        2: exceeds1 != exceeds2,
        3: abs(c1 - c2) > a_band,
        4: not exceeds1 and not exceeds2,
        5: c1 * c2 < 0,
    }
    if conditions[5]:
        headline = 5
    elif conditions[1]:
        headline = 1
    elif conditions[2]:
        headline = 2
    else:
        headline = 4

    statements = [CASE_TEXT[headline]]
    statements += [CASE_TEXT[c] for c in (1, 2, 3, 4) if c != headline and conditions[c]]

    notes = []
    if a_band == 0.0:
        notes.append("A_band is zero, so every non-zero effect trivially exceeds it; the "
                     "classification is uninformative.")
    if c1 < -a_band and c2 < -a_band:
        notes.append("Both C launches fall below the A_k4 mean by more than the A_k4 "
                     "launch spread.")
    return {
        "c1_effect": c1,
        "c2_effect": c2,
        "a_band": a_band,
        "abs_c1_minus_c2": abs(c1 - c2),
        "c1_exceeds_band": exceeds1,
        "c2_exceeds_band": exceeds2,
        "conditions": {str(k): v for k, v in conditions.items()},
        "headline_case": headline,
        "statements": statements,
        "notes": notes,
    }


def mechanism_profile(deltas: dict, bands: dict, floor: float = U5_2_FLOOR) -> dict:
    """`deep`, `first_slot_only` or `none`, per A4.

    An offset moves only if it clears U5-2's floor *and* the A_k4 launch band at that
    offset. The second condition is the point: without it, an offset profile could be
    produced by training noise alone.
    """
    def moves(offset: str) -> bool:
        return offset in deltas and deltas[offset] >= floor and deltas[offset] > bands[offset]

    moving = [o for o in (FIRST_SLOT, *DEEP_SLOTS) if moves(o)]
    if any(o in DEEP_SLOTS for o in moving):
        label = "deep"
    elif FIRST_SLOT in moving:
        label = "first_slot_only"
    else:
        label = "none"
    return {"label": label, "moving_offsets": moving,
            "deltas": {k: deltas[k] for k in (FIRST_SLOT, *DEEP_SLOTS) if k in deltas}}


def replication_outcome(headline_case: int, label1: str, label2: str) -> dict:
    performance = headline_case == 1
    profile = label1 == label2 and label1 != "none"
    if performance and profile:
        text = f"Performance and mechanism profile both replicate (profile: {label1})."
        if label1 == "deep":
            text += " The deeper-offset profile replicates."
        code = "performance_and_profile"
    elif performance:
        text = (f"Performance replicates but the mechanism profile does not "
                f"({label1} vs {label2}).")
        code = "performance_only"
    elif profile:
        text = f"The mechanism profile replicates ({label1}) but performance does not."
        code = "profile_only"
    else:
        text = (f"Neither performance nor the mechanism profile replicates "
                f"({label1} vs {label2}).")
        code = "neither"
    return {"code": code, "performance_replicates": performance,
            "profile_replicates": profile, "text": text}


# -------------------------------------------------------------------- analysis


def _identical(replicate_check: dict | None, a: str, b: str) -> bool | None:
    if not replicate_check:
        return None
    arms = replicate_check.get("arms", {})
    if a not in arms or b not in arms or not arms[a].get("exists") or not arms[b].get("exists"):
        return None
    return arms[a]["adapter"]["combined"] == arms[b]["adapter"]["combined"]


def analyze(payload: dict, replicate_check: dict | None = None,
            provenance: dict | None = None) -> dict:
    if payload.get("primary_block") != PRIMARY_BLOCK:
        raise SystemExit(f"eval primary_block is {payload.get('primary_block')}, need "
                         f"{PRIMARY_BLOCK}; this is not the A4 session")
    by_name = {entry["arm"]: entry for entry in payload["arms"]}
    required = (*A_ARMS, C_ORIGINAL, C_REPLICATE, *OTHER_ARMS)
    missing = [name for name in required if name not in by_name]
    if missing:
        raise SystemExit(f"supplementary session is missing arms {missing}; A4 requires "
                         f"all seven endpoints in one session")

    metrics = {name: arm_metrics(by_name[name]) for name in required}
    backbone_ok = all(by_name[name].get("backbone_unchanged", False) for name in required)

    a_values = {m: [metrics[a][m] for a in A_ARMS] for m in METRICS}
    a_mean = {m: statistics.fmean(v) for m, v in a_values.items()}
    a_band = {m: band(v) for m, v in a_values.items()}
    offsets = sorted({o for a in A_ARMS for o in metrics[a]["agree_per_offset"]})
    a_offset_mean = {o: statistics.fmean(metrics[a]["agree_per_offset"][o] for a in A_ARMS)
                     for o in offsets}
    a_offset_band = {o: band([metrics[a]["agree_per_offset"][o] for a in A_ARMS])
                     for o in offsets}

    # A3: every non-control arm against the control's launch band. A3's label applies
    # only to arms that passed a frozen gate in the primary scorecard; that join is done
    # in the written report, not here, so this script never re-implements frozen scoring.
    a3 = []
    for name in (*OTHER_ARMS[:1], C_ORIGINAL, C_REPLICATE, *OTHER_ARMS[1:]):
        row = {"arm": name}
        for m in METRICS:
            effect = metrics[name][m] - a_mean[m]
            row[m] = {"value": metrics[name][m], "effect": effect,
                      "exceeds_band": effect > a_band[m]}
        a3.append(row)

    c_by_metric = {
        m: classify_c_replication(metrics[C_ORIGINAL][m] - a_mean[m],
                                  metrics[C_REPLICATE][m] - a_mean[m], a_band[m])
        for m in METRICS
    }
    headlines = {m: c_by_metric[m]["headline_case"] for m in METRICS}

    profiles = {}
    for label, name in (("C1", C_ORIGINAL), ("C2", C_REPLICATE)):
        deltas = {o: metrics[name]["agree_per_offset"][o] - a_offset_mean[o]
                  for o in offsets}
        profiles[label] = mechanism_profile(deltas, a_offset_band)

    outcome = replication_outcome(headlines[PRINCIPAL], profiles["C1"]["label"],
                                  profiles["C2"]["label"])

    degeneracy = {
        "a_launches_identical": (
            None if not replicate_check else all(
                _identical(replicate_check, "A_k4", other) for other in ("A_k4_r2", "A_k4_r3"))
        ),
        "c_launches_identical": _identical(replicate_check, "C_k8", "C_k8_r2"),
    }
    notes = []
    if degeneracy["a_launches_identical"]:
        notes.append("The three A_k4 adapters are tensor-identical: A1 did not generalise to "
                     "a full run, A_band is zero by construction, and the classification "
                     "is uninformative.")
    if degeneracy["c_launches_identical"]:
        notes.append("C_k8 and C_k8_r2 are tensor-identical: this check says nothing about "
                     "C launch sensitivity.")

    valid = True
    if provenance is not None:
        record = provenance.get("replicates", {}).get("C_k8_r2", {})
        valid = bool(record.get("identical_launch", False))
        if not valid:
            notes.append("C_k8_r2 FAILED its provenance identity checks and is NOT A VALID "
                         "REPLICATE. The classification below is recorded but must not be "
                         "interpreted.")
    else:
        notes.append("No provenance record was supplied; launch identity is unverified.")

    if len(set(headlines.values())) > 1:
        notes.append(f"The metrics disagree on the case ({headlines}). The disagreement is "
                     f"reported, not resolved in favour of any metric.")

    return {
        "session": {"eval_file": payload.get("provenance", {}).get("stage", "u5-eval"),
                    "primary_block": PRIMARY_BLOCK,
                    "harness": payload.get("harness"),
                    "backbone_unchanged_for_all_arms": backbone_ok},
        "frozen_scoring": {
            "c_arm_for_frozen_gates": "C_k8 (original), primary sessions",
            "statement": "Not part of the preregistered scorecard. C_k8_r2 is not "
                         "substituted for C_k8, the C launches are not averaged, and no "
                         "frozen gate is rescored.",
        },
        "principal_metric": PRINCIPAL,
        "a_launches": {m: {"values": dict(zip(A_ARMS, a_values[m])), "mean": a_mean[m],
                           "band": a_band[m]} for m in METRICS},
        "a_offsets": {o: {"mean": a_offset_mean[o], "band": a_offset_band[o]}
                      for o in offsets},
        "a3_effects": a3,
        "c_replication": c_by_metric,
        "headline_case": headlines[PRINCIPAL],
        "headline_statements": c_by_metric[PRINCIPAL]["statements"],
        "mechanism_profiles": profiles,
        "replication_outcome": outcome,
        "valid_replicate": valid,
        "degeneracy": degeneracy,
        "notes": notes,
        "limits": LIMITS,
        "tok_s_caveat": TOK_S_CAVEAT,
    }


# ---------------------------------------------------------------------- output


def render_markdown(result: dict) -> str:
    lines = ["## Supplementary training-launch robustness", "",
             f"*{result['frozen_scoring']['statement']}* Criteria §11 A3–A4.", ""]
    if not result["valid_replicate"]:
        lines += ["> **C_k8_r2 is not a valid replicate.** See notes.", ""]
    lines += ["**Headline (principal metric: mean accepted prefix).**", ""]
    lines += [f"> {s}" for s in result["headline_statements"]]
    lines += ["", f"**Replication outcome.** {result['replication_outcome']['text']}", ""]

    lines += ["| metric | A_k4 launches | A mean | A band | C1 effect | C2 effect | "
              "\\|C1 − C2\\| | case |", "|---|---|---|---|---|---|---|---|"]
    for m, c in result["c_replication"].items():
        values = ", ".join(f"{v:.4f}" for v in result["a_launches"][m]["values"].values())
        lines.append(
            f"| {m} | {values} | {result['a_launches'][m]['mean']:.4f} | "
            f"{c['a_band']:.4f} | {c['c1_effect']:+.4f} | {c['c2_effect']:+.4f} | "
            f"{c['abs_c1_minus_c2']:.4f} | {c['headline_case']} |")

    lines += ["", "| arm | Δ prefix vs A mean | > band | Δ TPF | > band | Δ tok/s | > band |",
              "|---|---|---|---|---|---|---|"]
    for row in result["a3_effects"]:
        cells = [f"{row[m]['effect']:+.4f} | {'yes' if row[m]['exceeds_band'] else 'no'}"
                 for m in METRICS]
        lines.append(f"| {row['arm']} | " + " | ".join(cells) + " |")

    lines += ["", "| launch | Δ +2 | Δ +3 | Δ +4 | profile |", "|---|---|---|---|---|"]
    for label, profile in result["mechanism_profiles"].items():
        d = profile["deltas"]
        lines.append(f"| {label} | " + " | ".join(
            f"{d[o]:+.4f}" if o in d else "—" for o in (FIRST_SLOT, *DEEP_SLOTS))
            + f" | {profile['label']} |")

    if result["notes"]:
        lines += ["", "**Notes.**", ""] + [f"* {n}" for n in result["notes"]]
    lines += ["", f"**Limits.** {result['limits']}", "", f"*{result['tok_s_caveat']}*", ""]
    return "\n".join(lines)


def _load(path: str | None) -> dict | None:
    if not path:
        return None
    p = REPO / path
    return json.loads(p.read_text()) if p.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Act IV-U5 supplementary robustness (A3/A4)")
    parser.add_argument("--eval", default="runs/u5/eval_replicates.json")
    parser.add_argument("--replicate-check", default="results/act4u5/u5_replicate_check.json")
    parser.add_argument("--provenance", default="results/act4u5/u5_replicate_provenance.json")
    parser.add_argument("--out-json", default="results/act4u5/u5_supplementary_robustness.json")
    parser.add_argument("--out-md", default="results/act4u5/u5_supplementary_robustness.md")
    args = parser.parse_args(argv)

    payload = _load(args.eval)
    if payload is None:
        print(f"[u5-supp] {args.eval} not found", file=sys.stderr)
        return 1
    result = analyze(payload, _load(args.replicate_check), _load(args.provenance))

    for path, text in ((args.out_json, json.dumps(result, indent=2) + "\n"),
                       (args.out_md, render_markdown(result))):
        out = REPO / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"[u5-supp] wrote {out}")
    print("\n".join(f"  {s}" for s in result["headline_statements"]))
    print(f"  {result['replication_outcome']['text']}")
    for note in result["notes"]:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
