"""Act IV-U6 scoring rules, pinned before the pilot and the decisive run.

These encode `docs/act4u6_preregistered_criteria.md` §4–§8. Editing a test here after a
U6 result exists would move a threshold after seeing the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import u6_report as rep  # noqa: E402


def fake_boot(pairs):
    """Deterministic stand-in: the CI is mean ± 0.1 × (mean |diff| + 1) of the differences."""
    diffs = [t - c for c, t in pairs]
    m = sum(diffs) / len(diffs)
    w = 0.1 * (sum(abs(d) for d in diffs) / len(diffs) + 1)
    return {"mean_diff": m, "ci_low": m - w, "ci_high": m + w, "significant": abs(m) > w}


# ------------------------------------------------------------------- floor §6


def test_floor_is_never_below_one_percent():
    rec = rep.floor_from_differences([50.0] * 10, [50.01] * 10, boot=fake_boot)
    assert rec["floor_percent"] == 1.0


def test_floor_rounds_up_to_the_next_half_percent():
    control = [50.0] * 10
    twin = [50.7] * 10  # mean diff 0.7, CI half-width 0.17 -> max |bound| 0.87 = 1.74%
    rec = rep.floor_from_differences(control, twin, boot=fake_boot)
    assert rec["raw_percent"] == pytest.approx(1.74)
    assert rec["floor_percent"] == 2.0


# --------------------------------------------------------------- pilot kill §5


def _arm(pct, lo, hi):
    return {"tok_s": {"percent": pct, "ci_percent": [lo, hi]}}


def test_pilot_kills_only_when_every_decoupled_arm_is_clearly_below():
    arms = {"staged_d8v2": _arm(-9, -12, -6), "staged_d6v4": _arm(-3, -5, -1.5),
            "dec_d4v4": _arm(0, -1, 1)}
    assert rep.pilot_kill(arms, floor=1.0)["kill"] is True
    arms["staged_d6v4"] = _arm(-1, -3, 0.5)
    assert rep.pilot_kill(arms, floor=1.0)["kill"] is False


def test_the_control_arm_never_decides_the_pilot():
    arms = {"staged_d8v2": _arm(-9, -12, -6), "dec_d4v4": _arm(+5, +3, +7)}
    assert rep.pilot_kill(arms, floor=1.0)["kill"] is True


# ------------------------------------------------------------ U6-4 / U6-5 §7


def _full(pct, lo, hi, clean_pct=1.0, beyond=0, committed_lo=-1.0, accepted_lo=-1.0,
          add_verify=0.0, add_draft=0.0):
    return {"tok_s": {"percent": pct, "ci_percent": [lo, hi], "units": 81},
            "tok_s_clean": {"percent": clean_pct, "ci_percent": [clean_pct - 1, clean_pct + 1], "units": 40},
            "beyond_one_ulp": beyond,
            "committed_per_cycle": {"mean_diff": 0.1, "ci": [committed_lo, 1.0]},
            "accepted_per_cycle": {"mean_diff": 0.1, "ci": [accepted_lo, 1.0]},
            "added_verify_ms_per_cycle": add_verify, "added_draft_ms_per_cycle": add_draft}


def test_a_pass_needs_ci_above_zero_and_the_floor():
    assert rep.score_arm(_full(2.5, 0.2, 4.0), floor=2.0)["passes"] is True
    assert rep.score_arm(_full(1.5, 0.2, 3.0), floor=2.0)["passes"] is False   # below floor
    assert rep.score_arm(_full(2.5, -0.1, 5.0), floor=2.0)["passes"] is False  # CI spans zero


def test_a_pass_that_reverses_on_clean_text_is_not_robust_and_cannot_win():
    s = rep.score_arm(_full(3.0, 1.0, 5.0, clean_pct=-0.5), floor=1.0)
    assert s["passes"] and not s["robust"] and not s["wins"]


def test_an_unexplained_divergence_blocks_a_win():
    s = rep.score_arm(_full(3.0, 1.0, 5.0, beyond=1), floor=1.0)
    assert s["passes"] and not s["lossless"] and not s["wins"]


# ---------------------------------------------------------------- verdict §8


def test_verdict_rule_1_staged_wins():
    v = rep.verdict({"staged_d8v4": _full(3, 1, 5)}, variant1_alive=False, floor=1.0)
    assert v["rule"] == 1 and v["verdict"] == "STAGED VERIFICATION WINS"


def test_verdict_rule_2_truncate_wins_only_without_a_staged_win():
    arms = {"staged_d8v4": _full(-2, -4, 0), "trunc_d8v4": _full(3, 1, 5)}
    assert rep.verdict(arms, variant1_alive=True, floor=1.0)["rule"] == 2


def test_verdict_rule_3_verify_cost_dominates():
    arms = {"staged_d8v4": _full(-2, -4, -0.5, committed_lo=0.02, add_verify=1.5, add_draft=1.3)}
    v = rep.verdict(arms, variant1_alive=False, floor=1.0)
    assert v["rule"] == 3 and v["verdict"] == "VERIFY COST DOMINATES"


def test_rule_3_needs_the_added_cost_to_be_mostly_verification():
    arms = {"staged_d8v4": _full(-2, -4, -0.5, committed_lo=0.02, accepted_lo=0.01,
                                 add_verify=0.5, add_draft=1.3)}
    assert rep.verdict(arms, variant1_alive=False, floor=1.0)["rule"] == 4


def test_verdict_rule_5_width_does_not_affect_useful_prefix():
    arms = {"staged_d8v4": _full(-2, -4, -0.5)}
    v = rep.verdict(arms, variant1_alive=False, floor=1.0)
    assert v["rule"] == 5 and v["verdict"] == "DRAFT WIDTH DOES NOT AFFECT USEFUL PREFIX"


def test_verdict_rule_4_when_variant_1_survived_but_nothing_won():
    arms = {"trunc_d8v4": _full(-1, -3, 0.5)}
    assert rep.verdict(arms, variant1_alive=True, floor=1.0)["rule"] == 4


def test_tpf_or_acceptance_alone_never_produces_a_win():
    arms = {"staged_d8v4": _full(-5, -7, -3, committed_lo=0.5, accepted_lo=0.5, add_verify=5, add_draft=1)}
    assert rep.verdict(arms, variant1_alive=False, floor=1.0)["verdict"] != "STAGED VERIFICATION WINS"


# ------------------------------------------------------------ U6-2 bound §4


def _cost(verify, draft, commit=0.3):
    return {"verify_ms_by_width": {str(w): {"median": v} for w, v in verify.items()},
            "commit_ms_by_verify_width": {str(w): {"median": commit} for w in verify},
            "draft_ms_by_width": {str(w): {"median": v} for w, v in draft.items()}}


def test_perfect_drafter_stage_widths():
    cost = _cost({w: 20 + w for w in range(2, 10)}, {4: 25, 6: 26, 8: 27})
    bound = rep.perfect_drafter_bound(cost)["arms"]
    assert bound["incumbent_d4v4"]["stage_widths"] == [5]
    assert bound["staged_d8v4"]["stage_widths"] == [5, 4]
    assert bound["staged_d8v2"]["stage_widths"] == [3, 3, 3]
    assert bound["staged_d6v4"]["stage_widths"] == [5, 2]
    assert bound["staged_d6v3"]["stage_widths"] == [4, 3]
    assert bound["staged_d8v2"]["tokens"] == 9


def test_staged_is_killed_only_if_even_a_perfect_drafter_cannot_win():
    cheap = _cost({w: 1 + 0.1 * w for w in range(2, 10)}, {4: 25, 6: 25, 8: 25})
    assert rep.perfect_drafter_bound(cheap)["staged_killed"] is False
    dear = _cost({w: 200 for w in range(2, 10)}, {4: 1, 6: 1, 8: 1})
    assert rep.perfect_drafter_bound(dear)["staged_killed"] is True
