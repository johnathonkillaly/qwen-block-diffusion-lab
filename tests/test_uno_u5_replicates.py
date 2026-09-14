"""Act IV-U5 supplementary robustness (criteria §11 A3–A4): the rules, not the plumbing.

These pin the interpretation A4 fixed before any U5 endpoint was evaluated. Changing a
test here after results exist would be moving a threshold after seeing the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import u5_replicate_provenance as prov  # noqa: E402
import u5_replicate_report as rep  # noqa: E402


# ------------------------------------------------------------ the five cases


def test_case_1_both_positive_and_both_beyond_band():
    out = rep.classify_c_replication(0.10, 0.08, 0.03)
    assert out["headline_case"] == 1
    assert out["conditions"]["3"] is False
    assert out["statements"] == [rep.CASE_TEXT[1]]


def test_case_2_exactly_one_beyond_band():
    out = rep.classify_c_replication(0.10, 0.01, 0.03)
    assert out["headline_case"] == 2


def test_case_3_is_reported_alongside_case_1_not_instead_of_it():
    """Both launches beat the control, yet differ by more than the A spread. The brief is
    explicit that this matters even when both C runs win."""
    out = rep.classify_c_replication(0.20, 0.05, 0.03)
    assert out["headline_case"] == 1
    assert out["conditions"]["3"] is True
    assert rep.CASE_TEXT[3] in out["statements"]


def test_case_4_neither_beyond_band():
    out = rep.classify_c_replication(0.02, 0.01, 0.03)
    assert out["headline_case"] == 4
    assert out["statements"][0] == rep.CASE_TEXT[4]


def test_case_5_opposite_signs_takes_the_headline_and_keeps_case_2_visible():
    out = rep.classify_c_replication(0.10, -0.02, 0.03)
    assert out["headline_case"] == 5
    assert out["statements"][0] == rep.CASE_TEXT[5]
    assert rep.CASE_TEXT[2] in out["statements"]


def test_equal_to_band_does_not_exceed_it():
    out = rep.classify_c_replication(0.03, 0.03, 0.03)
    assert out["c1_exceeds_band"] is False and out["headline_case"] == 4


def test_a_large_negative_effect_does_not_exceed_the_band():
    out = rep.classify_c_replication(-0.10, -0.12, 0.03)
    assert out["headline_case"] == 4
    assert any("below the A_k4 mean" in n for n in out["notes"])


def test_zero_band_is_flagged_as_uninformative():
    out = rep.classify_c_replication(0.01, 0.02, 0.0)
    assert any("uninformative" in n for n in out["notes"])


# ------------------------------------------------------------ mechanism profile


BANDS = {"+2": 0.005, "+3": 0.005, "+4": 0.005}


def test_deep_profile_needs_floor_and_band_at_a_deep_offset():
    assert rep.mechanism_profile({"+2": 0.0, "+3": 0.03, "+4": 0.0}, BANDS)["label"] == "deep"


def test_first_slot_only_profile():
    assert rep.mechanism_profile({"+2": 0.03, "+3": 0.01, "+4": 0.0},
                                 BANDS)["label"] == "first_slot_only"


def test_an_offset_inside_the_launch_band_does_not_move_even_above_the_floor():
    bands = {"+2": 0.005, "+3": 0.05, "+4": 0.005}
    assert rep.mechanism_profile({"+2": 0.0, "+3": 0.03, "+4": 0.0}, bands)["label"] == "none"


def test_floor_is_inclusive_as_in_u5_2():
    assert rep.mechanism_profile({"+2": 0.0, "+3": 0.022, "+4": 0.0}, BANDS)["label"] == "deep"


@pytest.mark.parametrize("case,l1,l2,code", [
    (1, "deep", "deep", "performance_and_profile"),
    (1, "deep", "first_slot_only", "performance_only"),
    (2, "first_slot_only", "first_slot_only", "profile_only"),
    (4, "deep", "none", "neither"),
    (1, "none", "none", "performance_only"),  # two empty profiles are not a replication
])
def test_replication_outcomes(case, l1, l2, code):
    assert rep.replication_outcome(case, l1, l2)["code"] == code


def test_deeper_offset_wording_only_for_a_deep_profile():
    assert "deeper-offset" in rep.replication_outcome(1, "deep", "deep")["text"]
    assert "deeper-offset" not in rep.replication_outcome(1, "first_slot_only",
                                                          "first_slot_only")["text"]


# ------------------------------------------------------------ end to end


def _entry(name, prefix, tpf, tok_s, offsets):
    return {
        "arm": name, "path": "", "backbone_unchanged": True,
        "per_block": {"K4": {
            "teacher_forced": {"agree_per_offset": {"+1": 1.0, **offsets}},
            "free_running": {"mean_accepted_specs": prefix, "tokens_per_forward": tpf,
                             "per_prompt": [{"tokens_per_second": t} for t in tok_s]},
        }},
    }


def _payload(c2_prefix=1.09, drop=()):
    flat = {"+2": 0.48, "+3": 0.27, "+4": 0.18}
    arms = [
        _entry("A_k4@16000", 1.00, 1.40, [55.0, 57.0], flat),
        _entry("A_k4_r2@16000", 1.02, 1.41, [55.5, 57.5], flat),
        _entry("A_k4_r3@16000", 1.04, 1.42, [56.0, 58.0], flat),
        _entry("B_k6@16000", 1.01, 1.40, [55.0, 57.0], flat),
        _entry("C_k8@16000", 1.10, 1.45, [59.0, 61.0], {"+2": 0.48, "+3": 0.31, "+4": 0.18}),
        _entry("C_k8_r2@16000", c2_prefix, 1.44, [58.5, 60.5],
               {"+2": 0.48, "+3": 0.30, "+4": 0.18}),
        _entry("D_curr@16000", 1.03, 1.41, [56.0, 58.0], flat),
    ]
    return {"primary_block": 4, "arms": [a for a in arms if a["arm"] not in drop],
            "harness": {}, "provenance": {}}


def test_analysis_uses_the_mean_of_three_controls_and_their_range():
    out = rep.analyze(_payload())
    assert out["a_launches"]["mean_accepted_prefix"]["mean"] == pytest.approx(1.02)
    assert out["a_launches"]["mean_accepted_prefix"]["band"] == pytest.approx(0.04)
    c = out["c_replication"]["mean_accepted_prefix"]
    assert c["c1_effect"] == pytest.approx(0.08) and c["c2_effect"] == pytest.approx(0.07)
    assert out["headline_case"] == 1
    assert out["replication_outcome"]["code"] == "performance_and_profile"


def test_c1_is_the_original_c_arm_and_is_never_averaged_with_the_replicate():
    """Changing only the replicate must leave C1 untouched: no pooled C anywhere."""
    base = rep.analyze(_payload(c2_prefix=1.09))["c_replication"]["mean_accepted_prefix"]
    moved = rep.analyze(_payload(c2_prefix=0.90))["c_replication"]["mean_accepted_prefix"]
    assert base["c1_effect"] == pytest.approx(moved["c1_effect"])
    assert rep.analyze(_payload())["frozen_scoring"]["c_arm_for_frozen_gates"].startswith(
        "C_k8 (original)")


def test_a_session_missing_any_endpoint_is_refused():
    with pytest.raises(SystemExit):
        rep.analyze(_payload(drop=("C_k8_r2@16000",)))


def test_tok_s_is_the_per_prompt_mean():
    out = rep.analyze(_payload())
    assert out["a_launches"]["tok_s"]["values"]["A_k4@16000"] == pytest.approx(56.0)


def test_an_invalid_provenance_record_is_flagged_not_hidden():
    bad = {"replicates": {"C_k8_r2": {"identical_launch": False}}}
    out = rep.analyze(_payload(), provenance=bad)
    assert out["valid_replicate"] is False
    assert "not a valid replicate" in rep.render_markdown(out)


def test_markdown_is_its_own_labelled_section_and_carries_the_limits():
    text = rep.render_markdown(rep.analyze(_payload()))
    assert text.startswith("## Supplementary training-launch robustness")
    assert "Not part of the preregistered scorecard" in text
    assert rep.LIMITS in text


# ------------------------------------------------------------ provenance


PY = "/Users/x/.venv-unsloth/bin/python"
BASE = (f"{PY} -u scripts/uno.py train --lr 1e-5 --rank 16 --seed 20260903 "
        f"--resume-from runs/u4a/step-12800 --steps 16000 --block-size 8")


def test_argv_identical_apart_from_out():
    out = prov.compare_argv(f"{BASE} --out runs/u5/C_k8", f"{BASE} --out runs/u5/C_k8_r2")
    assert out["identical"] is True
    assert out["primary_args_sha256"] == out["replicate_args_sha256"]


def test_argv_difference_is_caught_and_named():
    changed = BASE.replace("--block-size 8", "--block-size 4")
    out = prov.compare_argv(f"{BASE} --out runs/u5/C_k8", f"{changed} --out runs/u5/C_k8_r2")
    assert out["identical"] is False
    assert ("8", "4") in out["differing_tokens"]


def test_uncaptured_primary_argv_is_not_treated_as_identical():
    assert prov.compare_argv(None, f"{BASE} --out runs/u5/C_k8_r2")["identical"] is False


def test_snapshots_ignore_only_the_label_line():
    same = prov.compare_snapshots({"a": "label a\nmlx 0.32.1", "b": "label b\nmlx 0.32.1"})
    diff = prov.compare_snapshots({"a": "label a\nmlx 0.32.1", "b": "label b\nmlx 0.33.0"})
    assert same["identical"] is True and diff["identical"] is False


def test_environment_must_be_captured_and_equal():
    env = "HF_HOME=/Volumes/SHUTTLE\nPYTHONPATH=/repo/src"
    assert prov.compare_environ(env, env)["identical"] is True
    assert prov.compare_environ(env, "HF_HOME=/Volumes/SHUTTLE\nPYTHONPATH=/elsewhere/src"
                                )["identical"] is False
    assert prov.compare_environ(None, env)["identical"] is False


def test_git_head_is_informational_so_a_mid_run_commit_is_not_a_failure():
    out = prov.compare_snapshots({"a": "label a\ninfo git_head 111\nmlx 0.32.1",
                                  "b": "label b\ninfo git_head 222\nmlx 0.32.1"})
    assert out["identical"] is True


def test_end_state_divergence_is_allowed_but_config_mismatch_is_not():
    common = {"exists": True, "config_hash": "h", "seed": 1, "block_size": 8, "step": 16000,
              "data_position": {"train": 1}, "rng": {"s": 1}, "optimizer": {"combined": "o"}}
    diverged = prov.compare_end({**common, "adapter": {"combined": "x"}},
                                {**common, "adapter": {"combined": "y"}})
    assert diverged["identical"] is True and diverged["adapter_diverged"] is True
    mismatch = prov.compare_end({**common, "adapter": {"combined": "x"}},
                                {**common, "config_hash": "other", "adapter": {"combined": "x"}})
    assert mismatch["identical"] is False
