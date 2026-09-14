"""The paired bootstrap that scores Act IV-U5's gates.

U5 is looking for effects of a few percent in a quantity whose *unpaired* spread is
much larger than that: criteria sec.2.1 measured a between-session sd of 2.36 tok/s at
K=4 against a within-unit repeat noise of 0.15. The difference is prompt difficulty and
the draft-noise realisation, both of which cancel when the same prompt with the same
noise stream is compared across arms.

So the statistic has to be paired, and it has to be *correct*. These tests check it
against cases with known answers, including two that would let a bad implementation
declare a win: an all-zero effect and a large-but-inconsistent one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("uno_cli", REPO / "scripts" / "uno.py")
uno_cli = importlib.util.module_from_spec(_spec)
sys.modules["uno_cli"] = uno_cli
_spec.loader.exec_module(uno_cli)

paired_bootstrap = uno_cli._paired_bootstrap


def test_a_constant_shift_is_detected_exactly():
    pairs = [(10.0, 11.0)] * 15
    result = paired_bootstrap(pairs, resamples=2000)
    assert result["n"] == 15
    assert result["mean_diff"] == pytest.approx(1.0)
    assert result["ci_low"] == pytest.approx(1.0)
    assert result["ci_high"] == pytest.approx(1.0)
    assert result["significant"]
    assert result["wins"] == 15 and result["losses"] == 0


def test_no_effect_is_not_significant():
    pairs = [(10.0 + i, 10.0 + i) for i in range(15)]
    result = paired_bootstrap(pairs, resamples=2000)
    assert result["mean_diff"] == 0.0
    assert not result["significant"]


def test_large_but_inconsistent_differences_are_not_significant():
    """The case a naive 'mean went up' test gets wrong.

    Half the prompts improve by 10 and half worsen by 10. The mean difference is tiny
    but the spread is enormous; a paired interval must straddle zero.
    """
    pairs = [(10.0, 20.0) if i % 2 else (10.0, 0.0) for i in range(16)]
    result = paired_bootstrap(pairs, resamples=4000)
    assert abs(result["mean_diff"]) < 1e-9
    assert not result["significant"]
    assert result["ci_low"] < 0 < result["ci_high"]
    assert result["wins"] == 8 and result["losses"] == 8


def test_pairing_beats_the_unpaired_view():
    """A consistent +1 hidden inside a spread of 100 must still be detected.

    Unpaired, these two arms are indistinguishable. Paired, the effect is exact. This
    is the entire reason U5 measures per prompt rather than per arm.
    """
    controls = [float(10 * i) for i in range(15)]
    pairs = [(c, c + 1.0) for c in controls]
    result = paired_bootstrap(pairs, resamples=4000)
    assert result["mean_diff"] == pytest.approx(1.0)
    assert result["significant"]

    unpaired_spread = (
        sum((c - sum(controls) / len(controls)) ** 2 for c in controls) / (len(controls) - 1)
    ) ** 0.5
    assert unpaired_spread > 40  # ...and the effect is 1.0


def test_a_negative_effect_is_detected_as_negative():
    result = paired_bootstrap([(10.0, 9.0)] * 12, resamples=2000)
    assert result["mean_diff"] == pytest.approx(-1.0)
    assert result["significant"]
    assert result["ci_high"] < 0
    assert result["losses"] == 12


def test_interval_brackets_the_mean():
    pairs = [(1.0, 1.0 + 0.1 * i) for i in range(20)]
    result = paired_bootstrap(pairs, resamples=4000)
    assert result["ci_low"] <= result["mean_diff"] <= result["ci_high"]


def test_paired_se_matches_the_closed_form():
    diffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    pairs = [(0.0, d) for d in diffs]
    result = paired_bootstrap(pairs, resamples=1000)
    mean = sum(diffs) / len(diffs)
    variance = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    assert result["paired_se"] == pytest.approx((variance / len(diffs)) ** 0.5, rel=1e-6)


def test_bootstrap_is_deterministic():
    pairs = [(1.0, 1.0 + 0.3 * (i % 5)) for i in range(20)]
    assert paired_bootstrap(pairs, resamples=1500) == paired_bootstrap(pairs, resamples=1500)


def test_degenerate_inputs_do_not_crash():
    assert paired_bootstrap([])["n"] == 0
    single = paired_bootstrap([(1.0, 2.0)])
    assert single["n"] == 1 and single["mean_diff"] == pytest.approx(1.0)
    assert not single["significant"]


def test_p_positive_reports_the_direction():
    assert paired_bootstrap([(0.0, 1.0)] * 10, resamples=2000)["p_positive"] == 1.0
    assert paired_bootstrap([(0.0, -1.0)] * 10, resamples=2000)["p_positive"] == 0.0
