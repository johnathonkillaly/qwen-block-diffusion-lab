"""Act IV-U5 report tooling, driven by a synthetic evaluation payload.

The point is to exercise the whole `u5-eval -> u5_report` path — table construction,
delta-vs-control arithmetic, transfer flattening and all nine plots — while the real
model is untouchable. If the reporting only ever runs for the first time on real
results, a shape bug costs a re-run of an eight-hour experiment.

The payload is fabricated with a *known* answer: arm C is built to be faster than the
control and arm B to be identical to it, so the derived columns can be checked against
values computed by hand rather than against whatever the code happens to produce.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("u5_report", REPO / "scripts" / "u5_report.py")
u5_report = importlib.util.module_from_spec(_spec)
sys.modules["u5_report"] = u5_report
_spec.loader.exec_module(u5_report)

PROMPTS = 15
ROWS = 24


def _block(k_decode: int, tok_s: float, tpf: float, prefix: float, agree: float) -> dict:
    offered = k_decode - 1
    # a histogram whose mean equals `prefix`, spread over the two nearest buckets
    low = int(prefix)
    frac = prefix - low
    hist = {str(m): 0 for m in range(k_decode)}
    hist[str(min(low, offered))] = int(round(100 * (1 - frac)))
    hist[str(min(low + 1, offered))] = hist.get(str(min(low + 1, offered)), 0) + int(round(100 * frac))
    total = sum(hist.values()) or 1
    survival = {
        str(j): round(sum(v for m, v in hist.items() if int(m) >= j) / total, 5)
        for j in range(1, offered + 1)
    }
    return {
        "teacher_forced": {
            "agree_specs": agree,
            "agree_per_slot": [1.0] + [round(agree * (1 - 0.2 * i), 5) for i in range(offered)],
            "agree_per_offset": {"+1": 1.0, **{
                f"+{i + 2}": round(agree * (1 - 0.2 * i), 5) for i in range(offered)}},
            "seed_row_agreement": 1.0,
            "mean_accepted_prefix": prefix,
            "full_block_rate": 0.07,
            "validation_tv": 0.96,
            "rows": ROWS,
            "per_row_accepted_prefix": [prefix + 0.01 * (i % 5) for i in range(ROWS)],
            "per_row_agreement": [agree + 0.01 * (i % 3) for i in range(ROWS)],
        },
        "free_running": {
            "tokens_per_forward": tpf,
            "forwards_per_token": round(1 / tpf, 5),
            "replay_forwards": 0,
            "cycles": total,
            "median_tokens_per_second": tok_s,
            "mean_tokens_per_second": tok_s,
            "std_tokens_per_second": 5.0,
            "speedup_vs_ar": round(tok_s / 49.0, 4),
            "acceptance_histogram": hist,
            "mean_accepted_specs": prefix,
            "acceptance_rate": round(prefix / offered, 5),
            "full_block_rate": 0.07,
            "survival": {"accepted_specs": survival, "committed_tokens": {},
                         "mean_accepted_specs": prefix, "mean_committed_tokens": prefix + 2},
            "cycle_cost_ms": {"proposal_ms": 24.2, "verify_ms": 23.8, "commit_ms": 0.3,
                              "overhead_ms": 0.9, "total_ms": 49.2},
            "prefill_ms": 36.5,
            "predicted_tokens_per_second": tok_s + 1.0,
            "predicted_steady_state_tokens_per_second": tok_s + 2.0,
            "predicted_tpf": tpf,
            "ar_agreement": 0.86,
            "ceiling": {"block_size": k_decode, "max_speedup_vs_ar": 2.05},
            "speedup_requirements": [],
            "cost_model_error_pct": 1.8,
            "per_prompt": [
                {"domain": "prose", "prompt": f"p{i}",
                 "tokens_per_second": tok_s + i,          # prompt difficulty, shared
                 "tokens_per_forward": tpf, "mean_accepted_specs": prefix,
                 "cycles": 12, "ar_agreement": 0.86}
                for i in range(PROMPTS)
            ],
        },
    }


def _arm(name: str, tok_s: float, tpf: float, prefix: float, agree: float) -> dict:
    return {
        "arm": name, "path": f"runs/u5/{name}", "adapter_tensors": 256,
        "per_block": {f"K{k}": _block(k, tok_s - 3 * (k - 4), tpf, prefix, agree)
                      for k in (4, 8)},
        "draftability": {
            "block_size": 4, "positions": 512,
            "buckets": [{"bucket": b, "n": 100, "mean_gap": 1.5 + 0.1 * b,
                         "agreement": round(0.30 - 0.03 * b + (tok_s - 56) * 0.01, 5)}
                        for b in range(5)],
            "pearson_gap": -0.17, "pearson_entropy": -0.07,
            "by_offset": [{"offset": o, "n": 128, "mean_gap": 1.8, "mean_entropy": 2.0,
                           "agreement": 0.3, "pearson_gap": -0.1, "pearson_entropy": -0.1}
                          for o in (2, 3, 4)],
        },
        "peak_memory_gb": 11.6,
        "backbone_digest": "d" * 64,
        "backbone_unchanged": True,
    }


def _transfer(arm: str, mean_diff: float, significant: bool) -> dict:
    boot = {"n": PROMPTS, "mean_diff": mean_diff, "paired_se": 0.1,
            "ci_low": mean_diff - 0.4, "ci_high": mean_diff + 0.4,
            "p_positive": 1.0 if mean_diff > 0 else 0.0, "significant": significant,
            "wins": PROMPTS if mean_diff > 0 else 0,
            "losses": 0 if mean_diff > 0 else PROMPTS}
    return {
        "arm": arm, "control": "A_k4", "K_decode": 4,
        "tok_s": boot, "tpf": boot, "accepted_specs": boot,
        "tf_accepted_prefix": boot, "tf_agreement": boot,
        "delta_acceptance": round(mean_diff * 0.02, 5),
        "delta_tpf": round(mean_diff * 0.01, 5),
        "delta_tok_s_pct": round(100 * mean_diff / 56.0, 3),
        "delta_agreement_by_offset": {"+2": 0.02, "+3": 0.01, "+4": 0.005},
    }


def _payload() -> dict:
    return {
        "provenance": {"timestamp": "2026-09-09T12:00:00", "git_sha": "abc123",
                       "stage": "u5-eval"},
        "offset_convention": "future offset +j == supervised slot j-1",
        "primary_block": 4,
        "control_arm": "A_k4",
        "ar_baseline": {"median_tokens_per_second": 49.0, "mean_tokens_per_second": 48.6,
                        "std_tokens_per_second": 0.9, "rows": []},
        "harness": {"tokens": 48, "repeats": 3, "warmup": 1, "eval_rows": ROWS,
                    "prompts": PROMPTS, "window": 128, "seed": 20260903,
                    "noise_align_width": 8, "bootstrap": 1000},
        "backbone_digest": "d" * 64,
        "arms": [
            _arm("A_k4@16000", 56.0, 1.4035, 0.968, 0.3125),
            _arm("B_k6@16000", 56.0, 1.4035, 0.968, 0.3125),   # identical to control
            _arm("C_k8@16000", 59.0, 1.4314, 1.057, 0.3229),   # the hypothesised win
            _arm("D_curr@16000", 58.0, 1.4200, 1.020, 0.3200),
        ],
        "transfer": [
            _transfer("B_k6@16000", 0.0, False),
            _transfer("C_k8@16000", 3.0, True),
            _transfer("D_curr@16000", 2.0, True),
        ],
    }


@pytest.fixture
def written(tmp_path):
    payload = _payload()
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload))
    return payload, path


# ---------------------------------------------------------------------- rows


def test_rows_carry_the_arm_family_and_step(written):
    payload, _ = written
    rows = u5_report.build_rows([payload], {})
    families = {r["arm"] for r in rows}
    assert families == {"A_k4", "B_k6", "C_k8", "D_curr"}
    assert {r["step"] for r in rows} == {16000}


def test_primary_block_is_flagged(written):
    payload, _ = written
    rows = u5_report.build_rows([payload], {})
    assert {r["K_decode"] for r in rows if r["is_primary"]} == {4}
    assert any(not r["is_primary"] for r in rows)


def test_delta_vs_control_is_computed_against_the_named_control(written):
    payload, _ = written
    rows = u5_report.build_rows([payload], {})
    primary = {r["arm"]: r for r in rows if r["K_decode"] == 4}
    assert primary["A_k4"]["delta_tok_s_vs_control"] == 0.0
    assert primary["B_k6"]["delta_tok_s_vs_control"] == pytest.approx(0.0)
    assert primary["C_k8"]["delta_tok_s_vs_control"] == pytest.approx(3.0)
    assert primary["C_k8"]["delta_tok_s_pct_vs_control"] == pytest.approx(5.357, abs=1e-3)
    assert primary["C_k8"]["delta_prefix_vs_control"] == pytest.approx(0.089, abs=1e-6)


def test_offsets_use_the_u4_convention(written):
    payload, _ = written
    rows = u5_report.build_rows([payload], {})
    row = next(r for r in rows if r["arm"] == "C_k8" and r["K_decode"] == 4)
    assert row["agree_plus_1"] == 1.0  # seed row, adapter off
    assert row["agree_plus_4"] != ""
    assert row["agree_plus_5"] == ""   # does not exist at K_decode=4


def test_training_metadata_is_joined_and_arms_are_compute_matched(tmp_path, written):
    payload, _ = written
    meta = {}
    for arm, block, curriculum in (("A_k4", 4, []), ("B_k6", 6, []),
                                   ("C_k8", 8, []), ("D_curr", 4, [4, 6, 8])):
        directory = tmp_path / arm
        directory.mkdir()
        (directory / "result.json").write_text(json.dumps({
            "config": {"block_size": block, "block_curriculum": curriculum,
                       "steps": 16000, "batch_size": 4, "window": 128},
            "resume": {"start_step": 12800, "end_step": 16000},
            "wall_seconds": 4600.0,
        }))
        meta[arm] = directory / "result.json"

    training = u5_report.load_training(list(meta.values()))
    assert set(training) == {"A_k4", "B_k6", "C_k8", "D_curr"}
    # every arm pushes the same number of tokens through the model: both forwards are
    # `window - 1` wide whatever the block size. That is what makes the budget matched.
    assert len({v["tokens"] for v in training.values()}) == 1
    assert training["A_k4"]["tokens"] == 3200 * 4 * 127 * 2
    # ...but they differ in supervised positions, which is the treatment
    assert training["A_k4"]["supervised_positions"] == 3200 * 4 * 3
    assert training["C_k8"]["supervised_positions"] == 3200 * 4 * 7
    assert training["D_curr"]["k_train"] == "4/6/8"

    rows = u5_report.build_rows([payload], training)
    row = next(r for r in rows if r["arm"] == "C_k8" and r["K_decode"] == 4)
    assert row["training_steps"] == 3200
    assert row["K_train"] == 8
    assert row["prefix_gain_per_training_hour"] != ""


# -------------------------------------------------------------------- output


def test_tables_and_plots_are_written(tmp_path, written):
    payload, _ = written
    out = tmp_path / "results"
    rows = u5_report.build_rows([payload], {})
    u5_report.write_tables([payload], rows, out)
    u5_report.make_plots([payload], rows, out)

    for name in ("u5_arms.csv", "u5_transfer.csv", "u5_summary.json"):
        assert (out / name).is_file(), name
    plots = sorted(p.name for p in out.glob("plot*.png"))
    assert len(plots) == 9, plots
    assert all((out / p).stat().st_size > 1000 for p in plots)


def test_transfer_table_flattens_the_bootstrap(tmp_path, written):
    payload, _ = written
    out = tmp_path / "results"
    u5_report.write_tables([payload], u5_report.build_rows([payload], {}), out)
    import csv

    rows = list(csv.DictReader(open(out / "u5_transfer.csv")))
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["C_k8@16000"]["tok_s_significant"] == "True"
    assert by_arm["B_k6@16000"]["tok_s_significant"] == "False"
    assert float(by_arm["C_k8@16000"]["tok_s_mean_diff"]) == pytest.approx(3.0)
    assert by_arm["C_k8@16000"]["delta_agree_+2"] == "0.02"


def test_main_runs_end_to_end(tmp_path, written, capsys, monkeypatch):
    payload, path = written
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv",
                        ["u5_report.py", "--eval", str(path), "--out", str(out)])
    monkeypatch.setattr(u5_report, "REPO", Path("/"))
    assert u5_report.main() == 0
    captured = capsys.readouterr().out
    assert "C_k8@16000" in captured and "beating the control" in captured


def test_report_reports_no_winner_when_there_is_none(tmp_path, capsys, monkeypatch):
    payload = _payload()
    for blob in payload["transfer"]:
        blob["tok_s"]["significant"] = False
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(sys, "argv",
                        ["u5_report.py", "--eval", str(path), "--out", str(tmp_path / "o")])
    monkeypatch.setattr(u5_report, "REPO", Path("/"))
    assert u5_report.main() == 0
    assert "no arm beat the control" in capsys.readouterr().out
