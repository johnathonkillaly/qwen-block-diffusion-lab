"""The Act IV-U5 launch guard.

Every test here runs against an *injected* process table, so the suite behaves the same
whether or not a BoothGPT job is running on the machine at the time. The one test that
looks at the real process table only asserts that looking is safe and returns the right
shape.

The property that matters most is the one in `test_guard_never_signals_anything`: this
module must be incapable of touching the process it detects.
"""

from __future__ import annotations

import pytest

from qdif.uno.resource_guard import (
    COMMAND_MARKERS,
    OVERRIDE_ENV,
    WATCHED_PIDS,
    HeavyExecutionBlocked,
    assert_heavy_execution_allowed,
    detect_pretraining,
    guard_report,
    heavy_execution_allowed,
)

BUSY = (
    "  501 /usr/sbin/notifyd\n"
    "35477 /opt/homebrew/Cellar/python@3.11/.../Python train.py "
    "--config configs/v2/pretrain_h1_4k_working_set_v1.yaml\n"
    " 1234 /bin/zsh\n"
)

IDLE = "  501 /usr/sbin/notifyd\n 1234 /bin/zsh\n 9999 /usr/bin/vim notes.txt\n"


# ------------------------------------------------------------------- detection


def test_detects_the_pretraining_job():
    signals = detect_pretraining(BUSY, own_pids=set())
    assert len(signals) == 1
    assert signals[0].pid == 35477
    assert "pretrain_h1_4k_working_set_v1.yaml" in signals[0].reason


def test_idle_machine_is_allowed():
    assert detect_pretraining(IDLE, own_pids=set()) == []
    allowed, signals = heavy_execution_allowed(IDLE)
    assert allowed and not signals


def test_watched_pid_alone_is_enough():
    """A renamed or relaunched job must still be caught by PID."""
    table = " 501 /usr/sbin/notifyd\n35477 /some/other/python thing.py\n"
    signals = detect_pretraining(table, own_pids=set())
    assert len(signals) == 1
    assert signals[0].reason == "watched PID is alive"


def test_command_marker_alone_is_enough():
    """...and a job under a different PID must still be caught by its command."""
    table = " 777 python /Users/x/code/boothgpt-v2-pretrain-launch/train.py\n"
    signals = detect_pretraining(table, own_pids=set())
    assert len(signals) == 1
    assert signals[0].pid == 777


@pytest.mark.parametrize("marker", COMMAND_MARKERS)
def test_every_declared_marker_actually_matches(marker):
    signals = detect_pretraining(f" 888 python /path/{marker} --flag\n", own_pids=set())
    assert len(signals) == 1


def test_a_bare_train_py_is_not_enough():
    """Our own scripts must not trip the guard.

    `train.py` appears in this repo's own commands. Matching on it would make the guard
    permanently self-blocking, which is the failure mode that gets a guard deleted.
    """
    table = " 555 python scripts/uno.py train --block-size 4\n 556 python train.py\n"
    assert detect_pretraining(table, own_pids=set()) == []


# ------------------------------------------------------------------ self-blindness


def test_the_guard_does_not_detect_itself():
    table = f"{__import__('os').getpid()} python -c 'boothgpt'\n"
    assert detect_pretraining(table) == []


def test_a_grep_for_the_marker_is_not_the_job():
    table = " 601 grep -i boothgpt /var/log/system.log\n 602 pgrep -f boothgpt\n"
    assert detect_pretraining(table, own_pids=set()) == []


def test_the_guards_own_test_process_is_not_the_job():
    table = " 603 python -m pytest tests/test_uno_resource_guard.py\n"
    assert detect_pretraining(table, own_pids=set()) == []


# ----------------------------------------------------------------- fail closed


def test_unreadable_process_table_blocks():
    """Not being able to look is not evidence that the machine is free."""
    signals = detect_pretraining(None, own_pids=set()) if False else None
    # `None` means "run ps"; force the unreadable path explicitly instead.
    from qdif.uno import resource_guard

    original = resource_guard._process_table
    resource_guard._process_table = lambda timeout=10.0: None
    try:
        signals = resource_guard.detect_pretraining()
        assert len(signals) == 1
        assert signals[0].pid == -1
        with pytest.raises(HeavyExecutionBlocked, match="blocked"):
            resource_guard.assert_heavy_execution_allowed()
    finally:
        resource_guard._process_table = original


def test_malformed_lines_are_skipped_not_crashed():
    table = "not-a-pid whatever\n\n   \n35477 python train.py --config x.yaml\n"
    signals = detect_pretraining(table, own_pids=set())
    assert [s.pid for s in signals] == [35477]


# ---------------------------------------------------------------------- gating


def test_assert_raises_with_an_actionable_message():
    with pytest.raises(HeavyExecutionBlocked) as excinfo:
        assert_heavy_execution_allowed("loading Qwen3.5-4B", ps_output=BUSY)
    message = str(excinfo.value)
    assert "BoothGPT pretraining is active" in message
    assert "35477" in message
    assert "has NOT been touched" in message
    assert OVERRIDE_ENV in message


def test_assert_is_silent_when_the_machine_is_free():
    assert_heavy_execution_allowed("loading Qwen3.5-4B", ps_output=IDLE)


def test_override_allows_but_warns(monkeypatch):
    monkeypatch.setenv(OVERRIDE_ENV, "1")
    said = []
    assert_heavy_execution_allowed("benchmarking", ps_output=BUSY, echo=said.append)
    assert said and "WARNING" in said[0] and "35477" in said[0]


def test_override_is_off_by_default(monkeypatch):
    monkeypatch.delenv(OVERRIDE_ENV, raising=False)
    with pytest.raises(HeavyExecutionBlocked):
        assert_heavy_execution_allowed(ps_output=BUSY)


@pytest.mark.parametrize("value", ["0", "", "no", "false"])
def test_override_rejects_non_affirmative_values(monkeypatch, value):
    monkeypatch.setenv(OVERRIDE_ENV, value)
    with pytest.raises(HeavyExecutionBlocked):
        assert_heavy_execution_allowed(ps_output=BUSY)


# ----------------------------------------------------------------- no signalling


def test_guard_never_signals_anything():
    """The module must be structurally incapable of touching another process.

    Asserted on the source text rather than by behaviour, because the risk is a future
    edit adding a 'helpful' cleanup path, and no runtime test would catch that until it
    had already run.
    """
    import inspect
    import io
    import tokenize

    from qdif.uno import resource_guard

    source = inspect.getsource(resource_guard)

    # Scan executable tokens only. The module docstring *describes* the things it must
    # not do, so a plain substring search over the file would flag its own explanation.
    code_tokens = [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type not in (tokenize.COMMENT, tokenize.STRING)
    ]
    code = " ".join(code_tokens)

    # Whole tokens, not substrings: `signals` is a legitimate identifier here and would
    # collide with a substring search for `signal`.
    forbidden = {
        "kill", "killpg", "SIGKILL", "SIGTERM", "SIGSTOP", "SIGHUP",
        "terminate", "pkill", "killall", "renice", "signal", "system", "Popen",
    }
    used = forbidden & set(code_tokens)
    assert not used, f"resource_guard code must not reference {sorted(used)}"

    # exactly one subprocess invocation, and the string literal shows it is a read
    assert code.count("subprocess . run") == 1
    assert '["ps", "-axo", "pid=,command="]' in source


def test_report_is_json_shaped():
    report = guard_report(BUSY)
    assert report["heavy_execution_allowed"] is False
    assert report["watched_pids"] == list(WATCHED_PIDS)
    assert report["signals"][0]["pid"] == 35477

    free = guard_report(IDLE)
    assert free["heavy_execution_allowed"] is True
    assert free["signals"] == []


def test_reading_the_real_process_table_is_safe():
    """Looking at the machine must not raise, whatever is running on it."""
    allowed, signals = heavy_execution_allowed()
    assert isinstance(allowed, bool)
    assert all(hasattr(s, "describe") for s in signals)
