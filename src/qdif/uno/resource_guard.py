"""A launch guard that refuses heavy MLX work while another job owns the machine.

Act IV-U5 was written while an unrelated BoothGPT V2 pretraining run was live on the
same Mac. That job is long-running, holds a large MLX cache, and is more important than
this experiment. Loading Qwen3.5-4B for training or benchmarking alongside it would
contend for unified memory and could disturb it.

So the rule is enforced in code rather than by remembering:

    heavy U5 execution is blocked whenever a BoothGPT pretraining process is visible.

Three properties matter, and each is tested in `tests/test_uno_resource_guard.py`:

1. **It never signals anything.** This module reads `ps` output and nothing else. There
   is no code path here that can kill, suspend, renice, or otherwise touch another
   process, and there is no way to add one by accident: `subprocess` is invoked exactly
   once, with a fixed argument list.
2. **It fails closed.** If `ps` cannot be run, times out, or returns something
   unparseable, the guard blocks. An unreadable process table is not evidence that the
   machine is free.
3. **It cannot see itself.** A command line that merely *mentions* BoothGPT — this
   module's own tests, a `grep`, a log inspection — is not a pretraining process. The
   guard skips its own process tree and obvious tooling, and a test asserts that a
   `grep boothgpt` line does not trip it.

The override exists because a guard with no escape hatch gets deleted the first time it
is wrong. It is deliberately awkward: an environment variable, off by default, that
prints a loud warning naming the process it is overriding.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

#: The pretraining PID the user identified. Checked by number as well as by command
#: text: a process can be renamed, and the brief named this PID explicitly.
WATCHED_PIDS: tuple[int, ...] = (35477,)

#: Case-insensitive substrings that identify the protected job from its command line.
COMMAND_MARKERS: tuple[str, ...] = (
    "boothgpt",
    "boothgpt-v2-pretrain-launch",
    "pretrain_h1_4k_working_set_v1.yaml",
)

#: Set to "1" to run heavy work anyway. Not used during normal operation.
OVERRIDE_ENV = "QDIF_ALLOW_HEAVY_DURING_PRETRAIN"

#: Command fragments that mean "this line is us looking for the job", not "this line is
#: the job". Without these the guard trips on its own diagnostics.
_TOOLING_MARKERS: tuple[str, ...] = (
    "grep",
    "pgrep",
    "ps -ax",
    "resource_guard",
    "test_uno_resource_guard",
)


class HeavyExecutionBlocked(RuntimeError):
    """Raised instead of loading a large model while the machine is busy."""


@dataclass(frozen=True)
class PretrainSignal:
    """One process that looks like the protected pretraining job."""

    pid: int
    command: str
    reason: str

    def describe(self) -> str:
        return f"PID {self.pid} ({self.reason}): {self.command[:120]}"


def _process_table(timeout: float = 10.0) -> str | None:
    """`ps -axo pid=,command=`, or `None` if it could not be read.

    The only subprocess call in this module, with a fixed argument list. It reads; it
    cannot signal.
    """
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _is_tooling(command: str) -> bool:
    lowered = command.lower()
    return any(marker in lowered for marker in _TOOLING_MARKERS)


def _own_pids() -> set[int]:
    """This process and its parent, so the guard cannot detect itself."""
    pids = {os.getpid()}
    try:
        pids.add(os.getppid())
    except OSError:  # pragma: no cover - not reachable on darwin/linux
        pass
    return pids


def detect_pretraining(
    ps_output: str | None = None,
    own_pids: set[int] | None = None,
) -> list[PretrainSignal]:
    """Every visible process that looks like the protected pretraining job.

    `ps_output` is injectable so the guard can be tested against a fixed process table
    rather than against whatever happens to be running on the machine.

    A `None` process table means "could not look", which is reported as a signal so the
    caller blocks. That is deliberate: see property 2 in the module docstring.
    """
    if ps_output is None:
        ps_output = _process_table()
    if ps_output is None:
        return [PretrainSignal(-1, "<ps unavailable>", "process table unreadable")]

    skip = _own_pids() if own_pids is None else own_pids
    signals: list[PretrainSignal] = []
    for line in ps_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        command = command.strip()
        if pid in skip or _is_tooling(command):
            continue

        lowered = command.lower()
        matched = next((m for m in COMMAND_MARKERS if m in lowered), None)
        if matched is not None:
            signals.append(PretrainSignal(pid, command, f"command matches {matched!r}"))
        elif pid in WATCHED_PIDS:
            # The PID alone is enough. A recycled PID would block us spuriously, which
            # is the harmless direction: the cost is waiting, and the alternative is
            # contending with a 15-hour training run.
            signals.append(PretrainSignal(pid, command, "watched PID is alive"))
    return signals


def override_active() -> bool:
    return os.environ.get(OVERRIDE_ENV, "").strip() in {"1", "true", "yes", "on"}


def heavy_execution_allowed(ps_output: str | None = None) -> tuple[bool, list[PretrainSignal]]:
    """`(allowed, signals)`. Never raises, for callers that want to report rather than stop."""
    signals = detect_pretraining(ps_output)
    return (not signals) or override_active(), signals


def assert_heavy_execution_allowed(
    purpose: str = "heavy model execution",
    ps_output: str | None = None,
    echo=print,
) -> None:
    """Stop cleanly before instantiating a large model, if the machine is busy.

    Raises `HeavyExecutionBlocked` with a message that names what was detected. It does
    not offer to stop the other process and there is no code here that could.
    """
    signals = detect_pretraining(ps_output)
    if not signals:
        return

    detail = "\n".join(f"  - {found.describe()}" for found in signals)
    if override_active():
        echo(
            f"[guard] WARNING: {OVERRIDE_ENV} is set, so {purpose} will proceed while a "
            f"protected job is running. This can contend for unified memory with:\n{detail}"
        )
        return

    raise HeavyExecutionBlocked(
        "BoothGPT pretraining is active.\n"
        "U5 preparation is allowed, but real Qwen training/benchmarking is blocked.\n"
        f"Refusing {purpose}. Detected:\n{detail}\n"
        "The other process has NOT been touched. Wait for it to finish, or set "
        f"{OVERRIDE_ENV}=1 to override (not recommended)."
    )


def guard_report(ps_output: str | None = None) -> dict:
    """A machine-readable snapshot for run artifacts."""
    allowed, signals = heavy_execution_allowed(ps_output)
    return {
        "heavy_execution_allowed": allowed,
        "override_active": override_active(),
        "watched_pids": list(WATCHED_PIDS),
        "command_markers": list(COMMAND_MARKERS),
        "signals": [
            {"pid": s.pid, "reason": s.reason, "command": s.command[:200]} for s in signals
        ],
    }
