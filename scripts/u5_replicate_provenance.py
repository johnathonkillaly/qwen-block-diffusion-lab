#!/usr/bin/env python
"""Prove each Act IV-U5 replicate launch was identical to its primary arm (A2, A4).

"Identical" means everything that defines the run matches, and the only permitted
difference is the launch-to-launch nondeterminism A1 observed. Checked here, from
artifacts rather than from trust:

* **start**: the checkpoint the replicate resumed from, hashed immediately before its
  launch, equals the record the primary run wrote at its own launch;
* **command line**: the primary arm's argv, captured read-only from the process table
  while it ran, equals the replicate's token for token, excluding `--out`;
* **process environment**: `HF_HOME` and `PYTHONPATH` of the primary training process,
  also captured from the process table, equal the replicate's;
* **code and packages**: digests of the training code and the Python/MLX versions are the
  same at orchestrator start, at primary exit, and before each replicate launch;
* **end state**: at step 16000, config hash, seed, block size, step, data position and
  RNG state match. Adapter and optimizer digests are reported as identical or diverged;
  divergence is allowed, not required.

Anything not captured counts as **not verified**, never as identical. Exits 4 if any
check fails, so the orchestrator logs it visibly. It never modifies what it reads, and it
imports `u5_start_check.describe` rather than copying it, so both use one hashing
implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

#: replicate -> the primary arm it replicates
REPLICATES = {"A_k4_r2": "A_k4", "A_k4_r3": "A_k4", "C_k8_r2": "C_k8"}
MUST_MATCH_AT_END = ("config_hash", "seed", "block_size", "step", "data_position", "rng")
START_FIELDS = ("step", "config_hash", "data_position")
#: Snapshot lines that identify or describe but must not be compared. `info git_head`
#: is here on purpose: a commit made mid-run changes HEAD without changing any code, and
#: the file digests are what prove the code is the same.
UNCOMPARED_PREFIXES = ("label ", "info ")


def normalized_argv(command: str) -> dict:
    """Split a captured command line into the interpreter and the arguments that define
    the run, dropping `--out <dir>`, which is the one argument a replicate must change."""
    tokens = command.split()
    index = next((i for i, t in enumerate(tokens) if t.endswith("scripts/uno.py")), None)
    if index is None:
        raise ValueError(f"not a scripts/uno.py command: {command[:80]!r}")
    interpreter = next((t for t in tokens[:index] if not t.startswith("-")), None)
    kept, skip = [], False
    for token in tokens[index:]:
        if skip:
            skip = False
            continue
        if token == "--out":
            skip = True
            continue
        kept.append(token)
    return {"interpreter": interpreter, "args": kept,
            "args_sha256": hashlib.sha256(" ".join(kept).encode()).hexdigest()}


def compare_argv(primary: str | None, replicate: str | None) -> dict:
    if not primary or not replicate:
        return {"checked": False, "identical": False,
                "reason": "primary command line not captured" if not primary
                else "replicate command line not recorded"}
    p, r = normalized_argv(primary), normalized_argv(replicate)
    differing = [(a, b) for a, b in zip(p["args"], r["args"]) if a != b]
    if len(p["args"]) != len(r["args"]):
        differing.append(("<length>", f"{len(p['args'])} vs {len(r['args'])}"))
    return {"checked": True,
            "identical": p["args"] == r["args"] and p["interpreter"] == r["interpreter"],
            "primary_args_sha256": p["args_sha256"], "replicate_args_sha256": r["args_sha256"],
            "interpreter_identical": p["interpreter"] == r["interpreter"],
            "differing_tokens": differing}


def compare_environ(primary: str | None, replicate: str | None) -> dict:
    if not primary or not replicate:
        return {"checked": False, "identical": False,
                "reason": "primary environment not captured" if not primary
                else "replicate environment not recorded"}
    p = sorted(l for l in primary.splitlines() if l.strip())
    r = sorted(l for l in replicate.splitlines() if l.strip())
    return {"checked": True, "identical": p == r, "primary": p, "replicate": r}


def compare_snapshots(texts: dict[str, str]) -> dict:
    """Code/package snapshots must match apart from their uncompared lines."""
    def body(text: str) -> str:
        return "\n".join(l for l in text.splitlines() if not l.startswith(UNCOMPARED_PREFIXES))
    bodies = {name: body(text) for name, text in texts.items()}
    distinct = set(bodies.values())
    return {"snapshots": sorted(texts), "identical": len(distinct) == 1 and bool(texts),
            "distinct_bodies": len(distinct)}


def compare_start(primary_start: dict, prelaunch_start: dict) -> dict:
    checks = {f: primary_start.get(f) == prelaunch_start.get(f) for f in START_FIELDS}
    for label in ("adapter", "optimizer"):
        for key in ("file_sha256", "combined"):
            value = (primary_start.get(label) or {}).get(key)
            checks[f"{label}_{key}"] = (
                value is not None and value == (prelaunch_start.get(label) or {}).get(key))
    return {"identical": all(checks.values()), "checks": checks,
            "adapter_file_sha256": (prelaunch_start.get("adapter") or {}).get("file_sha256"),
            "optimizer_file_sha256": (prelaunch_start.get("optimizer") or {}).get("file_sha256"),
            "config_hash": prelaunch_start.get("config_hash"),
            "step": prelaunch_start.get("step")}


def compare_end(primary_end: dict, replicate_end: dict) -> dict:
    exists = bool(primary_end.get("exists") and replicate_end.get("exists"))
    checks = {f: exists and primary_end.get(f) == replicate_end.get(f)
              for f in MUST_MATCH_AT_END}

    def digest(record, label):
        return (record.get(label) or {}).get("combined")

    return {
        "identical": all(checks.values()), "checks": checks,
        "config_hash": replicate_end.get("config_hash"),
        "primary_adapter_combined": digest(primary_end, "adapter"),
        "replicate_adapter_combined": digest(replicate_end, "adapter"),
        "adapter_diverged": digest(primary_end, "adapter") != digest(replicate_end, "adapter"),
        "optimizer_diverged": digest(primary_end, "optimizer") != digest(replicate_end, "optimizer"),
    }


def main(argv: list[str] | None = None) -> int:
    from u5_start_check import describe

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prov-dir", default="runs/u5/provenance")
    # The copy, not `u5_start_check.json`: u5_launch.sh rewrites that file after training,
    # and the comparison A4 promises is against the record made at the primary's launch.
    parser.add_argument("--primary-start-check",
                        default="results/act4u5/u5_start_check.at_primary_launch.json")
    parser.add_argument("--replicate", action="append", default=[],
                        help="replicate name; default: every known replicate")
    parser.add_argument("--out", default="results/act4u5/u5_replicate_provenance.json")
    args = parser.parse_args(argv)

    prov = REPO / args.prov_dir
    primary_start = json.loads((REPO / args.primary_start_check).read_text())["shared_start"]
    snapshots = {p.name: p.read_text() for p in sorted(prov.glob("env_*.txt"))}
    code = compare_snapshots(snapshots)

    def read(name: str) -> str | None:
        path = prov / name
        return path.read_text().strip() if path.is_file() else None

    report = {"code_and_packages": code, "replicates": {}}
    failed = not code["identical"]
    for name in args.replicate or list(REPLICATES):
        arm = REPLICATES[name]
        prelaunch_path = REPO / "results" / "act4u5" / f"u5_replicate_prelaunch_{name}.json"
        prelaunch = (json.loads(prelaunch_path.read_text())["shared_start"]
                     if prelaunch_path.is_file() else {})
        record = {
            "primary_arm": arm,
            "start": compare_start(primary_start, prelaunch) if prelaunch
            else {"identical": False, "reason": f"{prelaunch_path.name} missing"},
            "argv": compare_argv(read(f"primary_argv_{arm}.txt"),
                                 read(f"replicate_argv_{name}.txt")),
            "environ": compare_environ(read(f"primary_environ_{arm}.txt"),
                                       read(f"replicate_environ_{name}.txt")),
            "end": compare_end(describe(REPO / "runs" / "u5" / arm / "step-16000"),
                               describe(REPO / "runs" / "u5" / name / "step-16000")),
        }
        record["identical_launch"] = bool(
            record["start"]["identical"] and record["argv"]["identical"]
            and record["environ"]["identical"] and record["end"]["identical"]
            and code["identical"])
        failed |= not record["identical_launch"]
        report["replicates"][name] = record
        print(f"[u5-prov] {name} vs {arm}: start {record['start']['identical']}  "
              f"argv {record['argv']['identical']}  environ {record['environ']['identical']}  "
              f"end-state {record['end']['identical']}  code {code['identical']}  -> "
              f"{'IDENTICAL LAUNCH' if record['identical_launch'] else '!! NOT VERIFIED IDENTICAL'}; "
              f"adapter {'diverged' if record['end']['adapter_diverged'] else 'IDENTICAL'}")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"[u5-prov] wrote {out}")
    return 4 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
