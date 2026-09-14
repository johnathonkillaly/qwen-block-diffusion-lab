#!/usr/bin/env python
"""Evidence that every Act IV-U5 arm began from byte-identical state.

Brief sec.9 asks for hashes of the shared starting point before training. This script
produces them without loading the model, so it runs safely while another job owns the
GPU, and writes them into `results/act4u5/` so the claim "the arms started identically"
is checkable from an artifact rather than taken on trust.

It hashes the raw file bytes *and* the parsed tensors. The file digest is what actually
matters — every arm passes `--resume-from` the same directory — but a per-tensor digest
survives a re-serialisation and is what a later arm-vs-arm comparison would need.

Run again after training with `--arms` to confirm the arms diverged (they must: a pair
of arms with identical *final* adapters would mean the horizon changed nothing, and is
far more likely to mean a configuration mistake).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digests(path: Path) -> dict:
    """Per-tensor SHA-256 over canonical bytes, plus a digest of the whole set.

    Uses the same width-matched integer view as `model.fingerprint`: a `memoryview` of
    a lazy MLX array, or a narrowing `.view(uint8)`, gives different digests for
    identical tensors. That bug cost Act IV-U a day.
    """
    import mlx.core as mx
    import numpy as np

    width_view = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}
    tensors = mx.load(str(path))
    per_tensor = {}
    for name in sorted(tensors):
        value = mx.contiguous(tensors[name])
        mx.eval(value)
        itemsize = value.nbytes // max(value.size, 1)
        raw = np.ascontiguousarray(np.array(value.view(width_view[itemsize]))).tobytes()
        per_tensor[name] = hashlib.sha256(raw).hexdigest()
    combined = hashlib.sha256(
        json.dumps(per_tensor, sort_keys=True).encode()
    ).hexdigest()
    return {"num_tensors": len(per_tensor), "combined": combined, "per_tensor": per_tensor}


def describe(directory: Path, with_tensors: bool = True) -> dict:
    record: dict = {"path": str(directory), "exists": directory.is_dir()}
    if not directory.is_dir():
        return record

    state_path = directory / "checkpoint_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        record["step"] = state.get("step")
        record["config_hash"] = state.get("config_hash")
        record["block_size"] = state.get("block_size")
        record["seed"] = state.get("seed")
        record["data_position"] = state.get("data_position")
        record["rng"] = state.get("rng")
        record["optimizer_tensors"] = state.get("optimizer_tensors")

    for label, filename in (("adapter", "adapter.safetensors"),
                            ("optimizer", "optimizer.safetensors")):
        path = directory / filename
        if not path.is_file():
            record[label] = None
            continue
        blob = {"file_sha256": file_digest(path), "bytes": path.stat().st_size}
        if with_tensors:
            blob.update(tensor_digests(path))
            blob.pop("per_tensor", None)
        record[label] = blob
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Act IV-U5 shared-start verification")
    parser.add_argument("--start", default="runs/u4a/step-12800",
                        help="the checkpoint every arm resumes from")
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=DIR",
                        help="post-training arm checkpoint; repeatable")
    parser.add_argument("--out", default="results/act4u5/u5_start_check.json")
    args = parser.parse_args()

    start = describe(REPO / args.start)
    if not start["exists"]:
        print(f"[u5] shared start {args.start} not found", file=sys.stderr)
        return 1

    print(f"[u5] shared start: {args.start}")
    print(f"     step              {start.get('step')}")
    print(f"     config hash       {start.get('config_hash')}")
    print(f"     adapter sha256    {start['adapter']['file_sha256']}")
    print(f"     adapter tensors   {start['adapter']['num_tensors']} "
          f"(combined {start['adapter']['combined'][:16]}…)")
    print(f"     optimizer sha256  {start['optimizer']['file_sha256']}")
    print(f"     optimizer tensors {start['optimizer']['num_tensors']}")
    position = (start.get("data_position") or {}).get("train") or {}
    print(f"     data position     {position.get('draws')} train draws, "
          f"cursor {position.get('cursor')}")

    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = describe(REPO / path)

    divergence = []
    if arms:
        print("\n[u5] post-training arms")
        for name, record in arms.items():
            if not record["exists"]:
                print(f"     {name:>8s}  MISSING ({record['path']})")
                continue
            same = record["adapter"]["combined"] == start["adapter"]["combined"]
            print(f"     {name:>8s}  step {record.get('step')}  "
                  f"adapter {record['adapter']['combined'][:16]}…  "
                  f"{'IDENTICAL TO START (!)' if same else 'diverged from start'}")
            divergence.append({"arm": name, "diverged_from_start": not same})
        # arms must also differ from each other
        digests = {n: r["adapter"]["combined"] for n, r in arms.items()
                   if r["exists"]}
        collisions = [
            (a, b) for i, a in enumerate(digests) for b in list(digests)[i + 1:]
            if digests[a] == digests[b]
        ]
        if collisions:
            print(f"     !! arms with identical adapters: {collisions} — "
                  f"the horizon cannot have been the experimental variable")

    payload = {"shared_start": start, "arms": arms, "divergence": divergence}
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n[u5] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
