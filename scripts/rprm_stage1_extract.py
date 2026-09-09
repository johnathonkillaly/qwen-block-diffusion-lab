"""Stage 1 extraction: per-position denoiser uncertainty vs target acceptance.

Emits the machine-readable dataset pre-registered in `RPRM_DIFFUSION_PREREG.md` §3.
This script *measures*; it computes no correlation and renders no verdict. Analysis
lives in `scripts/rprm_stage1_analyze.py` so that the measurement cannot be quietly
tuned against the result.

Every quantity is taken from the existing Act IV-U code paths, unchanged:

    P_full    = model.ar_logits(teacher_ids)                  adapter OFF
    P_corrupt = model.draft_logits(student_ids, lora_mask)    adapter ZEROED
    P_student = model.draft_logits(student_ids, lora_mask)    adapter TRAINED

`P_full` and `P_corrupt` reproduce `cmd_draftability` exactly, so
`existing_draftability_score` here *is* the published draftability gap and
`teacher_entropy` here *is* the already-tested "predictor entropy" (audit §4, §5).
The new quantity is `P_student`'s distribution -- which the existing code computes
and then discards via `argmax`.

Two passes over the data rather than one, so the adapter is loaded once rather than
once per batch. Batches are rebuilt in pass B from the *same* PRNG keys, so the
student sees bit-identical noise to what the frozen pass measured. That equality is
asserted, not assumed.

Usage:
    HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/rprm_stage1_extract.py \
        --adapter runs/u4b2-k8/adapter.safetensors --block-size 8 --batches 320
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MODEL_ID = "unsloth/Qwen3.5-4B-Base"
STRATA = ("FULL", "SCHEDULE")

FIELDS = [
    "example_id", "stratum", "position", "offset", "timestep",
    "corruption_level", "n_corrupted_predecessors",
    "clean_token", "proposed_token", "verifier_token", "accepted",
    "entropy", "normalized_entropy", "top1_probability", "top1_top2_margin",
    "effective_support",
    "existing_draftability_score", "teacher_entropy", "corrupt_entropy",
]


def provenance(extra: dict | None = None) -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(args, cwd=REPO, text=True).strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    import mlx.core as mx

    record = {
        "git_sha": run("git", "rev-parse", "HEAD"),
        "git_dirty": bool(run("git", "status", "--porcelain")),
        "python": sys.version.split()[0],
        "mlx": getattr(mx, "__version__", "unknown"),
        "model": MODEL_ID,
        "stage": "rprm_stage1_extract",
    }
    record.update(extra or {})
    return record


def _dist_stats(logits, vocab_size):
    """Entropy, top-1 probability, top-1/top-2 margin and argmax for `[B, L, V]`."""
    import mlx.core as mx

    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    prob = mx.exp(logp)
    entropy = -mx.sum(prob * logp, axis=-1)
    top1 = mx.max(prob, axis=-1)
    index = mx.argmax(prob, axis=-1)
    # Second-largest probability: blank out the argmax, take the max again. Cheaper
    # and more memory-predictable than a full sort over a ~150k vocabulary.
    arange = mx.arange(vocab_size).reshape(1, 1, vocab_size)
    masked = mx.where(arange == index[:, :, None], mx.array(-1.0, dtype=prob.dtype), prob)
    top2 = mx.max(masked, axis=-1)
    return entropy, top1, top1 - top2, index, logp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="runs/u4b2-k8/adapter.safetensors")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--val-windows", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--tag", default="primary")
    parser.add_argument("--out-dir", default="results/rprm_stage1")
    args = parser.parse_args()

    import mlx.core as mx

    from qdif.mlx_backend.loader import resolve_mlx_model_path
    from qdif.uno.data import build_sources
    from qdif.uno.gated_lora import gated_lora_modules
    from qdif.uno.model import load_uno_model
    from qdif.uno.teacher import build_uno_batch
    from qdif.uno.trainer import load_adapter

    path = resolve_mlx_model_path(MODEL_ID, None)
    print(f"[rprm] checkpoint: {path}")
    model, _ = load_uno_model(path)
    model.attach_adapter(rank=args.rank, full_attention=True, mlp=True)
    vocab = int(model.vocab_size)
    print(f"[rprm] vocab={vocab}  block_size={args.block_size}  adapter={args.adapter}")

    pristine = {id(m): (m.lora_a, m.lora_b) for m in gated_lora_modules(model.base)}

    def reset_adapter() -> None:
        """Adapter back to its zero-init no-op: the frozen-model measurement state."""
        for module in gated_lora_modules(model.base):
            a, b = pristine[id(module)]
            module.lora_a, module.lora_b = a, mx.zeros_like(b)
        mx.eval(model.base.parameters())

    _, val_source = build_sources(
        model.tokenizer, width=args.window, batch_size=args.batch_size,
        train_windows=8, val_windows=args.val_windows, seed=args.seed,
        eos_id=model.tokenizer.eos_token_id,
    )
    # Draw the windows ONCE and reuse them for both strata, so the two strata differ
    # only in corruption and a window lands in the same split partition in both.
    windows = [next(val_source) for _ in range(args.batches)]
    print(f"[rprm] {len(windows)} batches x {args.batch_size} = "
          f"{len(windows) * args.batch_size} windows")

    # One key per (stratum, batch). Explicit keys only -- evaluation must consume no
    # global RNG state (pre-registered gate U3-0b).
    root = mx.random.key(args.seed)
    stratum_keys = dict(zip(STRATA, mx.random.split(root, len(STRATA))))
    keys = {s: list(mx.random.split(stratum_keys[s], len(windows))) for s in STRATA}

    def batch_for(stratum: str, index: int):
        return build_uno_batch(
            windows[index], block_size=args.block_size,
            mask_token_id=model.mask_token_id, vocab_size=vocab,
            corruption="full" if stratum == "FULL" else "uniform",
            key=keys[stratum][index],
        )

    rows: dict[tuple, dict] = {}

    # ---- pass A: frozen model. gap, teacher entropy, corrupt entropy, verifier token
    reset_adapter()
    print("[rprm] pass A (frozen model)")
    for stratum in STRATA:
        for index in range(len(windows)):
            batch = batch_for(stratum, index)
            sl = batch.supervised_slice
            full = mx.stop_gradient(model.ar_logits(batch.teacher_ids)[:, sl, :]).astype(mx.float32)
            corrupt = mx.stop_gradient(
                model.draft_logits(batch.student_ids, batch.lora_mask)[:, sl, :]
            ).astype(mx.float32)

            h_full, _, _, verifier, lf = _dist_stats(full, vocab)
            h_corrupt, _, _, _, lc = _dist_stats(corrupt, vocab)
            gap = mx.sum(mx.abs(mx.exp(lf) - mx.exp(lc)), axis=-1)

            gap_l = gap.tolist()
            hf_l, hc_l = h_full.tolist(), h_corrupt.tolist()
            ver_l = verifier.tolist()
            tgt_l = batch.targets.tolist()
            cor_l = batch.corrupted.tolist()
            rate_l = batch.corruption_rate.tolist()

            for row in range(batch.batch_size):
                for slot in range(1, args.block_size):  # slot 0 is the unadapted seed
                    rows[(stratum, index, row, slot)] = {
                        "example_id": f"w{index:04d}r{row}",
                        "stratum": stratum,
                        "position": batch.block_start + slot,
                        "offset": slot + 1,
                        "timestep": round(float(rate_l[row]), 6),
                        "corruption_level": int(cor_l[row][slot]),
                        "n_corrupted_predecessors": int(sum(cor_l[row][:slot])),
                        "clean_token": int(tgt_l[row][slot]),
                        "verifier_token": int(ver_l[row][slot]),
                        "existing_draftability_score": round(float(gap_l[row][slot]), 6),
                        "teacher_entropy": round(float(hf_l[row][slot]), 6),
                        "corrupt_entropy": round(float(hc_l[row][slot]), 6),
                    }
            if (index + 1) % 40 == 0:
                print(f"  {stratum} {index + 1}/{len(windows)}")

    # ---- pass B: trained adapter. the denoiser distribution itself
    loaded = load_adapter(model, Path(args.adapter))
    print(f"[rprm] pass B (trained adapter, {loaded} tensors)")
    log_vocab = float(mx.log(mx.array(float(vocab))).item())
    for stratum in STRATA:
        for index in range(len(windows)):
            batch = batch_for(stratum, index)
            sl = batch.supervised_slice
            student = mx.stop_gradient(
                model.draft_logits(batch.student_ids, batch.lora_mask)[:, sl, :]
            ).astype(mx.float32)
            h, top1, margin, proposed, _ = _dist_stats(student, vocab)

            h_l, t1_l, m_l = h.tolist(), top1.tolist(), margin.tolist()
            pro_l = proposed.tolist()
            tgt_l = batch.targets.tolist()

            for row in range(batch.batch_size):
                for slot in range(1, args.block_size):
                    record = rows[(stratum, index, row, slot)]
                    # The batch must be bit-identical to pass A's, or the frozen
                    # quantities describe different noise than the student saw.
                    if record["clean_token"] != int(tgt_l[row][slot]):
                        raise RuntimeError(
                            f"batch mismatch between passes at {stratum}/{index}/{row}/{slot}"
                        )
                    entropy = float(h_l[row][slot])
                    proposal = int(pro_l[row][slot])
                    record.update({
                        "proposed_token": proposal,
                        "accepted": int(proposal == record["verifier_token"]),
                        "entropy": round(entropy, 6),
                        "normalized_entropy": round(entropy / log_vocab, 6),
                        "top1_probability": round(float(t1_l[row][slot]), 6),
                        "top1_top2_margin": round(float(m_l[row][slot]), 6),
                        "effective_support": round(float(mx.exp(mx.array(entropy)).item()), 4),
                    })
            if (index + 1) % 40 == 0:
                print(f"  {stratum} {index + 1}/{len(windows)}")

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"positions_{args.tag}.csv"
    ordered = [rows[k] for k in sorted(rows)]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    print(f"[rprm] wrote {csv_path} ({len(ordered)} positions)")

    meta = {
        "provenance": provenance({"tag": args.tag}),
        "adapter": args.adapter,
        "block_size": args.block_size,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "window": args.window,
        "val_windows": args.val_windows,
        "seed": args.seed,
        "vocab_size": vocab,
        "log_vocab": log_vocab,
        "positions": len(ordered),
        "windows": len(windows) * args.batch_size,
        "positions_per_stratum": {
            s: sum(1 for r in ordered if r["stratum"] == s) for s in STRATA
        },
        "acceptance_rate_per_stratum": {
            s: round(
                sum(r["accepted"] for r in ordered if r["stratum"] == s)
                / max(1, sum(1 for r in ordered if r["stratum"] == s)), 6)
            for s in STRATA
        },
    }
    meta_path = out_dir / f"positions_{args.tag}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[rprm] wrote {meta_path}")
    print(json.dumps(meta["acceptance_rate_per_stratum"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
