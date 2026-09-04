#!/usr/bin/env python
"""Act IV-U runner: `stage0`, `baseline`, `overfit`, `train`, `bench`.

Every subcommand writes a JSON artifact under `runs/` with the git SHA, package
versions, seed and full config, so no run is a mystery run.

    HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py stage0
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MODEL_ID = "unsloth/Qwen3.5-4B-Base"


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
        "platform": platform.platform(),
        "mlx": getattr(mx, "__version__", "unknown"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL_ID,
    }
    if extra:
        record.update(extra)
    return record


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"[uno] wrote {path}")


def load_model(rank: int | None = None, **adapter_kwargs):
    from qdif.mlx_backend.loader import resolve_mlx_model_path
    from qdif.uno.model import load_uno_model

    path = resolve_mlx_model_path(MODEL_ID, None)
    print(f"[uno] checkpoint: {path}")
    model, report = load_uno_model(path)
    if rank is not None:
        model.attach_adapter(rank=rank, **adapter_kwargs)
        params = model.parameter_report()
        print(
            f"[uno] adapter r={rank} -> {model.lora_report.num_injected} modules, "
            f"{params['trainable_params']:,} trainable "
            f"({params['percent_trainable']:.4f}%)"
        )
    return model, report


# --------------------------------------------------------------------- stage 0


def cmd_stage0(args) -> int:
    """Implementation sanity on the real 4B model. Proves plumbing, not the idea."""
    import mlx.core as mx

    from qdif.uno.data import pack_windows, load_wikitext
    from qdif.uno.decode import ar_greedy_generate, uno_greedy_generate
    from qdif.uno.gated_lora import gated_lora_modules
    from qdif.uno.teacher import build_uno_batch, student_logits_for, teacher_logits_for
    from qdif.uno.verifier import greedy_accept, sequential_reference_accept_length

    checks: list[dict] = []

    def check(name: str, condition, detail="") -> None:
        ok = bool(condition)
        checks.append({"check": name, "pass": ok, "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    model, report = load_model(rank=args.rank, full_attention=True, mlp=True)
    kinds = model.layer_kinds()

    print("\n-- architecture --")
    check("4B-class backbone", report.total_params > 4e9, f"{report.total_params:,} params")
    check("8 full-attention layers", len(kinds["full_attention"]) == 8, kinds["full_attention"])
    check("24 Gated DeltaNet layers", len(kinds["gated_deltanet"]) == 24)
    check("tied embeddings", report.tie_word_embeddings)
    params = model.parameter_report()
    check(
        "adapter is lightweight (<1% of backbone)",
        params["percent_trainable"] < 1.0,
        f"{params['percent_trainable']:.4f}% / {params['trainable_params']:,} params",
    )

    print("\n-- freezing --")
    from mlx.utils import tree_flatten

    trainable = [n for n, _ in tree_flatten(model.base.trainable_parameters())]
    check("only lora_a/lora_b are trainable", all(n.endswith(("lora_a", "lora_b")) for n in trainable))
    fingerprint = model.fingerprint(full=True)
    check("backbone fingerprint computed", len(fingerprint.digest) == 64, fingerprint.digest[:16] + "…")

    print("\n-- adapter gating --")
    tokenizer = model.tokenizer
    windows = pack_windows(load_wikitext("validation"), tokenizer, args.window, 8)
    ids = mx.array(windows[:2].astype("int32"))
    bare = model.ar_logits(ids)
    zero_adapter = model.draft_logits(ids, mx.ones(ids.shape, dtype=mx.float32))
    check("zero-init adapter is a bit-exact no-op", mx.array_equal(bare, zero_adapter))

    mx.random.seed(0)
    for module in gated_lora_modules(model.base):
        module.lora_b = mx.random.normal(module.lora_b.shape, scale=0.02).astype(mx.float32)
    mx.eval(model.base.parameters())

    perturbed = model.draft_logits(ids, mx.ones(ids.shape, dtype=mx.float32))
    check("a non-zero adapter changes the output", not mx.array_equal(bare, perturbed))
    check(
        "an all-zero token mask restores the backbone exactly",
        mx.array_equal(bare, model.draft_logits(ids, mx.zeros(ids.shape, dtype=mx.float32))),
    )
    check("adapter never touches backbone weights", model.fingerprint().digest == fingerprint.digest)

    print("\n-- training layout --")
    batch = build_uno_batch(
        mx.array(windows[:4].astype("int32")),
        block_size=args.block_size,
        mask_token_id=model.mask_token_id,
        vocab_size=model.vocab_size,
        corruption="full",
    )
    teacher = teacher_logits_for(model, batch)
    student = student_logits_for(model, batch)
    check(
        "teacher equals the plain AR forward",
        mx.array_equal(teacher, model.ar_logits(batch.teacher_ids)[:, batch.supervised_slice, :]),
    )
    check(
        "seed row is bit-identical between student and teacher",
        mx.array_equal(student[:, 0, :], teacher[:, 0, :]),
        "the gate is not leaking onto clean rows",
    )
    check(
        "adapted rows genuinely differ from the teacher",
        not mx.array_equal(student[:, 1:, :], teacher[:, 1:, :]),
    )
    cut = batch.block_start + 1
    check(
        "student and teacher share prefix and seed",
        mx.array_equal(batch.student_ids[:, :cut], batch.teacher_ids[:, :cut]),
    )
    leaked = int(
        mx.sum((batch.student_ids[:, cut:] == batch.teacher_ids[:, cut:]).astype(mx.int32)).item()
    )
    check("draft rows do not contain the gold future", leaked <= 1, f"{leaked} coincidental matches")

    print("\n-- verifier --")
    context = mx.array(windows[0, : args.window // 2].astype("int32"))
    running = context
    block = []
    for _ in range(args.block_size):
        token = int(mx.argmax(model.ar_logits(running[None])[0, -1]).item())
        block.append(token)
        running = mx.concatenate([running, mx.array([token], dtype=context.dtype)])
    verify_logits = model.ar_logits(
        mx.concatenate([context, mx.array(block, dtype=context.dtype)])[None]
    )[0, -args.block_size :]
    fast = greedy_accept(block, verify_logits)
    check("a true AR block is fully accepted", fast.full_block, f"{fast.accepted_specs} specs")
    slow = sequential_reference_accept_length(model, context, block)
    check("parallel verifier matches the sequential reference", fast.accepted_specs == slow,
          f"parallel={fast.accepted_specs} sequential={slow}")

    corrupted = list(block)
    corrupted[max(1, len(corrupted) // 2)] = 12345
    fast_bad = greedy_accept(corrupted, model.ar_logits(
        mx.concatenate([context, mx.array(corrupted, dtype=context.dtype)])[None]
    )[0, -args.block_size :])
    slow_bad = sequential_reference_accept_length(model, context, corrupted)
    check("…and on a corrupted block too", fast_bad.accepted_specs == slow_bad,
          f"parallel={fast_bad.accepted_specs} sequential={slow_bad}")

    print("\n-- decoding --")
    prompt = [int(t) for t in windows[1, :24]]

    # Gate 3 is asserted in the cacheless regime, where the AR baseline and the Uno
    # decoder push identical token spans through the model and any difference must
    # therefore be algorithmic. See the chunking measurement below for why exact
    # equality is not a fair demand of the cached regime on this backbone.
    reference_uncached, _ = ar_greedy_generate(
        model, prompt, max_tokens=args.decode_tokens, use_cache=False
    )
    for block_size in (1, 2, 4):
        tokens, _ = uno_greedy_generate(
            model, prompt, max_tokens=args.decode_tokens,
            block_size=block_size, use_cache=False,
        )
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(tokens, reference_uncached)) if a != b), None
        )
        check(
            f"Uno decoding is exactly lossless at B={block_size} (reference regime)",
            tokens == reference_uncached,
            "identical" if first_diff is None else f"diverges at token {first_diff}",
        )

    def agreement(a, b):
        return sum(x == y for x, y in zip(a, b)) / max(len(a), 1)

    reference_cached, ar_stats = ar_greedy_generate(
        model, prompt, max_tokens=args.decode_tokens
    )
    chunking_floor = agreement(reference_uncached, reference_cached)
    check(
        "backbone chunking-invariance measured (not a pass/fail)",
        True,
        f"AR cached vs AR uncached agreement {chunking_floor:.3f} — "
        f"{'NOT invariant' if chunking_floor < 1.0 else 'invariant'}",
    )
    for block_size in (2, 4):
        cached, stats = uno_greedy_generate(
            model, prompt, max_tokens=args.decode_tokens, block_size=block_size
        )
        check(
            f"cached Uno tracks cached AR at least as well as AR tracks itself (B={block_size})",
            agreement(cached, reference_cached) >= chunking_floor - 1e-9,
            f"agreement {agreement(cached, reference_cached):.3f} vs floor "
            f"{chunking_floor:.3f}; {stats.forwards} forwards vs {ar_stats.forwards} AR, "
            f"TPF {stats.tokens_per_forward:.2f}",
        )

    check("backbone unchanged at the end of stage 0",
          model.fingerprint().digest == fingerprint.digest)

    passed = sum(1 for c in checks if c["pass"])
    print(f"\n[uno] stage 0: {passed}/{len(checks)} passed")
    write_json(
        Path(args.out),
        {
            "provenance": provenance({"stage": "stage0"}),
            "architecture": {"layer_kinds": kinds, "params": params},
            "backbone_digest": fingerprint.digest,
            "backbone_chunking_agreement": chunking_floor,
            "checks": checks,
            "passed": passed,
            "total": len(checks),
        },
    )
    return 0 if passed == len(checks) else 1


# -------------------------------------------------------------------- baseline


def cmd_baseline(args) -> int:
    """Measure the frozen AR model before anything is trained. Gate 0's anchor."""
    import mlx.core as mx

    from qdif.uno.data import load_wikitext, pack_windows, prompt_suite_tokens
    from qdif.uno.decode import ar_greedy_generate

    model, report = load_model()
    tokenizer = model.tokenizer
    fingerprint = model.fingerprint(full=True)

    print("\n-- held-out language modelling --")
    windows = pack_windows(load_wikitext("validation"), tokenizer, args.window, args.eval_windows)
    total_nll, total_tokens, correct = 0.0, 0, 0
    for start in range(0, len(windows), args.batch_size):
        chunk = mx.array(windows[start : start + args.batch_size].astype("int32"))
        logits = model.ar_logits(chunk).astype(mx.float32)
        targets = chunk[:, 1:]
        log_probs = logits[:, :-1] - mx.logsumexp(logits[:, :-1], axis=-1, keepdims=True)
        picked = mx.take_along_axis(log_probs, targets[..., None].astype(mx.int32), axis=-1)[..., 0]
        total_nll += float(-mx.sum(picked).item())
        correct += int(mx.sum((mx.argmax(logits[:, :-1], axis=-1) == targets).astype(mx.int32)).item())
        total_tokens += int(targets.size)
    nll = total_nll / total_tokens
    lm = {
        "windows": len(windows),
        "tokens": total_tokens,
        "nll": round(nll, 6),
        "perplexity": round(float(mx.exp(mx.array(nll)).item()), 4),
        "token_accuracy": round(correct / total_tokens, 6),
    }
    print(f"  NLL {lm['nll']:.4f}  PPL {lm['perplexity']:.3f}  acc {lm['token_accuracy']:.4f}")

    print("\n-- deterministic generation --")
    generations = []
    for domain, text, ids in prompt_suite_tokens(tokenizer):
        timings = []
        tokens = None
        for repeat in range(args.repeats):
            tokens, stats = ar_greedy_generate(model, ids, max_tokens=args.decode_tokens)
            timings.append(stats.to_dict())
        median = sorted(t["tokens_per_second"] for t in timings)[len(timings) // 2]
        generations.append(
            {
                "domain": domain,
                "prompt": text,
                "token_ids": tokens,
                "completion": tokenizer.decode(tokens),
                "forwards": timings[-1]["forwards"],
                "prefill_seconds": timings[-1]["prefill_seconds"],
                "tokens_per_second_median": median,
                "repeats": timings,
            }
        )
        print(f"  [{domain}] {median:6.2f} tok/s  {text[:42]!r}")

    speeds = sorted(g["tokens_per_second_median"] for g in generations)
    payload = {
        "provenance": provenance({"stage": "baseline"}),
        "backbone_digest": fingerprint.digest,
        "backbone_params": fingerprint.num_params,
        "architecture": {"layer_kinds": model.layer_kinds(), "load": report.to_dict()},
        "language_modelling": lm,
        "decode_tokens": args.decode_tokens,
        "tokens_per_second": {
            "median": speeds[len(speeds) // 2],
            "p10": speeds[max(0, int(0.1 * len(speeds)))],
            "p90": speeds[min(len(speeds) - 1, int(0.9 * len(speeds)))],
        },
        "generations": generations,
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
    }
    print(f"\n[uno] AR baseline median {payload['tokens_per_second']['median']:.2f} tok/s, "
          f"peak {payload['peak_memory_gb']:.2f} GB")
    write_json(Path(args.out), payload)
    return 0


# --------------------------------------------------------------------- overfit


def cmd_overfit(args) -> int:
    """Milestone B: can the objective drive one batch to (near) zero, and does the
    shuffled control fail to follow? A pass here is plumbing, not evidence."""
    import itertools

    import mlx.core as mx

    from qdif.uno.data import load_wikitext, pack_windows
    from qdif.uno.trainer import UnoTrainConfig, evaluate, train_uno

    results = {}
    for arm, shuffle in (("true", False), ("shuffled", True)):
        print(f"\n=== overfit arm: {arm} ===")
        model, _ = load_model()
        windows = pack_windows(
            load_wikitext("train"), model.tokenizer, args.window, args.batch_size
        )
        fixed = mx.array(windows.astype("int32"))

        def repeat():
            while True:
                yield fixed

        config = UnoTrainConfig(
            block_size=args.block_size,
            steps=args.steps,
            batch_size=args.batch_size,
            window=args.window,
            lora_rank=args.rank,
            learning_rate=args.lr,
            warmup_steps=5,
            corruption="full",
            shuffle_targets=shuffle,
            eval_every=max(1, args.steps // 4),
            eval_batches=1,
            log_every=max(1, args.steps // 10),
        )
        result = train_uno(model, repeat(), repeat(), config, echo=print)
        results[arm] = {
            "first_loss": result["history"][0]["loss"],
            "final_loss": result["history"][-1]["loss"],
            "min_loss": min(h["loss"] for h in result["history"]),
            "evals": result["evals"],
            "integrity": result["integrity"],
            "wall_seconds": result["wall_seconds"],
        }
        del model

    print("\n=== overfit summary ===")
    for arm, data in results.items():
        final_eval = data["evals"][-1] if data["evals"] else {}
        print(
            f"  {arm:9s} loss {data['first_loss']:.4f} -> {data['final_loss']:.4f} "
            f"| agree(specs) {final_eval.get('agree_specs', float('nan')):.3f} "
            f"| prefix {final_eval.get('mean_accepted_prefix', float('nan')):.2f}"
        )
    write_json(
        Path(args.out),
        {"provenance": provenance({"stage": "overfit"}), "arms": results},
    )
    return 0


# ----------------------------------------------------------------------- train


def cmd_train(args) -> int:
    """Milestone C/D: held-out training."""
    from qdif.uno.data import build_sources
    from qdif.uno.trainer import UnoTrainConfig, train_uno

    model, _ = load_model()
    config = UnoTrainConfig(
        block_size=args.block_size,
        block_curriculum=[int(b) for b in args.curriculum.split(",")] if args.curriculum else [],
        steps=args.steps,
        batch_size=args.batch_size,
        window=args.window,
        lora_rank=args.rank,
        learning_rate=args.lr,
        corruption=args.corruption,
        shuffle_targets=args.shuffle_targets,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        checkpoint_steps=[int(x) for x in args.checkpoint_steps.split(",") if x.strip()],
    )
    train_source, val_source = build_sources(
        model.tokenizer,
        width=args.window,
        batch_size=args.batch_size,
        train_windows=args.train_windows,
        val_windows=args.val_windows,
        seed=args.seed,
        eos_id=model.tokenizer.eos_token_id,
    )
    print(
        f"[data] train {train_source.num_windows} windows / {train_source.num_tokens:,} tokens | "
        f"val {val_source.num_windows} windows / {val_source.num_tokens:,} tokens"
    )
    result = train_uno(model, train_source, val_source, config, output_dir=args.out, echo=print)
    result["provenance"] = provenance({"stage": "train", "run": args.out})
    result["data"] = {
        "train_windows": train_source.num_windows,
        "train_tokens": train_source.num_tokens,
        "val_windows": val_source.num_windows,
        "val_tokens": val_source.num_tokens,
        "corpus": "wikitext-103-raw-v1",
    }
    write_json(Path(args.out) / "result.json", result)
    return 0


# --------------------------------------------------------------------- compare


def cmd_compare(args) -> int:
    """Evaluate several adapter states on ONE held-out set, in one process.

    The arms differ only by which adapter tensors are loaded: same model object, same
    windows, same noise seed, same block size. That is the only way a difference of a
    few percent means anything — Act II's controls ran at a different setting from the
    arm they controlled for, which invalidated a whole phase.
    """
    import mlx.core as mx

    from qdif.uno.data import build_sources
    from qdif.uno.gated_lora import gated_lora_modules
    from qdif.uno.metrics import entropy_buckets
    from qdif.uno.teacher import build_uno_batch
    from qdif.uno.trainer import load_adapter
    from qdif.uno.losses import total_variation
    from qdif.uno.metrics import slot_metrics

    model, _ = load_model(rank=args.rank, full_attention=True, mlp=True)
    fingerprint = model.fingerprint(full=True)
    pristine = {
        id(mod): (mod.lora_a, mod.lora_b) for mod in gated_lora_modules(model.base)
    }

    _, val_source = build_sources(
        model.tokenizer, width=args.window, batch_size=args.batch_size,
        train_windows=8, val_windows=args.val_windows, seed=args.seed,
        eos_id=model.tokenizer.eos_token_id,
    )
    windows = [next(val_source) for _ in range(args.eval_batches)]
    print(f"[data] {len(windows)} eval batches x {args.batch_size} = "
          f"{len(windows) * args.batch_size} held-out rows")

    arms = [("untrained", "")] + [
        (name, path) for name, path in (a.split("=", 1) for a in args.arm)
    ]
    results = {}
    for name, path in arms:
        for mod in gated_lora_modules(model.base):
            a, b = pristine[id(mod)]
            mod.lora_a, mod.lora_b = a, mx.zeros_like(b)
        mx.eval(model.base.parameters())
        if path:
            load_adapter(model, path)

        per_block = {}
        for block_size in [int(b) for b in args.block_sizes.split(",")]:
            slot_totals, tv_total, rows = [], 0.0, 0
            students, teachers = [], []
            for chunk in windows:
                batch = build_uno_batch(
                    chunk, block_size=block_size, mask_token_id=model.mask_token_id,
                    vocab_size=model.vocab_size, corruption="full",
                )
                teacher = mx.stop_gradient(
                    model.ar_logits(batch.teacher_ids)[:, batch.supervised_slice, :]
                )
                student = mx.stop_gradient(
                    model.draft_logits(batch.student_ids, batch.lora_mask)[
                        :, batch.supervised_slice, :
                    ]
                )
                slot_totals.append(slot_metrics(student, teacher, batch.targets))
                tv_total += float(total_variation(student, teacher).item()) * batch.batch_size
                rows += batch.batch_size
                if block_size == 4:
                    students.append(student[:, 1:].reshape(-1, student.shape[-1]))
                    teachers.append(teacher[:, 1:].reshape(-1, teacher.shape[-1]))

            def mean(key):
                values = [s[key] for s in slot_totals]
                if isinstance(values[0], list):
                    return [round(sum(c) / len(c), 4) for c in zip(*values)]
                return round(sum(values) / len(values), 4)

            summary = {
                "agree_specs": mean("agree_specs"),
                "agree_per_slot": mean("agree_per_slot"),
                "gold_specs": mean("gold_specs"),
                "mean_accepted_prefix": mean("mean_accepted_prefix"),
                "full_block_rate": mean("full_block_rate"),
                "tv": round(tv_total / rows, 4),
                "rows": rows,
            }
            if students:
                summary["entropy_buckets"] = entropy_buckets(
                    mx.concatenate(students), mx.concatenate(teachers), num_buckets=5
                )
            per_block[f"B{block_size}"] = summary
            print(
                f"  [{name:>10}] B={block_size} tv {summary['tv']:.4f} "
                f"agree {summary['agree_specs']:.4f} "
                f"prefix {summary['mean_accepted_prefix']:.3f} "
                f"per-slot {summary['agree_per_slot']}"
            )
        results[name] = per_block

    payload = {
        "provenance": provenance({"stage": "compare"}),
        "arms": results,
        "backbone_unchanged": model.fingerprint(full=True).digest == fingerprint.digest,
        "eval_rows": len(windows) * args.batch_size,
    }
    write_json(Path(args.out), payload)
    return 0


# ------------------------------------------------------------------------ bench


def cmd_bench(args) -> int:
    """Decode benchmark: acceptance, sequential-step reduction, wall clock, entropy.

    `--adapter ''` benchmarks the untrained (zero-init) adapter, which is Control 1
    and the floor every trained number must clear.
    """
    from qdif.uno.data import prompt_suite_tokens
    from qdif.uno.decode import ar_greedy_generate, uno_greedy_generate
    from qdif.uno.metrics import pearson
    from qdif.uno.trainer import load_adapter

    model, _ = load_model(
        rank=args.rank, full_attention=True, mlp=True,
        deltanet=args.lora_deltanet, alpha=args.alpha,
    )
    if args.adapter:
        n = load_adapter(model, args.adapter)
        print(f"[uno] loaded {n} adapter tensors from {args.adapter}")
    else:
        print("[uno] no adapter loaded — this is Control 1 (untrained, zero-init)")

    fingerprint = model.fingerprint(full=True)
    prompts = prompt_suite_tokens(model.tokenizer)

    print("\n-- AR baseline (same harness) --")
    ar_rows = []
    for domain, text, ids in prompts:
        for _ in range(args.warmup):
            ar_greedy_generate(model, ids, max_tokens=8)
        best = None
        for _ in range(args.repeats):
            tokens, stats = ar_greedy_generate(model, ids, max_tokens=args.tokens)
            row = stats.to_dict()
            row.update({"domain": domain, "prompt": text})
            if best is None or row["tokens_per_second"] > best["tokens_per_second"]:
                best = row
            ar_reference = tokens
        best["token_ids"] = ar_reference
        ar_rows.append(best)
    ar_speed = sorted(r["tokens_per_second"] for r in ar_rows)
    ar_median = ar_speed[len(ar_speed) // 2]
    print(f"  median {ar_median:.2f} tok/s over {len(ar_rows)} prompts")

    results = {}
    modes = [m.strip() for m in args.transaction_modes.split(",") if m.strip()]
    for block_size in [int(b) for b in args.block_sizes.split(",")]:
      for txn_mode in modes:
        print(f"\n-- Uno B={block_size} txn={txn_mode} --")
        rows = []
        entropies, accepts = [], []
        for (domain, text, ids), ar_row in zip(prompts, ar_rows):
            for _ in range(args.warmup):
                uno_greedy_generate(model, ids, max_tokens=8, block_size=block_size,
                                    transaction_mode=txn_mode)
            best = None
            for _ in range(args.repeats):
                tokens, stats = uno_greedy_generate(
                    model, ids, max_tokens=args.tokens, block_size=block_size,
                    transaction_mode=txn_mode,
                )
                row = stats.to_dict()
                if best is None or row["tokens_per_second"] > best["tokens_per_second"]:
                    best = row
                    best_stats = stats
                    best_tokens = tokens
            agreement = sum(
                a == b for a, b in zip(best_tokens, ar_row["token_ids"])
            ) / max(len(best_tokens), 1)
            best.update(
                {
                    "domain": domain,
                    "prompt": text,
                    "ar_agreement": round(agreement, 4),
                    "forward_reduction": round(
                        ar_row["forwards"] / max(best["forwards"], 1), 4
                    ),
                    "wall_speedup": round(
                        best["tokens_per_second"] / max(ar_row["tokens_per_second"], 1e-9), 4
                    ),
                }
            )
            rows.append(best)
            entropies.extend(best_stats.entropy_per_cycle)
            accepts.extend(best_stats.accepted_per_cycle)

        speeds = sorted(r["tokens_per_second"] for r in rows)
        median_speed = speeds[len(speeds) // 2]
        total_tokens = sum(r["tokens"] for r in rows)
        total_forwards = sum(r["forwards"] for r in rows)
        total_accepted = sum(r["acceptance_rate"] * r["cycles"] for r in rows)
        total_replays = sum(r["replay_forwards"] for r in rows)
        rewind_steps = sum((r.get("transaction") or {}).get("rewind_steps", 0) for r in rows)
        rewind_fallbacks = sum((r.get("transaction") or {}).get("rewind_fallbacks", 0) for r in rows)
        summary = {
            "block_size": block_size,
            "transaction_mode": txn_mode,
            "peak_record_bytes": max(
                (r.get("transaction") or {}).get("peak_record_bytes", 0) for r in rows
            ),
            "rewind_fallback_rate": round(rewind_fallbacks / rewind_steps, 4) if rewind_steps else None,
            "tokens_per_forward": round(total_tokens / max(total_forwards, 1), 4),
            # counterfactual: what a rewindable (pure-attention) backbone would give,
            # i.e. without the DeltaNet replay tax. Accounting only, never a speed claim.
            "tokens_per_forward_without_replay": round(
                total_tokens / max(total_forwards - total_replays, 1), 4
            ),
            "replay_forwards": total_replays,
            "mean_committed_per_cycle": round(
                sum(r["mean_committed_per_cycle"] for r in rows) / len(rows), 4
            ),
            "acceptance_rate": round(
                sum(r["acceptance_rate"] for r in rows) / len(rows), 4
            ),
            "full_block_rate": round(
                sum(r["full_blocks"] for r in rows) / max(sum(r["cycles"] for r in rows), 1), 4
            ),
            "median_tokens_per_second": median_speed,
            "wall_speedup_vs_ar": round(median_speed / ar_median, 4),
            "forward_reduction_vs_ar": round(
                sum(r["forwards"] for r in ar_rows) / max(total_forwards, 1), 4
            ),
            "mean_ar_agreement": round(
                sum(r["ar_agreement"] for r in rows) / len(rows), 4
            ),
            "entropy_vs_accepted_pearson": round(pearson(entropies, [float(a) for a in accepts]), 4),
            "cycles": sum(r["cycles"] for r in rows),
            "rows": rows,
        }
        del total_accepted
        results[f"B{block_size}_{txn_mode}"] = summary
        print(
            f"  TPF {summary['tokens_per_forward']:.3f} "
            f"(no-replay {summary['tokens_per_forward_without_replay']:.3f}) | "
            f"accept {summary['acceptance_rate']:.3f} | "
            f"committed/cycle {summary['mean_committed_per_cycle']:.2f} | "
            f"{median_speed:.2f} tok/s ({summary['wall_speedup_vs_ar']:.2f}x AR) | "
            f"forward reduction {summary['forward_reduction_vs_ar']:.2f}x | "
            f"entropy~accept r={summary['entropy_vs_accepted_pearson']:+.3f}"
            + (f" | rewind fallback {100*summary['rewind_fallback_rate']:.1f}%"
               if summary["rewind_fallback_rate"] is not None else "")
            + f" | record {summary['peak_record_bytes']/1e6:.0f} MB"
        )

    payload = {
        "provenance": provenance({"stage": "bench", "adapter": args.adapter or "untrained"}),
        "backbone_digest_before": fingerprint.digest,
        "backbone_digest_after": model.fingerprint(full=True).digest,
        "tokens": args.tokens,
        "ar_baseline": {"median_tokens_per_second": ar_median, "rows": ar_rows},
        "uno": results,
    }
    payload["backbone_unchanged"] = (
        payload["backbone_digest_before"] == payload["backbone_digest_after"]
    )
    write_json(Path(args.out), payload)
    return 0


# ------------------------------------------------------------- draftability


def cmd_draftability(args) -> int:
    """Analysis only (§18): does 'draftability gap' predict acceptance better than entropy?

    Act IV-U found teacher entropy a poor predictor at the *low* end. The hypothesis is
    that what matters is not how uncertain the next token is, but how much it depends on
    the immediately preceding tokens a parallel draft cannot see.

    Both quantities are properties of the **frozen model**, measured with no adapter:

        P_full    = p(y_{t+j} | true causal predecessors)          -> ar_logits
        P_corrupt = p(y_{t+j} | predecessors replaced by noise)    -> bare model, noise rows
        draftability_gap(j) = TV(P_full, P_corrupt)

    A large gap means the position is intrinsically un-draftable. The comparison is then
    which of `entropy` or `gap` better predicts whether the *trained* adapter's argmax
    matches the teacher's.
    """
    import mlx.core as mx

    from qdif.uno.data import build_sources
    from qdif.uno.gated_lora import gated_lora_modules
    from qdif.uno.metrics import pearson
    from qdif.uno.teacher import build_uno_batch
    from qdif.uno.trainer import load_adapter

    model, _ = load_model(rank=args.rank, full_attention=True, mlp=True)
    pristine = {id(m): (m.lora_a, m.lora_b) for m in gated_lora_modules(model.base)}

    _, val_source = build_sources(
        model.tokenizer, width=args.window, batch_size=args.batch_size,
        train_windows=8, val_windows=args.val_windows, seed=args.seed,
        eos_id=model.tokenizer.eos_token_id,
    )
    windows = [next(val_source) for _ in range(args.eval_batches)]

    gaps, entropies, hits, slots = [], [], [], []
    for chunk in windows:
        batch = build_uno_batch(
            chunk, block_size=args.block_size, mask_token_id=model.mask_token_id,
            vocab_size=model.vocab_size, corruption="full",
        )
        # frozen-model quantities: adapter reset to its zero-init no-op
        for module in gated_lora_modules(model.base):
            a, b = pristine[id(module)]
            module.lora_a, module.lora_b = a, mx.zeros_like(b)
        mx.eval(model.base.parameters())

        full = mx.stop_gradient(
            model.ar_logits(batch.teacher_ids)[:, batch.supervised_slice, :]
        ).astype(mx.float32)
        corrupt = mx.stop_gradient(
            model.draft_logits(batch.student_ids, batch.lora_mask)[
                :, batch.supervised_slice, :
            ]
        ).astype(mx.float32)

        lf = full - mx.logsumexp(full, axis=-1, keepdims=True)
        lc = corrupt - mx.logsumexp(corrupt, axis=-1, keepdims=True)
        gap = mx.sum(mx.abs(mx.exp(lf) - mx.exp(lc)), axis=-1)          # [B, L]
        entropy = -mx.sum(mx.exp(lf) * lf, axis=-1)                      # [B, L]

        # trained adapter -> did it agree with the teacher at this position?
        load_adapter(model, args.adapter)
        student = mx.stop_gradient(
            model.draft_logits(batch.student_ids, batch.lora_mask)[
                :, batch.supervised_slice, :
            ]
        )
        hit = (mx.argmax(student, axis=-1) == mx.argmax(full, axis=-1)).astype(mx.float32)

        b, length = gap.shape
        index = mx.broadcast_to(mx.arange(length)[None, :], (b, length))
        for arr, sink in ((gap, gaps), (entropy, entropies), (hit, hits), (index, slots)):
            sink.extend([float(x) for x in arr.reshape(-1).tolist()])

    # slot 0 is the unadapted seed and is trivially a hit; exclude it.
    keep = [i for i, s in enumerate(slots) if s > 0]
    gaps = [gaps[i] for i in keep]
    entropies = [entropies[i] for i in keep]
    hits = [hits[i] for i in keep]

    r_gap = pearson(gaps, hits)
    r_entropy = pearson(entropies, hits)
    r_cross = pearson(gaps, entropies)

    def buckets(values, name):
        order = sorted(range(len(values)), key=lambda i: values[i])
        size = max(1, len(order) // 5)
        out = []
        for index in range(5):
            lo = index * size
            hi = len(order) if index == 4 else min(len(order), (index + 1) * size)
            chunk = order[lo:hi]
            if not chunk:
                continue
            out.append({
                "bucket": index, "n": len(chunk),
                "mean": round(sum(values[i] for i in chunk) / len(chunk), 4),
                "agreement": round(sum(hits[i] for i in chunk) / len(chunk), 4),
            })
        print(f"  {name}:")
        for row in out:
            print(f"    q{row['bucket']} n={row['n']:5d} mean {row['mean']:8.4f} -> agreement {row['agreement']:.4f}")
        return out

    print(f"\n-- draftability vs entropy (K={args.block_size}, {len(hits)} spec positions) --")
    gap_buckets = buckets(gaps, "draftability gap TV(P_full, P_corrupt)")
    entropy_buckets_ = buckets(entropies, "teacher entropy H(P_full)")
    print(f"\n  pearson(gap, agreement)     = {r_gap:+.4f}")
    print(f"  pearson(entropy, agreement) = {r_entropy:+.4f}")
    print(f"  pearson(gap, entropy)       = {r_cross:+.4f}")
    dominant = "draftability_gap" if abs(r_gap) > abs(r_entropy) else "entropy"
    print(f"  stronger predictor: {dominant}")

    write_json(Path(args.out), {
        "provenance": provenance({"stage": "draftability"}),
        "block_size": args.block_size,
        "positions": len(hits),
        "pearson": {"gap_vs_agreement": r_gap, "entropy_vs_agreement": r_entropy,
                    "gap_vs_entropy": r_cross},
        "stronger_predictor": dominant,
        "gap_buckets": gap_buckets,
        "entropy_buckets": entropy_buckets_,
    })
    return 0



# ------------------------------------------------------------------- u3 eval


def _median_std(values):
    ordered = sorted(values)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
    return median, variance ** 0.5, mean


def cmd_u3_eval(args) -> int:
    """Evaluate every U3 checkpoint in ONE session, on one harness.

    Everything that is compared is measured back to back in the same process: the AR
    baseline, each checkpoint's held-out teacher-forced agreement, its free-running
    decode statistics, and its draftability-gap stratification. Absolute tok/s drifts
    between sessions with machine load, so cross-session comparison of a few percent
    would be meaningless.
    """
    import mlx.core as mx

    from qdif.uno.cost_model import (
        CycleCost, ceiling, distribution_from_counts, predicted_tokens_per_second,
        predicted_tpf, speedup_table,
    )
    from qdif.uno.data import build_sources, prompt_suite_tokens
    from qdif.uno.decode import ar_greedy_generate, uno_greedy_generate
    from qdif.uno.gated_lora import gated_lora_modules
    from qdif.uno.losses import total_variation
    from qdif.uno.metrics import pearson, slot_metrics
    from qdif.uno.teacher import build_uno_batch
    from qdif.uno.trainer import load_adapter

    model, _ = load_model(rank=args.rank, full_attention=True, mlp=True)
    fingerprint = model.fingerprint(full=True)
    pristine = {id(m): (m.lora_a, m.lora_b) for m in gated_lora_modules(model.base)}
    block_sizes = [int(b) for b in args.block_sizes.split(",")]

    prompts = prompt_suite_tokens(model.tokenizer)
    _, val_source = build_sources(
        model.tokenizer, width=args.window, batch_size=args.batch_size,
        train_windows=8, val_windows=args.val_windows, seed=args.seed,
        eos_id=model.tokenizer.eos_token_id,
    )
    windows = [next(val_source) for _ in range(args.eval_batches)]
    print(f"[u3] {len(windows) * args.batch_size} held-out rows, "
          f"{len(prompts)} decode prompts, {args.repeats} repeats")

    def reset_adapter():
        for module in gated_lora_modules(model.base):
            a, b = pristine[id(module)]
            module.lora_a, module.lora_b = a, mx.zeros_like(b)
        mx.eval(model.base.parameters())

    # ---- AR baseline, measured once in this session -------------------------
    print("\n== AR baseline ==")
    reset_adapter()
    ar_rows, ar_reference = [], {}
    for domain, text, ids in prompts:
        for _ in range(args.warmup):
            ar_greedy_generate(model, ids, max_tokens=8)
        speeds = []
        for _ in range(args.repeats):
            tokens, st = ar_greedy_generate(model, ids, max_tokens=args.tokens)
            speeds.append(st.tokens_per_second)
        ar_reference[text] = tokens
        ar_rows.append({"domain": domain, "prompt": text,
                        "tokens_per_second": max(speeds), "forwards": st.forwards})
    ar_median, ar_std, ar_mean = _median_std([r["tokens_per_second"] for r in ar_rows])
    print(f"  AR {ar_median:.2f} tok/s (mean {ar_mean:.2f}, sd across prompts {ar_std:.2f})")

    checkpoints = [c.split("=", 1) for c in args.checkpoint]
    results = []

    for name, path in checkpoints:
        print(f"\n== checkpoint {name} ==")
        reset_adapter()
        n_loaded = load_adapter(model, str(Path(path) / "adapter.safetensors")
                                if Path(path).is_dir() else path)
        entry = {"checkpoint": name, "path": path, "adapter_tensors": n_loaded,
                 "per_block": {}}

        for block_size in block_sizes:
            # ---- held-out teacher-forced (verifier acceptance) --------------
            tf, tv_total, rows_n = [], 0.0, 0
            gaps, hits = [], []
            for chunk in windows:
                batch = build_uno_batch(
                    chunk, block_size=block_size, mask_token_id=model.mask_token_id,
                    vocab_size=model.vocab_size, corruption="full",
                )
                teacher = mx.stop_gradient(
                    model.ar_logits(batch.teacher_ids)[:, batch.supervised_slice, :])
                student = mx.stop_gradient(
                    model.draft_logits(batch.student_ids, batch.lora_mask)[
                        :, batch.supervised_slice, :])
                tf.append(slot_metrics(student, teacher, batch.targets))
                tv_total += float(total_variation(student, teacher).item()) * batch.batch_size
                rows_n += batch.batch_size

            def merge(key):
                vals = [m[key] for m in tf]
                if isinstance(vals[0], list):
                    return [round(sum(c) / len(c), 5) for c in zip(*vals)]
                return round(sum(vals) / len(vals), 5)

            teacher_forced = {
                "agree_specs": merge("agree_specs"),
                "agree_per_slot": merge("agree_per_slot"),
                "mean_accepted_prefix": merge("mean_accepted_prefix"),
                "full_block_rate": merge("full_block_rate"),
                "validation_tv": round(tv_total / rows_n, 5),
                "rows": rows_n,
            }

            # ---- free-running decode ---------------------------------------
            speeds, tpfs, hist, prop, ver, com = [], [], {}, [], [], []
            repeat_spread = []
            forwards = tokens_total = replays = 0
            agreement = []
            for domain, text, ids in prompts:
                for _ in range(args.warmup):
                    uno_greedy_generate(model, ids, max_tokens=8, block_size=block_size,
                                        transaction_mode="snapshot")
                best = None
                per_prompt = []
                for _ in range(args.repeats):
                    toks, st = uno_greedy_generate(
                        model, ids, max_tokens=args.tokens, block_size=block_size,
                        transaction_mode="snapshot")
                    per_prompt.append(st.tokens_per_second)
                    if best is None or st.tokens_per_second > best.tokens_per_second:
                        best, best_tokens = st, toks
                if len(per_prompt) > 1:
                    mu = sum(per_prompt) / len(per_prompt)
                    repeat_spread.append(
                        (sum((v - mu) ** 2 for v in per_prompt) / (len(per_prompt) - 1)) ** 0.5)
                speeds.append(best.tokens_per_second)
                tpfs.append(best.tokens_per_forward)
                forwards += best.forwards
                tokens_total += best.tokens
                replays += best.replay_forwards
                for a in best.accepted_per_cycle:
                    hist[a] = hist.get(a, 0) + 1
                d = best.to_dict()
                prop.append(d["proposal_ms_per_cycle"])
                ver.append(d["verify_ms_per_cycle"])
                com.append(d["commit_ms_per_cycle"])
                ref = ar_reference[text]
                agreement.append(sum(x == y for x, y in zip(best_tokens, ref)) / max(len(ref), 1))

            med, sd, mean = _median_std(speeds)
            cost = CycleCost(sum(prop) / len(prop), sum(ver) / len(ver), sum(com) / len(com))
            distribution = distribution_from_counts(hist)
            offered = block_size - 1
            free_running = {
                "tokens_per_forward": round(tokens_total / forwards, 5),
                "forwards_per_token": round(forwards / tokens_total, 5),
                "replay_forwards": replays,
                "median_tokens_per_second": round(med, 3),
                "mean_tokens_per_second": round(mean, 3),
                "std_tokens_per_second": round(sd, 3),
                # Spread across prompts mixes prompt difficulty with timing noise.
                # `repeat_noise_tok_s` is the pure measurement noise: the mean, over
                # prompts, of the standard deviation across repeats of the SAME prompt.
                # U3-4 requires beating the U2 baseline by more than this.
                "repeat_noise_tok_s": round(
                    sum(repeat_spread) / len(repeat_spread), 4) if repeat_spread else 0.0,
                "speedup_vs_ar": round(med / ar_median, 4),
                "acceptance_histogram": {str(k): v for k, v in sorted(hist.items())},
                "mean_accepted_specs": round(
                    sum(k * v for k, v in hist.items()) / sum(hist.values()), 4),
                "acceptance_rate": round(
                    sum(k * v for k, v in hist.items()) / (sum(hist.values()) * offered), 4)
                    if offered else 0.0,
                "full_block_rate": round(hist.get(offered, 0) / sum(hist.values()), 4),
                "p_accept_ge": {
                    str(k): round(sum(v for a, v in hist.items() if a >= k) / sum(hist.values()), 4)
                    for k in range(1, min(offered, 4) + 1)},
                "cycle_cost_ms": cost.to_dict(),
                "predicted_tokens_per_second": round(
                    predicted_tokens_per_second(block_size, distribution, cost), 3),
                "predicted_tpf": round(predicted_tpf(block_size, distribution), 5),
                "ar_agreement": round(sum(agreement) / len(agreement), 4),
                "ceiling": ceiling(block_size, cost, ar_median),
                "speedup_requirements": speedup_table(block_size, cost, ar_median),
            }
            free_running["cost_model_error_pct"] = round(
                100 * (free_running["predicted_tokens_per_second"] - med) / med, 3)

            entry["per_block"][f"K{block_size}"] = {
                "teacher_forced": teacher_forced, "free_running": free_running}
            print(f"  K={block_size}  tf-agree {teacher_forced['agree_specs']:.4f} "
                  f"val-tv {teacher_forced['validation_tv']:.4f} | "
                  f"accept {free_running['acceptance_rate']:.4f} "
                  f"TPF {free_running['tokens_per_forward']:.4f} | "
                  f"{med:.2f}+-{sd:.2f} tok/s ({free_running['speedup_vs_ar']:.3f}x AR) | "
                  f"cost-model err {free_running['cost_model_error_pct']:+.1f}%")

        # ---- draftability stratification, frozen-model metric ---------------
        entry["draftability"] = _draftability_quintiles(
            model, windows, args.draftability_block, reset_adapter, pristine)
        q = entry["draftability"]
        print("  draftability quintile acceptance: "
              + " ".join(f"q{b['bucket']}={b['agreement']:.3f}" for b in q["buckets"])
              + f" | r(gap)={q['pearson_gap']:+.4f} r(H)={q['pearson_entropy']:+.4f}")

        entry["backbone_digest"] = model.fingerprint(full=True).digest
        entry["backbone_unchanged"] = entry["backbone_digest"] == fingerprint.digest
        results.append(entry)

    payload = {
        "provenance": provenance({"stage": "u3-eval"}),
        "ar_baseline": {"median_tokens_per_second": round(ar_median, 3),
                        "mean_tokens_per_second": round(ar_mean, 3),
                        "std_tokens_per_second": round(ar_std, 3),
                        "rows": ar_rows},
        "harness": {"tokens": args.tokens, "repeats": args.repeats,
                    "warmup": args.warmup, "eval_rows": len(windows) * args.batch_size,
                    "prompts": len(prompts), "window": args.window, "seed": args.seed},
        "backbone_digest": fingerprint.digest,
        "checkpoints": results,
    }
    write_json(Path(args.out), payload)
    return 0


def _draftability_quintiles(model, windows, block_size, reset_adapter, pristine):
    """Acceptance stratified by the FROZEN model's draftability gap.

    The gap is a property of the base model and the text, so it is recomputed with the
    adapter reset to its zero-init no-op and never drifts as the adapter trains.
    """
    import mlx.core as mx

    from qdif.uno.gated_lora import gated_lora_modules
    from qdif.uno.metrics import pearson
    from qdif.uno.teacher import build_uno_batch

    trained = {id(m): (m.lora_a, m.lora_b) for m in gated_lora_modules(model.base)}
    gaps, entropies, hits = [], [], []

    for chunk in windows:
        batch = build_uno_batch(
            chunk, block_size=block_size, mask_token_id=model.mask_token_id,
            vocab_size=model.vocab_size, corruption="full")

        reset_adapter()
        full = mx.stop_gradient(
            model.ar_logits(batch.teacher_ids)[:, batch.supervised_slice, :]).astype(mx.float32)
        corrupt = mx.stop_gradient(
            model.draft_logits(batch.student_ids, batch.lora_mask)[
                :, batch.supervised_slice, :]).astype(mx.float32)
        lf = full - mx.logsumexp(full, axis=-1, keepdims=True)
        lc = corrupt - mx.logsumexp(corrupt, axis=-1, keepdims=True)
        gap = mx.sum(mx.abs(mx.exp(lf) - mx.exp(lc)), axis=-1)
        entropy = -mx.sum(mx.exp(lf) * lf, axis=-1)

        for module in gated_lora_modules(model.base):
            a, b = trained[id(module)]
            module.lora_a, module.lora_b = a, b
        mx.eval(model.base.parameters())
        student = mx.stop_gradient(
            model.draft_logits(batch.student_ids, batch.lora_mask)[
                :, batch.supervised_slice, :])
        hit = (mx.argmax(student, axis=-1) == mx.argmax(full, axis=-1)).astype(mx.float32)

        b_, length = gap.shape
        for row in range(b_):
            for slot in range(1, length):  # slot 0 is the unadapted seed
                gaps.append(float(gap[row, slot]))
                entropies.append(float(entropy[row, slot]))
                hits.append(float(hit[row, slot]))

    order = sorted(range(len(gaps)), key=lambda i: gaps[i])
    size = max(1, len(order) // 5)
    buckets = []
    for index in range(5):
        lo = index * size
        hi = len(order) if index == 4 else min(len(order), (index + 1) * size)
        chunk_idx = order[lo:hi]
        if not chunk_idx:
            continue
        buckets.append({
            "bucket": index, "n": len(chunk_idx),
            "mean_gap": round(sum(gaps[i] for i in chunk_idx) / len(chunk_idx), 5),
            "agreement": round(sum(hits[i] for i in chunk_idx) / len(chunk_idx), 5),
        })
    return {
        "block_size": block_size, "positions": len(hits), "buckets": buckets,
        "pearson_gap": round(pearson(gaps, hits), 5),
        "pearson_entropy": round(pearson(entropies, hits), 5),
    }


# ------------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description="Act IV-U (Uno) runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p0 = sub.add_parser("stage0", help="implementation sanity on the real model")
    p0.add_argument("--rank", type=int, default=16)
    p0.add_argument("--block-size", type=int, default=4)
    p0.add_argument("--window", type=int, default=128)
    p0.add_argument("--decode-tokens", type=int, default=16)
    p0.add_argument("--out", default="runs/uno-stage0/stage0.json")
    p0.set_defaults(func=cmd_stage0)

    pb = sub.add_parser("baseline", help="frozen AR baseline")
    pb.add_argument("--window", type=int, default=256)
    pb.add_argument("--eval-windows", type=int, default=64)
    pb.add_argument("--batch-size", type=int, default=4)
    pb.add_argument("--decode-tokens", type=int, default=64)
    pb.add_argument("--repeats", type=int, default=3)
    pb.add_argument("--out", default="runs/uno-baseline/baseline.json")
    pb.set_defaults(func=cmd_baseline)

    po = sub.add_parser("overfit", help="one-batch overfit plus shuffled control")
    po.add_argument("--rank", type=int, default=16)
    po.add_argument("--block-size", type=int, default=4)
    po.add_argument("--window", type=int, default=128)
    po.add_argument("--batch-size", type=int, default=4)
    po.add_argument("--steps", type=int, default=60)
    po.add_argument("--lr", type=float, default=1e-5)
    po.add_argument("--out", default="runs/uno-overfit/overfit.json")
    po.set_defaults(func=cmd_overfit)

    pt = sub.add_parser("train", help="held-out training run")
    pt.add_argument("--rank", type=int, default=16)
    pt.add_argument("--block-size", type=int, default=4)
    pt.add_argument("--curriculum", default="")
    pt.add_argument("--window", type=int, default=128)
    pt.add_argument("--batch-size", type=int, default=4)
    pt.add_argument("--steps", type=int, default=300)
    pt.add_argument("--lr", type=float, default=1e-5)
    pt.add_argument("--corruption", default="uniform", choices=["uniform", "full"])
    pt.add_argument("--shuffle-targets", action="store_true")
    pt.add_argument("--train-windows", type=int, default=4000)
    pt.add_argument("--val-windows", type=int, default=64)
    pt.add_argument("--eval-every", type=int, default=50)
    pt.add_argument("--eval-batches", type=int, default=4)
    pt.add_argument("--seed", type=int, default=20260903)
    pt.add_argument("--checkpoint-steps", default="", help="comma-separated steps to checkpoint")
    pt.add_argument("--out", default="runs/uno-train")
    pt.set_defaults(func=cmd_train)

    pc = sub.add_parser("compare", help="evaluate adapter states on one held-out set")
    pc.add_argument("--arm", action="append", default=[], metavar="NAME=PATH")
    pc.add_argument("--rank", type=int, default=16)
    pc.add_argument("--window", type=int, default=128)
    pc.add_argument("--batch-size", type=int, default=4)
    pc.add_argument("--eval-batches", type=int, default=32)
    pc.add_argument("--val-windows", type=int, default=256)
    pc.add_argument("--block-sizes", default="2,4,8")
    pc.add_argument("--seed", type=int, default=20260903)
    pc.add_argument("--out", default="runs/uno-compare/compare.json")
    pc.set_defaults(func=cmd_compare)

    pn = sub.add_parser("bench", help="decode benchmark against the AR baseline")
    pn.add_argument("--adapter", default="", help="path to adapter.safetensors; empty = Control 1")
    pn.add_argument("--rank", type=int, default=16)
    pn.add_argument("--alpha", type=float, default=None)
    pn.add_argument("--lora-deltanet", action="store_true")
    pn.add_argument("--block-sizes", default="1,2,4,8")
    pn.add_argument("--transaction-modes", default="replay")
    pn.add_argument("--tokens", type=int, default=48)
    pn.add_argument("--repeats", type=int, default=3)
    pn.add_argument("--warmup", type=int, default=1)
    pn.add_argument("--out", default="runs/uno-bench/bench.json")
    pn.set_defaults(func=cmd_bench)

    pd = sub.add_parser("draftability", help="predecessor-dependence analysis (§18)")
    pd.add_argument("--adapter", required=True)
    pd.add_argument("--rank", type=int, default=16)
    pd.add_argument("--block-size", type=int, default=4)
    pd.add_argument("--window", type=int, default=128)
    pd.add_argument("--batch-size", type=int, default=4)
    pd.add_argument("--eval-batches", type=int, default=32)
    pd.add_argument("--val-windows", type=int, default=256)
    pd.add_argument("--seed", type=int, default=20260903)
    pd.add_argument("--out", default="runs/uno-draftability/draftability.json")
    pd.set_defaults(func=cmd_draftability)

    p3 = sub.add_parser("u3-eval", help="evaluate U3 checkpoints in one session")
    p3.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    p3.add_argument("--rank", type=int, default=16)
    p3.add_argument("--block-sizes", default="2,4")
    p3.add_argument("--draftability-block", type=int, default=4)
    p3.add_argument("--window", type=int, default=128)
    p3.add_argument("--batch-size", type=int, default=4)
    p3.add_argument("--eval-batches", type=int, default=32)
    p3.add_argument("--val-windows", type=int, default=256)
    p3.add_argument("--tokens", type=int, default=48)
    p3.add_argument("--repeats", type=int, default=3)
    p3.add_argument("--warmup", type=int, default=1)
    p3.add_argument("--seed", type=int, default=20260903)
    p3.add_argument("--out", default="runs/u3a/eval.json")
    p3.set_defaults(func=cmd_u3_eval)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
