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
    for block_size in [int(b) for b in args.block_sizes.split(",")]:
        print(f"\n-- Uno B={block_size} --")
        rows = []
        entropies, accepts = [], []
        for (domain, text, ids), ar_row in zip(prompts, ar_rows):
            for _ in range(args.warmup):
                uno_greedy_generate(model, ids, max_tokens=8, block_size=block_size)
            best = None
            for _ in range(args.repeats):
                tokens, stats = uno_greedy_generate(
                    model, ids, max_tokens=args.tokens, block_size=block_size
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
        summary = {
            "block_size": block_size,
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
        results[f"B{block_size}"] = summary
        print(
            f"  TPF {summary['tokens_per_forward']:.3f} "
            f"(no-replay {summary['tokens_per_forward_without_replay']:.3f}) | "
            f"accept {summary['acceptance_rate']:.3f} | "
            f"committed/cycle {summary['mean_committed_per_cycle']:.2f} | "
            f"{median_speed:.2f} tok/s ({summary['wall_speedup_vs_ar']:.2f}x AR) | "
            f"forward reduction {summary['forward_reduction_vs_ar']:.2f}x | "
            f"entropy~accept r={summary['entropy_vs_accepted_pearson']:+.3f}"
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
    pn.add_argument("--tokens", type=int, default=48)
    pn.add_argument("--repeats", type=int, default=3)
    pn.add_argument("--warmup", type=int, default=1)
    pn.add_argument("--out", default="runs/uno-bench/bench.json")
    pn.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
