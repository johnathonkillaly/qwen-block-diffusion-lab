#!/usr/bin/env python
"""Act IV-S — finishing the speculative decoder experiment.

Act IV-U3/U4 established that the Uno-style diffusion adapter drafts tokens a frozen
Qwen3.5-4B verifies, losslessly, at up to 1.195x AR. This script measures the things
those Acts never did (`docs/STATE_RECOVERY.md` §5): prompt-class stratification,
context scaling, a non-diffusion drafter control, iterative refinement, and adaptive K.

Measurement hygiene, per brief §20, applied by construction rather than by care:

  * **Arms are interleaved inside each prompt inside each repeat.** Thermal drift and
    memory pressure then hit every arm equally. A baseline measured at the top of a
    session and a candidate measured an hour later is the one mistake this harness
    cannot make.
  * **Median and IQR, never best-of-N.** `scripts/uno.py bench` reports best-of-N;
    nothing here does.
  * **The draft noise is keyed** to an explicit stream, so two arms meet identical
    noise at the same cycle index. Act IV-U3 gate U3-0b exists because an unkeyed
    stream silently coupled evaluation to training.
  * **Warm-up runs are discarded**, so Metal kernel compilation lands outside the
    timed region.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RESULTS = ROOT / "results" / "speculative"
MODEL_ID = "unsloth/Qwen3.5-4B-Base"
DEFAULT_ADAPTER = "runs/u4b1/step-16000/adapter.safetensors"
NOISE_STREAM_SEED = 20260905

#: A continuation is "clean" when under this share of its 8-grams are repeats. The
#: threshold is not tuned: it is the point below which a continuation contains
#: essentially no loop at all, and the baseline run's clean/looped split at 128 tokens
#: (16 of 27 prompts clean) is reported both ways so the choice is visible.
CLEAN_THRESHOLD = 0.05


# ---------------------------------------------------------------- prompt suite

#: The brief's §7 categories. The first five are `data.PROMPT_SUITE` **verbatim**, so
#: that any number here can be put beside an Act IV-U number without a caveat about
#: the prompt set. The last four are new, and exist because the brief asks for
#: reasoning, dialogue, technical text and high-entropy continuation, none of which
#: the Act IV-U suite contained.
#:
#: `high_entropy` is the important one: it is the case where the next token is close to
#: unpredictable, so a drafter should fail on it. If acceptance there matches prose,
#: something is wrong with the measurement rather than right with the adapter.
SPEC_SUITE: dict[str, list[str]] = {
    "prose": [
        "The history of the printing press begins in the fifteenth century, when",
        "He looked across the room and",
        "In the years following the war, the city",
    ],
    "factual": [
        "The capital of France is",
        "Water boils at a temperature of",
        "The largest planet in the solar system is",
    ],
    "code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "import numpy as np\n\ndef normalize(x):\n    \"\"\"Scale x to zero mean and unit variance.\"\"\"\n    return",
        "for i in range(10):\n    print(",
    ],
    "math": [
        "To solve the equation 2x + 5 = 13, we first subtract 5 from both sides, giving",
        "The derivative of x^3 with respect to x is",
        "If a triangle has sides of length 3, 4 and 5, then",
    ],
    "structured": [
        '{"name": "Ada Lovelace", "born": 1815, "known_for":',
        "| Country | Capital |\n| --- | --- |\n| France | Paris |\n| Japan |",
        "Ingredients:\n- 2 cups flour\n- 1 cup sugar\n-",
    ],
    "technical": [
        "The TCP three-way handshake begins when the client sends a SYN segment, after which",
        "In a B-tree of order m, every internal node other than the root has at least",
        "Gradient descent converges to a local minimum provided the learning rate is",
    ],
    "dialogue": [
        '"I told you it would rain," she said. "Did you bring"',
        "User: How do I reset my password?\nAssistant: To reset your password,",
        '"Are you coming to the meeting?"\n"I can\'t," he replied, "because'
    ],
    "reasoning": [
        "All birds have feathers. A penguin is a bird. Therefore a penguin",
        "If it takes 5 machines 5 minutes to make 5 widgets, then 100 machines making 100 widgets would take",
        "Alice is taller than Bob, and Bob is taller than Carol. The shortest of the three is",
    ],
    "high_entropy": [
        "qx7 zephyr blorp 44 tangent mauve",
        "The seventeenth ingredient on the unlabelled jar read",
        "Random words: carpet, velocity, plum, ardent,",
    ],
}


def suite_prompts(tokenizer, categories: list[str] | None = None):
    """`(category, text, token_ids)` for every prompt in the suite."""
    out = []
    for category, prompts in SPEC_SUITE.items():
        if categories and category not in categories:
            continue
        for text in prompts:
            out.append((category, text, list(tokenizer.encode(text))))
    return out


def long_context_prompt(tokenizer, target_tokens: int) -> list[int]:
    """A real prompt of ~`target_tokens`, built from held-out WikiText-103 validation.

    Padding with a repeated phrase would be much easier and would invalidate the whole
    measurement: a repetitive context is trivially draftable, so acceptance would rise
    with context length for a reason that has nothing to do with context length. This
    uses genuine prose, and the same text for every arm.
    """
    from qdif.uno.data import load_wikitext

    ids: list[int] = []
    for text in load_wikitext("validation", min_chars=200):
        ids.extend(tokenizer.encode(text))
        if len(ids) >= target_tokens + 8:
            break
    if len(ids) < target_tokens:
        raise RuntimeError(
            f"only {len(ids)} validation tokens available, need {target_tokens}"
        )
    return ids[:target_tokens]


# -------------------------------------------------------------------- harness


def repetition_stats(tokens: list[int], n: int = 8) -> dict:
    """How much of this continuation is a loop?

    Greedy decoding from a *base* model degenerates into repetition, and a loop is
    trivially draftable — so acceptance measured on looped text is measuring
    degeneration, not drafting. The Act IV-S baseline run found mean looped fraction
    0.18 at 128 tokens and **0.50 at 512**, with r(loop, acceptance) = +0.43…+0.52.
    Every row therefore carries this so that any claim can be restated on clean text.

    `looped_fraction` is the share of positions starting an n-gram already seen earlier
    in the same continuation: ~0 for healthy text, ~1 for a model stuck in a cycle.
    """
    if len(tokens) < 2 * n:
        return {"looped_fraction": 0.0, "distinct_ratio": 1.0}
    seen: set[tuple] = set()
    repeats = 0
    total = 0
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        total += 1
        if gram in seen:
            repeats += 1
        seen.add(gram)
    return {
        "looped_fraction": round(repeats / total, 4) if total else 0.0,
        "distinct_ratio": round(len(set(tokens)) / len(tokens), 4),
    }


def provenance(model=None) -> dict:
    def git(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return "unknown"

    import mlx.core as mx

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx": getattr(mx, "__version__", "unknown"),
        "model": MODEL_ID,
    }
    if model is not None:
        payload["backbone_digest"] = model.fingerprint(full=True)
    return payload


def summarize(values: list[float]) -> dict:
    """Median and dispersion. Never a maximum."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "median": round(statistics.median(ordered), 4),
        "mean": round(statistics.fmean(ordered), 4),
        "sd": round(statistics.stdev(ordered), 4) if n > 1 else 0.0,
        "p25": round(ordered[max(0, int(0.25 * n) - (1 if n % 4 == 0 else 0))], 4),
        "p75": round(ordered[min(n - 1, int(0.75 * n))], 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def load(adapter: str | None, rank: int = 16):
    from uno import load_model
    from qdif.uno.trainer import load_adapter

    model, _ = load_model(rank=rank, full_attention=True, mlp=True)
    if adapter:
        n = load_adapter(model, adapter)
        print(f"[spec] loaded {n} adapter tensors from {adapter}")
    else:
        print("[spec] NO adapter — untrained zero-init control")
    return model


def run_arm(model, arm: dict, ids: list[int], tokens: int):
    """Dispatch one arm. Returns `(token_ids, DecodeStats, extra)`."""
    from qdif.uno.decode import ar_greedy_generate, uno_greedy_generate
    from qdif.uno.speculative import (
        POLICIES, adaptive_greedy_generate, ngram_greedy_generate,
    )

    kind = arm["kind"]
    if kind == "ar":
        out, stats = ar_greedy_generate(model, ids, max_tokens=tokens)
        return out, stats, {}
    if kind == "uno":
        out, stats = uno_greedy_generate(
            model, ids, max_tokens=tokens, block_size=arm["k"],
            transaction_mode="snapshot", noise_stream_seed=NOISE_STREAM_SEED,
        )
        return out, stats, {}
    if kind == "refine":
        out, stats, extra = adaptive_greedy_generate(
            model, ids, max_tokens=tokens,
            policy=POLICIES["fixed"](name="fixed", allowed=(arm["k"],), default_k=arm["k"]),
            noise_stream_seed=NOISE_STREAM_SEED, refine_steps=arm["refine"],
            collect_confidence=arm.get("confidence", False),
        )
        return out, stats, extra
    if kind == "adaptive":
        policy = POLICIES[arm["policy"]](
            name=arm["policy"], allowed=tuple(arm["allowed"]),
            default_k=arm.get("default_k", 4), **arm.get("policy_kwargs", {}),
        )
        out, stats, extra = adaptive_greedy_generate(
            model, ids, max_tokens=tokens, policy=policy,
            noise_stream_seed=NOISE_STREAM_SEED,
            collect_confidence=arm.get("confidence", False),
        )
        return out, stats, extra
    if kind == "decoupled":
        # Act IV-U6. Imported here so Act IV-S arms never touch the new module.
        from qdif.uno.decoupled import decoupled_greedy_generate

        return decoupled_greedy_generate(
            model, ids, max_tokens=tokens, draft_width=arm["draft"],
            verify_width=arm["verify"], staged=arm["staged"],
            transaction_mode="snapshot", noise_stream_seed=NOISE_STREAM_SEED,
        )
    if kind == "ngram":
        out, stats = ngram_greedy_generate(
            model, ids, max_tokens=tokens, block_size=arm["k"], order=arm.get("order", 3),
        )
        return out, stats, {}
    raise ValueError(f"unknown arm kind {kind!r}")


def interleaved_benchmark(
    model, arms: list[dict], prompts, tokens: int, repeats: int, warmup: int = 1,
    keep_traces: bool = False,
) -> list[dict]:
    """The measurement core. Every arm runs on every prompt on every repeat, with the
    arm loop innermost so that arms are never separated in time.

    The AR arm must come first: its output is the reference every other arm's
    losslessness is checked against, and that check is the point of the harness.
    """
    if arms[0]["kind"] != "ar":
        raise ValueError("arms[0] must be the AR reference arm")

    print(f"[spec] warm-up ({warmup} pass{'es' if warmup != 1 else ''})")
    for _ in range(warmup):
        for _, _, ids in prompts[: min(3, len(prompts))]:
            for arm in arms:
                run_arm(model, arm, ids, min(8, tokens))

    rows: list[dict] = []
    total = repeats * len(prompts)
    done = 0
    started = time.perf_counter()
    for repeat in range(repeats):
        for category, text, ids in prompts:
            reference = None
            rep = None
            for arm in arms:
                out, stats, extra = run_arm(model, arm, ids, tokens)
                if arm["kind"] == "ar":
                    reference = out
                    rep = repetition_stats(out)
                divergence = next(
                    (i for i, (a, b) in enumerate(zip(out, reference)) if a != b), None
                )
                row = stats.to_dict()
                row.update(
                    {
                        "arm": arm["name"],
                        "kind": arm["kind"],
                        "repeat": repeat,
                        "category": category,
                        "prompt": text,
                        "prompt_tokens": len(ids),
                        # Measured on the AR reference, so every arm on this prompt
                        # carries the same value and the stratification is by prompt
                        # difficulty rather than by arm behaviour.
                        "ar_looped_fraction": rep["looped_fraction"],
                        "ar_distinct_ratio": rep["distinct_ratio"],
                        "identical_to_ar": out == reference,
                        "first_divergence": divergence,
                        # Kept only when something diverged, so the audit pass can
                        # recompute the target's own logits at exactly that position.
                        "divergence_context": (
                            list(ids) + list(reference[:divergence])
                            if divergence is not None else None
                        ),
                        "divergence_tokens": (
                            [reference[divergence], out[divergence]]
                            if divergence is not None else None
                        ),
                        "accepted_per_cycle": stats.accepted_per_cycle,
                        "committed_per_cycle": stats.committed_per_cycle,
                        "entropy_per_cycle": [round(e, 5) for e in stats.entropy_per_cycle],
                    }
                )
                for field in ("k", "refine", "policy", "order", "draft", "verify", "staged"):
                    if field in arm:
                        row[field] = arm[field]
                if keep_traces and extra.get("trace"):
                    row["trace"] = extra["trace"]
                rows.append(row)
            done += 1
            if done % 5 == 0 or done == total:
                rate = (time.perf_counter() - started) / done
                print(
                    f"  [{done}/{total}] {rate:.1f}s/prompt-set, "
                    f"~{rate * (total - done) / 60:.1f} min left",
                    flush=True,
                )
    return rows


def audit_divergences(model, rows: list[dict]) -> list[dict]:
    """Classify every divergence: numerical tie, or a real defect?

    Speculative decoding guarantees the committed token is the *verifier's* argmax.
    Native AR computes that argmax in a width-1 forward; the verify pass computes it in
    a width-(L+1) forward. Those are different reduction orders over the same maths, and
    the logits are bfloat16 — one ULP at logit magnitude ~25 is 0.125. When the top two
    logits land within that, the two paths can legitimately disagree, and *neither is
    wrong*: both are "the frozen model's greedy next token".

    So a divergence is only a bug if the target has a **clear** preference at that
    position. This recomputes the full-context logits and reports the top-2 gap, which
    separates the two cases with a number instead of an argument.

    Act IV-U4's gate U4-B4 tested the *cacheless* regime, where the AR reference and the
    speculative path use the same forward width and ties therefore break identically.
    That is why this never appeared before, and it does not invalidate U4 — it means
    U4's guarantee was the narrower one.
    """
    import mlx.core as mx

    audited = []
    for row in rows:
        if row.get("first_divergence") is None or not row.get("divergence_context"):
            continue
        ctx = mx.array([row["divergence_context"]], dtype=mx.int32)
        logits = model.ar_logits(ctx)[0, -1]
        native_dtype = str(logits.dtype)
        row_f32 = logits.astype(mx.float32)
        top2 = mx.topk(row_f32, 2)
        mx.eval(top2)
        gap = float(top2[-1].item() - top2[-2].item())
        ar_token, spec_token = row["divergence_tokens"]
        ar_logit = float(row_f32[ar_token].item())
        spec_logit = float(row_f32[spec_token].item())
        audited.append(
            {
                "arm": row["arm"],
                "category": row["category"],
                "prompt": row["prompt"][:70],
                "position": row["first_divergence"],
                "ar_token": ar_token,
                "spec_token": spec_token,
                "logit_dtype": native_dtype,
                "top2_gap": gap,
                "contested_logit_gap": abs(ar_logit - spec_logit),
                # Exactly equal in the model's own dtype: no information distinguishes
                # the two tokens, and the choice is pure tie-breaking order.
                "numerical_tie": abs(ar_logit - spec_logit) == 0.0,
            }
        )
    return audited


def check_losslessness(rows: list[dict], audit: list[dict] | None = None) -> dict:
    """Gate 1, computed over every non-AR row in the run.

    Two verdicts are reported and never conflated: `lossless` (byte-identical token
    IDs) and `lossless_modulo_ties` (identical except where the target's top-2 logits
    are exactly equal in bf16, so both continuations are the model's own greedy path).
    """
    spec = [r for r in rows if r["kind"] != "ar"]
    bad = [r for r in spec if not r["identical_to_ar"]]
    audit = audit or []
    ties = {
        (a["arm"], a["prompt"], a["position"]) for a in audit if a["numerical_tie"]
    }
    real = [
        r for r in bad
        if (r["arm"], r["prompt"][:70], r["first_divergence"]) not in ties
    ]
    return {
        "rows_checked": len(spec),
        "identical": len(spec) - len(bad),
        "divergent": len(bad),
        "divergent_from_numerical_ties": len(bad) - len(real),
        "divergent_unexplained": len(real),
        "lossless": not bad,
        "lossless_modulo_ties": not real,
        "failures": [
            {
                "arm": r["arm"], "category": r["category"], "prompt": r["prompt"][:60],
                "first_divergence": r["first_divergence"],
            }
            for r in bad[:20]
        ],
        "audit": audit[:50],
    }


def measure_tie_rate(model, prompts, tokens: int, limit: int = 12) -> dict:
    """How often does the target have *no* strict preference for a next token?

    This is the base rate that bounds exact token identity for any decoder whose
    forward width differs from native AR's. Measured in one forward per prompt: a
    causal pass over `prompt + continuation` exposes the next-token distribution after
    every prefix at once, which is the same trick the verifier uses.

    Reported as a property of the *target model in bf16*, not of the adapter. Nothing
    about speculative decoding can drive it to zero.
    """
    import mlx.core as mx

    from qdif.uno.decode import ar_greedy_generate

    total, tied, gaps = 0, 0, []
    for category, _text, ids in prompts[:limit]:
        out, _ = ar_greedy_generate(model, ids, max_tokens=tokens)
        full = mx.array([list(ids) + list(out)], dtype=mx.int32)
        logits = model.ar_logits(full)[0].astype(mx.float32)
        # Position len(ids)-1+j predicts generated token j.
        start = len(ids) - 1
        rows = logits[start : start + len(out)]
        top2 = mx.topk(rows, 2, axis=-1)
        mx.eval(top2)
        gap = (top2[:, -1] - top2[:, -2]).tolist()
        for g in gap:
            total += 1
            gaps.append(float(g))
            if float(g) == 0.0:
                tied += 1
    ordered = sorted(gaps)
    return {
        "positions": total,
        "exact_ties": tied,
        "tie_rate": round(tied / total, 5) if total else 0.0,
        "gap_median": round(statistics.median(ordered), 5) if ordered else 0.0,
        "gap_p05": round(ordered[int(0.05 * len(ordered))], 5) if ordered else 0.0,
        "note": (
            "Fraction of decoded positions where the target's top-2 logits are exactly "
            "equal in its native dtype. At these positions two different forward widths "
            "may pick different tokens and both are the model's own greedy output."
        ),
    }


def clean_rows(rows: list[dict]) -> list[dict]:
    """Rows whose AR reference continuation is not a loop. The defensible subset."""
    return [r for r in rows if r.get("ar_looped_fraction", 0.0) < CLEAN_THRESHOLD]


def losslessness_report(model, rows: list[dict]) -> dict:
    """Audit every divergence, then score Gate 1. Cheap: the audit touches only the
    rows that actually diverged."""
    audit = audit_divergences(model, rows)
    return check_losslessness(rows, audit)


def aggregate(rows: list[dict], by: tuple[str, ...] = ("arm",)) -> list[dict]:
    """Group rows and summarise the metrics that matter, with dispersion."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in by), []).append(row)

    out = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        accepted = [a for r in items for a in r["accepted_per_cycle"]]
        committed = [c for r in items for c in r["committed_per_cycle"]]
        record = dict(zip(by, key))
        record.update(
            {
                "rows": len(items),
                "tokens_per_second": summarize([r["tokens_per_second"] for r in items]),
                # Steady-state decode rate, prefill removed. At 16K context the
                # prompt prefill is ~14 s against ~3 s of generation and is *identical*
                # for every arm, so end-to-end ratios are dragged toward 1.0 by a cost
                # no decoder can change. Both are reported: end-to-end is what a user
                # feels for one request, decode-only is the property of the decoder.
                "decode_tokens_per_second": summarize(
                    [
                        r["tokens"] / max(r["wall_seconds"] - r["prefill_seconds"], 1e-9)
                        for r in items
                    ]
                ),
                "tokens_per_forward": summarize([r["tokens_per_forward"] for r in items]),
                "forwards": summarize([float(r["forwards"]) for r in items]),
                "ms_per_token": summarize(
                    [1000.0 * r["wall_seconds"] / max(r["tokens"], 1) for r in items]
                ),
                "ttft_ms": summarize([1000.0 * r["prefill_seconds"] for r in items]),
                "acceptance_rate": summarize([r["acceptance_rate"] for r in items]),
                "mean_accepted_prefix": round(
                    statistics.fmean(accepted), 4) if accepted else 0.0,
                "mean_committed_per_cycle": round(
                    statistics.fmean(committed), 4) if committed else 0.0,
                "cycles": sum(r["cycles"] for r in items),
                "lossless": all(r["identical_to_ar"] for r in items),
                "proposal_ms_per_cycle": summarize(
                    [r["proposal_ms_per_cycle"] for r in items]),
                "verify_ms_per_cycle": summarize([r["verify_ms_per_cycle"] for r in items]),
                "commit_ms_per_cycle": summarize([r["commit_ms_per_cycle"] for r in items]),
                "overhead_ms_per_cycle": summarize(
                    [r["overhead_ms_per_cycle"] for r in items]),
                "accepted_histogram": histogram(accepted),
                "survival": survival(accepted),
            }
        )
        out.append(record)
    return out


def histogram(accepted: list[int]) -> dict:
    if not accepted:
        return {}
    n = len(accepted)
    counts: dict[int, int] = {}
    for a in accepted:
        counts[a] = counts.get(a, 0) + 1
    return {str(k): round(v / n, 5) for k, v in sorted(counts.items())}


def survival(accepted: list[int]) -> dict:
    """`P(accepted >= j)` — the prefix-survival curve. Its area is `E[accepted]`."""
    if not accepted:
        return {}
    n = len(accepted)
    top = max(accepted)
    return {
        str(j): round(sum(1 for a in accepted if a >= j) / n, 5)
        for j in range(1, top + 2)
    }


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"[spec] wrote {path}")


def speedup_table(summary: list[dict], ar_name: str = "ar") -> list[dict]:
    ar = next((s for s in summary if s.get("arm") == ar_name), None)
    if ar is None:
        return summary
    base = ar["tokens_per_second"]["median"]
    base_decode = ar.get("decode_tokens_per_second", {}).get("median")
    for s in summary:
        s["speedup_vs_ar"] = round(s["tokens_per_second"]["median"] / base, 4)
        if base_decode:
            s["decode_speedup_vs_ar"] = round(
                s["decode_tokens_per_second"]["median"] / base_decode, 4
            )
    return summary


# ------------------------------------------------------------------ commands


def cmd_ksweep(args) -> int:
    """Gates 2-4: the K sweep with a K=1 control, stratified by prompt class."""
    model = load(args.adapter, args.rank)
    prompts = suite_prompts(model.tokenizer)
    ks = [int(k) for k in args.block_sizes.split(",")]
    arms = [{"name": "ar", "kind": "ar"}]
    arms += [{"name": f"uno_k{k}", "kind": "uno", "k": k} for k in ks]

    print(f"[spec] {len(prompts)} prompts x {len(arms)} arms x {args.repeats} repeats "
          f"@ {args.tokens} tokens")
    rows = interleaved_benchmark(
        model, arms, prompts, args.tokens, args.repeats, args.warmup
    )
    payload = {
        "provenance": provenance(model),
        "config": vars(args),
        "losslessness": losslessness_report(model, rows),
        "summary": speedup_table(aggregate(rows, ("arm",))),
        "summary_clean": speedup_table(aggregate(clean_rows(rows), ("arm",))),
        "by_category": aggregate(rows, ("arm", "category")),
        "rows": rows,
    }
    write(RESULTS / f"{args.tag}_ksweep.json", payload)
    report_summary(payload["summary"], payload["losslessness"])
    print("\n-- clean subset (AR continuation not looping) --")
    report_summary(payload["summary_clean"], payload["losslessness"])
    return 0


def cmd_context(args) -> int:
    """Brief §19: does the speedup grow with context length?"""
    model = load(args.adapter, args.rank)
    lengths = [int(x) for x in args.lengths.split(",")]
    ks = [int(k) for k in args.block_sizes.split(",")]

    all_rows = []
    for length in lengths:
        ids = long_context_prompt(model.tokenizer, length)
        prompts = [("wikitext", f"<{length}-token held-out context>", ids)]
        arms = [{"name": "ar", "kind": "ar"}]
        arms += [{"name": f"uno_k{k}", "kind": "uno", "k": k} for k in ks]
        print(f"\n[spec] context {length} tokens")
        rows = interleaved_benchmark(
            model, arms, prompts, args.tokens, args.repeats, args.warmup
        )
        for row in rows:
            row["context_length"] = length
        all_rows.extend(rows)

    summary = aggregate(all_rows, ("context_length", "arm"))
    for length in lengths:
        at = [s for s in summary if s["context_length"] == length]
        speedup_table(at)
    payload = {
        "provenance": provenance(model),
        "config": vars(args),
        "losslessness": losslessness_report(model, all_rows),
        "summary": summary,
        "rows": all_rows,
    }
    write(RESULTS / f"{args.tag}_context.json", payload)
    for s in summary:
        print(f"  ctx={s['context_length']:>6} {s['arm']:<10} "
              f"{s['tokens_per_second']['median']:>7.2f} tok/s  "
              f"x{s.get('speedup_vs_ar', 0):.3f}  TTFT {s['ttft_ms']['median']:.0f} ms")
    return 0


def cmd_controls(args) -> int:
    """Brief §14/§17: refinement steps, the n-gram drafter, and the untrained adapter."""
    model = load(args.adapter, args.rank)
    prompts = suite_prompts(model.tokenizer)
    k = args.k
    arms = [
        {"name": "ar", "kind": "ar"},
        {"name": f"uno_k{k}", "kind": "uno", "k": k},
        {"name": f"refine1_k{k}", "kind": "refine", "k": k, "refine": 1},
        {"name": f"refine2_k{k}", "kind": "refine", "k": k, "refine": 2},
        {"name": f"refine4_k{k}", "kind": "refine", "k": k, "refine": 4},
        {"name": f"ngram3_k{k}", "kind": "ngram", "k": k, "order": 3},
        {"name": f"ngram2_k{k}", "kind": "ngram", "k": k, "order": 2},
    ]
    rows = interleaved_benchmark(
        model, arms, prompts, args.tokens, args.repeats, args.warmup
    )
    payload = {
        "provenance": provenance(model),
        "config": vars(args),
        "losslessness": losslessness_report(model, rows),
        "summary": speedup_table(aggregate(rows, ("arm",))),
        "summary_clean": speedup_table(aggregate(clean_rows(rows), ("arm",))),
        "by_category": aggregate(rows, ("arm", "category")),
        "rows": rows,
    }
    write(RESULTS / f"{args.tag}_controls.json", payload)
    report_summary(payload["summary"], payload["losslessness"])
    print("\n-- clean subset (AR continuation not looping) --")
    report_summary(payload["summary_clean"], payload["losslessness"])
    return 0


def cmd_adaptive(args) -> int:
    """Gate 5: does an adaptive-K policy beat the best fixed K?"""
    model = load(args.adapter, args.rank)
    prompts = suite_prompts(model.tokenizer)
    allowed = [int(k) for k in args.allowed.split(",")]
    cost_ms = json.loads(args.cost_ms) if args.cost_ms else {}
    cost_ms = {int(k): float(v) for k, v in cost_ms.items()}

    arms = [{"name": "ar", "kind": "ar"}]
    arms += [{"name": f"uno_k{k}", "kind": "uno", "k": k} for k in allowed]
    arms += [
        {"name": "adapt_entropy", "kind": "adaptive", "policy": "entropy",
         "allowed": allowed, "default_k": args.default_k,
         "policy_kwargs": {"low": args.entropy_low, "high": args.entropy_high}},
        {"name": "adapt_survival", "kind": "adaptive", "policy": "survival",
         "allowed": allowed, "default_k": args.default_k},
        {"name": "adapt_ev", "kind": "adaptive", "policy": "ev",
         "allowed": allowed, "default_k": args.default_k,
         "policy_kwargs": {"cost_ms": cost_ms} if cost_ms else {}},
    ]
    rows = interleaved_benchmark(
        model, arms, prompts, args.tokens, args.repeats, args.warmup, keep_traces=True
    )
    payload = {
        "provenance": provenance(model),
        "config": vars(args),
        "losslessness": losslessness_report(model, rows),
        "summary": speedup_table(aggregate(rows, ("arm",))),
        "summary_clean": speedup_table(aggregate(clean_rows(rows), ("arm",))),
        "by_category": aggregate(rows, ("arm", "category")),
        "rows": rows,
    }
    write(RESULTS / f"{args.tag}_adaptive.json", payload)
    report_summary(payload["summary"], payload["losslessness"])
    print("\n-- clean subset (AR continuation not looping) --")
    report_summary(payload["summary_clean"], payload["losslessness"])
    return 0


def cmd_confidence(args) -> int:
    """Brief §10: how predictive are the drafter's free confidence features?"""
    model = load(args.adapter, args.rank)
    prompts = suite_prompts(model.tokenizer)
    k = args.k
    arms = [
        {"name": "ar", "kind": "ar"},
        {"name": f"conf_k{k}", "kind": "refine", "k": k, "refine": 1, "confidence": True},
    ]
    rows = interleaved_benchmark(
        model, arms, prompts, args.tokens, args.repeats, args.warmup, keep_traces=True
    )
    payload = {
        "provenance": provenance(model),
        "config": vars(args),
        "losslessness": losslessness_report(model, rows),
        "summary": aggregate(rows, ("arm",)),
        "rows": rows,
    }
    write(RESULTS / f"{args.tag}_confidence.json", payload)
    report_summary(payload["summary"], payload["losslessness"])
    return 0


def cmd_baseline(args) -> int:
    """Gate 0: the native AR baseline at several generation lengths."""
    model = load(args.adapter, args.rank)
    prompts = suite_prompts(model.tokenizer)
    out_rows = []
    for tokens in [int(t) for t in args.token_counts.split(",")]:
        arms = [{"name": "ar", "kind": "ar"}, {"name": "uno_k4", "kind": "uno", "k": 4}]
        print(f"\n[spec] generation length {tokens}")
        rows = interleaved_benchmark(model, arms, prompts, tokens, args.repeats, args.warmup)
        for row in rows:
            row["max_tokens"] = tokens
        out_rows.extend(rows)
    summary = aggregate(out_rows, ("max_tokens", "arm"))
    payload = {
        "provenance": provenance(model),
        "config": vars(args),
        "tie_rate": measure_tie_rate(model, prompts, 128),
        "losslessness": losslessness_report(model, out_rows),
        "summary": summary,
        "by_category": aggregate(out_rows, ("max_tokens", "arm", "category")),
        "rows": out_rows,
    }
    write(RESULTS / f"{args.tag}_baseline.json", payload)
    for s in summary:
        print(f"  n={s['max_tokens']:>4} {s['arm']:<8} "
              f"{s['tokens_per_second']['median']:>7.2f} tok/s "
              f"(sd {s['tokens_per_second']['sd']:.2f}) "
              f"TTFT {s['ttft_ms']['median']:.1f} ms  lossless={s['lossless']}")
    tr = payload["tie_rate"]
    print(f"\n  bf16 top-2 tie rate: {tr['exact_ties']}/{tr['positions']} "
          f"= {100 * tr['tie_rate']:.2f}% of decoded positions")
    return 0


def report_summary(summary: list[dict], lossless: dict) -> None:
    print("\n-- summary (median over prompts x repeats) --")
    print(f"{'arm':<16}{'tok/s':>9}{'sd':>7}{'xAR':>8}{'TPF':>8}"
          f"{'accept':>8}{'prefix':>8}{'lossless':>10}")
    for s in summary:
        print(
            f"{str(s.get('arm')):<16}"
            f"{s['tokens_per_second']['median']:>9.2f}"
            f"{s['tokens_per_second']['sd']:>7.2f}"
            f"{s.get('speedup_vs_ar', float('nan')):>8.3f}"
            f"{s['tokens_per_forward']['median']:>8.3f}"
            f"{s['acceptance_rate']['median']:>8.3f}"
            f"{s['mean_accepted_prefix']:>8.3f}"
            f"{str(s['lossless']):>10}"
        )
    print(f"\nlosslessness: {lossless['identical']}/{lossless['rows_checked']} rows "
          f"identical to native AR -> {'PASS' if lossless['lossless'] else 'FAIL'}")
    if lossless["failures"]:
        for f in lossless["failures"]:
            print(f"  DIVERGED {f['arm']} [{f['category']}] at token {f['first_divergence']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER,
                        help="adapter safetensors; '' for the untrained control")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tag", default="main")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("baseline", help="Gate 0: native AR at several lengths")
    p.add_argument("--token-counts", default="128,256,512")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("ksweep", help="Gates 2-4: K sweep by prompt class")
    p.add_argument("--block-sizes", default="1,2,4,8,16")
    p.set_defaults(func=cmd_ksweep)

    p = sub.add_parser("context", help="Brief 19: speedup vs context length")
    p.add_argument("--lengths", default="512,2048,8192,16384")
    p.add_argument("--block-sizes", default="2,4,8")
    p.set_defaults(func=cmd_context)

    p = sub.add_parser("controls", help="Brief 14/17: refinement and n-gram controls")
    p.add_argument("--k", type=int, default=4)
    p.set_defaults(func=cmd_controls)

    p = sub.add_parser("adaptive", help="Gate 5: adaptive K vs best fixed K")
    p.add_argument("--allowed", default="2,4,8")
    p.add_argument("--default-k", type=int, default=4)
    p.add_argument("--entropy-low", type=float, default=0.5)
    p.add_argument("--entropy-high", type=float, default=2.0)
    p.add_argument("--cost-ms", default="", help='JSON, e.g. {"2":45.8,"4":49.2,"8":55.0}')
    p.set_defaults(func=cmd_adaptive)

    p = sub.add_parser("confidence", help="Brief 10: free confidence features")
    p.add_argument("--k", type=int, default=4)
    p.set_defaults(func=cmd_confidence)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
