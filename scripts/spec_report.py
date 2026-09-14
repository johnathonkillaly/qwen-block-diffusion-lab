#!/usr/bin/env python
"""Machine-readable tables and the brief's §25 plots, from `results/speculative/*.json`.

Runs on the committed JSON alone — no model, no MLX, seconds. That separation is
deliberate: the expensive measurement happens once and every later question is asked of
the recorded data rather than by re-running the GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results" / "speculative"
PLOTS = ROOT / "plots" / "speculative"


def load(tag: str, name: str, by: tuple[str, ...] = ("arm",)) -> dict | None:
    """Load a result file and **recompute** its summaries from the raw rows.

    Recomputing rather than trusting the stored `summary` means a metric added to
    `aggregate()` after a run (such as prefill-excluded decode throughput) appears in
    every earlier result without re-running an hour of GPU work. The raw rows are the
    record; the summaries are a view of them.
    """
    path = RESULTS / f"{tag}_{name}.json"
    if not path.exists():
        print(f"[report] missing {path.name} — skipping")
        return None
    data = json.loads(path.read_text())
    from spec_decode import aggregate, clean_rows, speedup_table

    rows = data["rows"]
    if "context_length" in (rows[0] if rows else {}):
        summary = aggregate(rows, ("context_length", "arm"))
        for length in sorted({r["context_length"] for r in rows}):
            speedup_table([s for s in summary if s["context_length"] == length])
        data["summary"] = summary
    elif "max_tokens" in (rows[0] if rows else {}):
        summary = aggregate(rows, ("max_tokens", "arm"))
        for n in sorted({r["max_tokens"] for r in rows}):
            speedup_table([s for s in summary if s["max_tokens"] == n])
        data["summary"] = summary
        data["by_category"] = aggregate(rows, ("max_tokens", "arm", "category"))
    else:
        data["summary"] = speedup_table(aggregate(rows, ("arm",)))
        data["summary_clean"] = speedup_table(aggregate(clean_rows(rows), ("arm",)))
        data["by_category"] = aggregate(rows, ("arm", "category"))
    return data


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[report] wrote {path.relative_to(ROOT)}  ({len(rows)} rows)")


def flat(summary: list[dict], extra: tuple[str, ...] = ()) -> list[dict]:
    """Flatten the nested summarise() dicts into one CSV row per group."""
    out = []
    for s in summary:
        row = {k: s.get(k) for k in ("arm", "category", "context_length",
                                     "max_tokens", *extra) if k in s}
        row.update(
            {
                "rows": s.get("rows"),
                "tok_s_median": s["tokens_per_second"]["median"],
                "tok_s_mean": s["tokens_per_second"]["mean"],
                "tok_s_sd": s["tokens_per_second"]["sd"],
                "decode_tok_s_median": s.get("decode_tokens_per_second", {}).get("median"),
                "decode_speedup_vs_ar": s.get("decode_speedup_vs_ar"),
                "tok_s_p25": s["tokens_per_second"]["p25"],
                "tok_s_p75": s["tokens_per_second"]["p75"],
                "speedup_vs_ar": s.get("speedup_vs_ar"),
                "tpf_median": s["tokens_per_forward"]["median"],
                "ms_per_token_median": s["ms_per_token"]["median"],
                "ttft_ms_median": s["ttft_ms"]["median"],
                "acceptance_rate": s["acceptance_rate"]["median"],
                "mean_accepted_prefix": s["mean_accepted_prefix"],
                "mean_committed_per_cycle": s["mean_committed_per_cycle"],
                "cycles": s.get("cycles"),
                "proposal_ms": s["proposal_ms_per_cycle"]["median"],
                "verify_ms": s["verify_ms_per_cycle"]["median"],
                "commit_ms": s["commit_ms_per_cycle"]["median"],
                "overhead_ms": s["overhead_ms_per_cycle"]["median"],
                "lossless": s.get("lossless"),
            }
        )
        out.append(row)
    return out


def k_of(arm: str) -> int | None:
    if "_k" in arm:
        try:
            return int(arm.rsplit("_k", 1)[1])
        except ValueError:
            return None
    return None


# ------------------------------------------------------------------- plotting


def plots(tag: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[report] matplotlib unavailable — tables only")
        return

    PLOTS.mkdir(parents=True, exist_ok=True)
    ks = load(tag, "ksweep")
    ctx = load(tag, "context")
    ctrl = load(tag, "controls")
    adapt = load(tag, "adaptive")

    def save(fig, name: str) -> None:
        fig.tight_layout()
        path = PLOTS / name
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"[report] wrote {path.relative_to(ROOT)}")

    # 1 native vs speculative tok/s; 2 speedup vs K; 3 accepted vs K
    if ks:
        rows = [r for r in flat(ks["summary"]) if r["arm"] != "ar"]
        rows = sorted([r for r in rows if k_of(r["arm"]) is not None],
                      key=lambda r: k_of(r["arm"]))
        ar = next(r for r in flat(ks["summary"]) if r["arm"] == "ar")
        kk = [k_of(r["arm"]) for r in rows]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(["native AR"] + [f"K={k}" for k in kk],
               [ar["tok_s_median"]] + [r["tok_s_median"] for r in rows],
               yerr=[ar["tok_s_sd"]] + [r["tok_s_sd"] for r in rows],
               color=["#888"] + ["#2b7bba"] * len(rows), capsize=3)
        ax.axhline(ar["tok_s_median"], ls="--", c="k", lw=0.8)
        ax.set_ylabel("committed tok/s (median)")
        ax.set_title("1. Native AR vs speculative throughput")
        save(fig, "01_native_vs_speculative.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(kk, [r["speedup_vs_ar"] for r in rows], "o-")
        ax.axhline(1.0, ls="--", c="k", lw=0.8, label="native AR")
        ax.set_xscale("log", base=2); ax.set_xticks(kk); ax.set_xticklabels(kk)
        ax.set_xlabel("block size K"); ax.set_ylabel("speedup vs native AR")
        ax.set_title("2. Speedup vs K"); ax.legend()
        save(fig, "02_speedup_vs_k.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(kk, [r["mean_accepted_prefix"] for r in rows], "o-", label="accepted")
        ax.plot(kk, [r["mean_committed_per_cycle"] for r in rows], "s--", label="committed")
        ax.set_xscale("log", base=2); ax.set_xticks(kk); ax.set_xticklabels(kk)
        ax.set_xlabel("block size K"); ax.set_ylabel("tokens per cycle")
        ax.set_title("3. Mean accepted / committed vs K"); ax.legend()
        save(fig, "03_accepted_vs_k.png")

        # 4 prefix survival
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for s in sorted(ks["summary"], key=lambda s: k_of(s["arm"]) or 0):
            if s["arm"] == "ar" or not s.get("survival"):
                continue
            js = sorted(int(j) for j in s["survival"])
            ax.plot(js, [s["survival"][str(j)] for j in js], "o-", label=s["arm"])
        ax.set_xlabel("draft position j"); ax.set_ylabel("P(accepted >= j)")
        ax.set_title("4. Prefix survival vs draft position"); ax.legend(); ax.grid(alpha=.3)
        save(fig, "04_prefix_survival.png")

        # 5/6/7 cost decomposition
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(kk, [r["proposal_ms"] for r in rows], "o-", label="draft (proposal)")
        ax.plot(kk, [r["verify_ms"] for r in rows], "s-", label="verify")
        ax.plot(kk, [r["commit_ms"] for r in rows], "^-", label="commit")
        ax.plot(kk, [r["overhead_ms"] for r in rows], "v-", label="overhead")
        ax.set_xscale("log", base=2); ax.set_xticks(kk); ax.set_xticklabels(kk)
        ax.set_xlabel("block size K"); ax.set_ylabel("ms per cycle")
        ax.set_title("5/6. Draft and verify latency vs K"); ax.legend(); ax.grid(alpha=.3)
        save(fig, "05_06_stage_latency_vs_k.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(kk, [r["ms_per_token_median"] for r in rows], "o-", label="speculative")
        ax.axhline(ar["ms_per_token_median"], ls="--", c="k", label="native AR")
        ax.set_xscale("log", base=2); ax.set_xticks(kk); ax.set_xticklabels(kk)
        ax.set_xlabel("block size K"); ax.set_ylabel("ms per committed token")
        ax.set_title("7. Committed ms/token vs K"); ax.legend(); ax.grid(alpha=.3)
        save(fig, "07_ms_per_token_vs_k.png")

        # 9 acceptance by prompt category
        by_cat = flat(ks["by_category"])
        cats = sorted({r["category"] for r in by_cat})
        fig, ax = plt.subplots(figsize=(9, 4.5))
        width = 0.8 / max(len(kk), 1)
        for i, k in enumerate(kk):
            vals = [
                next((r["mean_accepted_prefix"] for r in by_cat
                      if r["category"] == c and k_of(r["arm"]) == k), 0)
                for c in cats
            ]
            ax.bar([x + i * width for x in range(len(cats))], vals, width, label=f"K={k}")
        ax.set_xticks([x + 0.4 - width / 2 for x in range(len(cats))])
        ax.set_xticklabels(cats, rotation=30, ha="right")
        ax.set_ylabel("mean accepted prefix")
        ax.set_title("9. Acceptance by prompt category"); ax.legend(fontsize=8)
        save(fig, "09_acceptance_by_category.png")

    # 8 refinement steps vs acceptance
    if ctrl:
        rows = flat(ctrl["summary"])
        ref = sorted([r for r in rows if r["arm"].startswith("refine")],
                     key=lambda r: int(r["arm"][6]))
        if ref:
            steps = [int(r["arm"][6]) for r in ref]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(steps, [r["mean_accepted_prefix"] for r in ref], "o-", label="accepted prefix")
            ax2 = ax.twinx()
            ax2.plot(steps, [r["tok_s_median"] for r in ref], "s--", c="crimson",
                     label="tok/s")
            ar = next((r for r in rows if r["arm"] == "ar"), None)
            if ar:
                ax2.axhline(ar["tok_s_median"], ls=":", c="k")
            ax.set_xlabel("refinement steps"); ax.set_ylabel("mean accepted prefix")
            ax2.set_ylabel("tok/s (median)", color="crimson")
            ax.set_xticks(steps)
            ax.set_title("8. Refinement steps vs acceptance and speed")
            save(fig, "08_refinement_steps.png")

        fig, ax = plt.subplots(figsize=(7.5, 4))
        named = [r for r in rows if r["arm"] != "ar"]
        ax.bar([r["arm"] for r in named], [r["speedup_vs_ar"] or 0 for r in named],
               color=["#2b7bba" if "ngram" not in r["arm"] else "#d95f02" for r in named])
        ax.axhline(1.0, ls="--", c="k", lw=0.8)
        ax.set_ylabel("speedup vs native AR"); ax.tick_params(axis="x", rotation=30)
        ax.set_title("Controls: diffusion drafter vs n-gram lookup")
        save(fig, "13_controls.png")

    # 10 speedup vs context length
    if ctx:
        rows = flat(ctx["summary"])
        lengths = sorted({r["context_length"] for r in rows})
        arms = sorted({r["arm"] for r in rows if r["arm"] != "ar"},
                      key=lambda a: k_of(a) or 0)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for arm in arms:
            ys = [next((r["speedup_vs_ar"] for r in rows
                        if r["context_length"] == L and r["arm"] == arm), None)
                  for L in lengths]
            ax.plot(lengths, ys, "o-", label=arm)
        ax.axhline(1.0, ls="--", c="k", lw=0.8)
        ax.set_xscale("log", base=2); ax.set_xticks(lengths)
        ax.set_xticklabels([str(L) for L in lengths])
        ax.set_xlabel("context length (tokens)"); ax.set_ylabel("speedup vs native AR")
        ax.set_title("10. Speedup vs context length"); ax.legend(); ax.grid(alpha=.3)
        save(fig, "10_speedup_vs_context.png")

        fig, ax = plt.subplots(figsize=(6.5, 4))
        for arm in ["ar"] + arms:
            ys = [next((r["tok_s_median"] for r in rows
                        if r["context_length"] == L and r["arm"] == arm), None)
                  for L in lengths]
            ax.plot(lengths, ys, "o-", label=arm)
        ax.set_xscale("log", base=2); ax.set_xticks(lengths)
        ax.set_xticklabels([str(L) for L in lengths])
        ax.set_xlabel("context length (tokens)"); ax.set_ylabel("tok/s (median)")
        ax.set_title("10b. Absolute throughput vs context"); ax.legend(); ax.grid(alpha=.3)
        save(fig, "10b_throughput_vs_context.png")

    # 11 confidence vs actual survival; 12 fixed vs adaptive
    conf = load(tag, "confidence")
    if conf:
        pairs = []
        for row in conf["rows"]:
            for t in row.get("trace", []):
                if t["draft_top1"]:
                    pairs.append((statistics.fmean(t["draft_top1"]), t["accepted"]))
        if pairs:
            pairs.sort()
            n = len(pairs)
            bins = 10
            xs, ys, ns = [], [], []
            for b in range(bins):
                chunk = pairs[b * n // bins : (b + 1) * n // bins]
                if chunk:
                    xs.append(statistics.fmean(p[0] for p in chunk))
                    ys.append(statistics.fmean(p[1] for p in chunk))
                    ns.append(len(chunk))
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(xs, ys, "o-")
            ax.set_xlabel("mean drafter top-1 probability (decile bin)")
            ax.set_ylabel("mean accepted prefix")
            ax.set_title("11. Drafter confidence vs actual prefix survival")
            ax.grid(alpha=.3)
            save(fig, "11_confidence_calibration.png")
            write_csv(RESULTS / f"{tag}_confidence_bins.csv",
                      [{"bin": i, "mean_top1": x, "mean_accepted": y, "n": k}
                       for i, (x, y, k) in enumerate(zip(xs, ys, ns))])

    if adapt:
        rows = flat(adapt["summary"])
        named = [r for r in rows if r["arm"] != "ar"]
        fig, ax = plt.subplots(figsize=(7.5, 4))
        colors = ["#2b7bba" if r["arm"].startswith("uno") else "#7b3294" for r in named]
        ax.bar([r["arm"] for r in named], [r["tok_s_median"] for r in named],
               yerr=[r["tok_s_sd"] for r in named], color=colors, capsize=3)
        ar = next((r for r in rows if r["arm"] == "ar"), None)
        if ar:
            ax.axhline(ar["tok_s_median"], ls="--", c="k", lw=0.8, label="native AR")
        best = max((r for r in named if r["arm"].startswith("uno")),
                   key=lambda r: r["tok_s_median"], default=None)
        if best:
            ax.axhline(best["tok_s_median"], ls=":", c="#2b7bba",
                       label=f"best fixed ({best['arm']})")
        ax.set_ylabel("tok/s (median)"); ax.tick_params(axis="x", rotation=30)
        ax.set_title("12. Fixed-K vs adaptive-K throughput"); ax.legend(fontsize=8)
        save(fig, "12_fixed_vs_adaptive.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    for name, by in [
        ("baseline", ("max_tokens", "arm")),
        ("ksweep", ("arm",)),
        ("context", ("context_length", "arm")),
        ("controls", ("arm",)),
        ("adaptive", ("arm",)),
        ("confidence", ("arm",)),
    ]:
        data = load(args.tag, name)
        if not data:
            continue
        write_csv(RESULTS / f"{args.tag}_{name}_summary.csv", flat(data["summary"]))
        if data.get("summary_clean"):
            write_csv(RESULTS / f"{args.tag}_{name}_summary_clean.csv",
                      flat(data["summary_clean"]))
        if data.get("by_category"):
            write_csv(RESULTS / f"{args.tag}_{name}_by_category.csv",
                      flat(data["by_category"]))
        # survival curves as their own table
        surv = []
        for s in data["summary"]:
            for j, p in (s.get("survival") or {}).items():
                surv.append({"arm": s.get("arm"), "context_length": s.get("context_length"),
                             "j": int(j), "p_survive": p})
        if surv:
            write_csv(RESULTS / f"{args.tag}_{name}_survival.csv", surv)
        hist = []
        for s in data["summary"]:
            for a, p in (s.get("accepted_histogram") or {}).items():
                hist.append({"arm": s.get("arm"), "accepted": int(a), "probability": p})
        if hist:
            write_csv(RESULTS / f"{args.tag}_{name}_histogram.csv", hist)

    if not args.no_plots:
        plots(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
