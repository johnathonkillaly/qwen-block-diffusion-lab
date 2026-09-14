#!/usr/bin/env python
"""Turn `u3-eval` output into the Act IV-U3 tables and plots.

Reads the single-session evaluation JSON plus the training history, and writes
machine-readable CSV/JSON under `results/act4u3/` and the seven pre-registered plots.

Nothing here recomputes a metric — it only reshapes what was measured, so a number in
a plot is the same number as in the JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def load(evaluation: Path, training: Path | None):
    data = json.loads(evaluation.read_text())
    history = None
    if training and training.is_file():
        history = json.loads(training.read_text())
    return data, history


def checkpoint_step(name: str) -> int:
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else 0


def write_tables(data: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ar = data["ar_baseline"]["median_tokens_per_second"]
    rows = []

    for entry in data["checkpoints"]:
        step = checkpoint_step(entry["checkpoint"])
        for key, blob in entry["per_block"].items():
            k = int(key.lstrip("K"))
            tf, fr = blob["teacher_forced"], blob["free_running"]
            slots = tf["agree_per_slot"]
            row = {
                "checkpoint": entry["checkpoint"],
                "step": step,
                "K": k,
                "val_tv": tf["validation_tv"],
                "tf_agree_specs": tf["agree_specs"],
                "tf_mean_accepted_prefix": tf["mean_accepted_prefix"],
                "tf_full_block_rate": tf["full_block_rate"],
                "free_acceptance_rate": fr["acceptance_rate"],
                "free_mean_accepted_specs": fr["mean_accepted_specs"],
                "free_full_block_rate": fr["full_block_rate"],
                "tpf": fr["tokens_per_forward"],
                "forwards_per_token": fr["forwards_per_token"],
                "proposal_ms": fr["cycle_cost_ms"]["proposal_ms"],
                "verify_ms": fr["cycle_cost_ms"]["verify_ms"],
                "commit_ms": fr["cycle_cost_ms"]["commit_ms"],
                "tok_s_median": fr["median_tokens_per_second"],
                "tok_s_std": fr["std_tokens_per_second"],
                "speedup_vs_ar": fr["speedup_vs_ar"],
                "predicted_tok_s": fr["predicted_tokens_per_second"],
                "cost_model_error_pct": fr["cost_model_error_pct"],
                "ar_agreement": fr["ar_agreement"],
                "ar_baseline_tok_s": ar,
                "backbone_unchanged": entry["backbone_unchanged"],
            }
            for slot in range(1, 5):
                row[f"acc_plus_{slot}"] = slots[slot] if slot < len(slots) else ""
            for bucket in entry["draftability"]["buckets"]:
                row[f"draft_q{bucket['bucket'] + 1}_accept"] = bucket["agreement"]
            row["draft_pearson_gap"] = entry["draftability"]["pearson_gap"]
            row["draft_pearson_entropy"] = entry["draftability"]["pearson_entropy"]
            rows.append(row)

    rows.sort(key=lambda r: (r["K"], r["step"]))
    fields = list(rows[0].keys())
    with open(out / "u3_checkpoints.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # acceptance histograms, long form
    with open(out / "u3_acceptance_histogram.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["checkpoint", "step", "K", "accepted_specs", "cycles", "fraction"])
        for entry in data["checkpoints"]:
            for key, blob in entry["per_block"].items():
                hist = blob["free_running"]["acceptance_histogram"]
                total = sum(hist.values())
                for accepted, n in sorted(hist.items(), key=lambda kv: int(kv[0])):
                    writer.writerow([entry["checkpoint"], checkpoint_step(entry["checkpoint"]),
                                     int(key.lstrip("K")), accepted, n, round(n / total, 5)])

    # draftability quintiles, long form
    with open(out / "u3_draftability.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["checkpoint", "step", "block_size", "quintile", "n",
                         "mean_gap", "acceptance", "pearson_gap", "pearson_entropy"])
        for entry in data["checkpoints"]:
            d = entry["draftability"]
            for bucket in d["buckets"]:
                writer.writerow([entry["checkpoint"], checkpoint_step(entry["checkpoint"]),
                                 d["block_size"], bucket["bucket"] + 1, bucket["n"],
                                 bucket["mean_gap"], bucket["agreement"],
                                 d["pearson_gap"], d["pearson_entropy"]])

    # cost-model requirements
    with open(out / "u3_cost_model.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["checkpoint", "K", "target_speedup", "reachable",
                         "required_per_slot_accept", "required_mean_committed",
                         "required_full_block_rate", "required_tpf",
                         "actual_per_slot_accept", "actual_speedup"])
        for entry in data["checkpoints"]:
            for key, blob in entry["per_block"].items():
                fr = blob["free_running"]
                for req in fr["speedup_requirements"]:
                    writer.writerow([entry["checkpoint"], int(key.lstrip("K")),
                                     req["target_speedup"], req["reachable"],
                                     req["required_per_slot_accept"],
                                     req["required_mean_committed"],
                                     req["required_full_block_rate"], req["required_tpf"],
                                     fr["acceptance_rate"], fr["speedup_vs_ar"]])

    json.dump({"ar_baseline": data["ar_baseline"], "harness": data["harness"],
               "provenance": data["provenance"], "rows": rows},
              open(out / "u3_summary.json", "w"), indent=2)
    return {"rows": rows, "ar": ar}


def make_plots(data: dict, history: dict | None, rows: list[dict], ar: float, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    ks = sorted({r["K"] for r in rows})
    style = {2: dict(color="#2563eb", marker="o"), 4: dict(color="#dc2626", marker="s"),
             8: dict(color="#059669", marker="^")}

    def series(k, field):
        pts = [(r["step"], r[field]) for r in rows if r["K"] == k and r[field] != ""]
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]

    def line_plot(field, ylabel, title, filename, hline=None, hlabel=None):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for k in ks:
            x, y = series(k, field)
            ax.plot(x, y, label=f"K={k}", **style.get(k, {}))
        if hline is not None:
            ax.axhline(hline, ls="--", lw=1, color="#666", label=hlabel)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / filename, dpi=140)
        plt.close(fig)

    # 1 validation TV loss
    line_plot("val_tv", "held-out TV loss", "1. Validation TV loss vs training step",
              "plot1_val_tv.png")
    # 2 mean accepted prefix (free-running)
    line_plot("free_mean_accepted_specs", "mean accepted specs / cycle",
              "2. Free-running accepted prefix vs training step", "plot2_accepted_prefix.png")
    # 3 TPF
    line_plot("tpf", "tokens per forward", "3. TPF vs training step", "plot3_tpf.png",
              hline=1.0, hlabel="AR (1.0)")
    # 4 wall clock
    line_plot("tok_s_median", "tokens / second", "4. Wall-clock throughput vs training step",
              "plot4_tok_s.png", hline=ar, hlabel=f"AR ({ar:.1f} tok/s)")

    # 5 future offset vs accuracy, grouped by checkpoint
    fig, axes = plt.subplots(1, len(ks), figsize=(5.6 * len(ks), 4.2), squeeze=False)
    for ax, k in zip(axes[0], ks):
        entries = sorted([r for r in rows if r["K"] == k], key=lambda r: r["step"])
        cmap = plt.get_cmap("viridis")
        for index, row in enumerate(entries):
            offsets = [s for s in range(1, k + 1) if row.get(f"acc_plus_{s}", "") != ""]
            values = [row[f"acc_plus_{s}"] for s in offsets]
            ax.plot(offsets, values, marker="o",
                    color=cmap(index / max(len(entries) - 1, 1)),
                    label=f"step {row['step']}")
        ax.set_xlabel("future offset")
        ax.set_ylabel("teacher-forced agreement")
        ax.set_title(f"5. Accuracy vs future offset (K={k})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "plot5_future_offset.png", dpi=140)
    plt.close(fig)

    # 6 draftability quintile vs acceptance, grouped by checkpoint
    fig, ax = plt.subplots(figsize=(7, 4.2))
    entries = sorted(data["checkpoints"], key=lambda e: checkpoint_step(e["checkpoint"]))
    cmap = plt.get_cmap("viridis")
    for index, entry in enumerate(entries):
        buckets = entry["draftability"]["buckets"]
        ax.plot([b["bucket"] + 1 for b in buckets], [b["agreement"] for b in buckets],
                marker="o", color=cmap(index / max(len(entries) - 1, 1)),
                label=f"step {checkpoint_step(entry['checkpoint'])}")
    ax.set_xlabel("draftability-gap quintile (1 = most draftable)")
    ax.set_ylabel("teacher-forced agreement")
    ax.set_title("6. Acceptance by frozen-model draftability quintile")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plot6_draftability.png", dpi=140)
    plt.close(fig)

    # 7 predicted vs measured tok/s -- do we understand the decoder?
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    for k in ks:
        subset = [r for r in rows if r["K"] == k]
        ax.scatter([r["predicted_tok_s"] for r in subset],
                   [r["tok_s_median"] for r in subset],
                   label=f"K={k}", **{kk: vv for kk, vv in style.get(k, {}).items()
                                      if kk != "marker"},
                   marker=style.get(k, {}).get("marker", "o"), s=60)
    lo = min([r["predicted_tok_s"] for r in rows] + [r["tok_s_median"] for r in rows]) * 0.95
    hi = max([r["predicted_tok_s"] for r in rows] + [r["tok_s_median"] for r in rows]) * 1.05
    ax.plot([lo, hi], [lo, hi], ls="--", color="#666", lw=1, label="y = x")
    ax.set_xlabel("cost-model predicted tok/s")
    ax.set_ylabel("measured tok/s")
    ax.set_title("7. Cost model vs measurement")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "plot7_cost_model.png", dpi=140)
    plt.close(fig)

    # training curve, if available
    if history:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        steps = [h["step"] for h in history["history"]]
        tv = [h["tv"] for h in history["history"]]
        window = 25
        smoothed = [sum(tv[max(0, i - window):i + 1]) / len(tv[max(0, i - window):i + 1])
                    for i in range(len(tv))]
        ax.plot(steps, tv, lw=0.5, alpha=0.3, color="#94a3b8", label="train TV (raw)")
        ax.plot(steps, smoothed, lw=1.6, color="#2563eb", label=f"train TV ({window}-step mean)")
        if history.get("evals"):
            ax.plot([e["step"] for e in history["evals"]],
                    [e["tv"] for e in history["evals"]],
                    marker="o", color="#dc2626", label="held-out TV")
        ax.set_xlabel("training step")
        ax.set_ylabel("TV loss")
        ax.set_title("Training and held-out TV loss")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "plot0_training_loss.png", dpi=140)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Act IV-U3 tables and plots")
    parser.add_argument("--eval", default="runs/u3a/eval.json")
    parser.add_argument("--training", default="runs/u3a/result.json")
    parser.add_argument("--out", default="results/act4u3")
    args = parser.parse_args()

    data, history = load(REPO / args.eval, REPO / args.training)
    out = REPO / args.out
    tables = write_tables(data, out)
    make_plots(data, history, tables["rows"], tables["ar"], out)
    print(f"[u3] wrote tables and plots to {out}")
    for path in sorted(out.iterdir()):
        print(f"   {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
