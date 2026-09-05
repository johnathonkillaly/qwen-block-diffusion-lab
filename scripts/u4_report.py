#!/usr/bin/env python
"""Turn `u4-eval` output into the Act IV-U4 tables and plots.

Reads one or more single-session evaluation JSONs plus training histories, and writes
machine-readable CSV/JSON under `results/act4u4/` together with the twelve
pre-registered plots.

Nothing here recomputes a metric — it only reshapes what was measured, so a number in
a plot is the same number as in the JSON. Anything derived (the saturation test, the
offset-improvement table) is computed from the recorded values and written out as its
own CSV so the derivation is inspectable rather than embedded in a figure.

Offset convention, restated because U3 used a different one: **future offset +j is
supervised slot j-1**, and +1 is the adapter-off seed row whose agreement is 1.0 by
construction. U3's "+1" is this file's "+2".
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

STYLE = {
    2: dict(color="#2563eb", marker="o"),
    4: dict(color="#dc2626", marker="s"),
    6: dict(color="#d97706", marker="D"),
    8: dict(color="#059669", marker="^"),
}


def checkpoint_step(name: str) -> int:
    """The training step in a checkpoint label, from its **trailing** digit group.

    Concatenating every digit in the name works for `step-12800` and breaks for
    `B1-13200` (-> 113200) and `B2k6-14400` (-> 2614400), which silently reorders the
    series and makes the saturation intervals nonsense. The step is always the last
    run of digits.
    """
    groups = re.findall(r"\d+", name)
    return int(groups[-1]) if groups else 0


def checkpoint_arm(name: str, default: str) -> str:
    """Which experimental arm a checkpoint belongs to, from its label.

    The U4-B arms share step numbers (both continue from step 12800), so the session
    tag is not enough to keep them apart -- B1-14400 and B2k6-14400 would collide on
    (arm, K, step) and one would overwrite the other in every ordered view.
    """
    if name.startswith("B1"):
        return "B1"
    if name.startswith("B2"):
        return "B2"
    if name == "untrained":
        return "control"
    return default


def load_evals(paths: list[Path]) -> list[dict]:
    return [json.loads(p.read_text()) for p in paths if p.is_file()]


# ------------------------------------------------------------------- tables


def build_rows(datasets: list[dict]) -> list[dict]:
    rows = []
    for data in datasets:
        ar = data["ar_baseline"]["median_tokens_per_second"]
        session = data["provenance"]["timestamp"]
        session_arm = data.get("arm", "A")
        for entry in data["checkpoints"]:
            step = checkpoint_step(entry["checkpoint"])
            arm = checkpoint_arm(entry["checkpoint"], session_arm)
            for key, blob in entry["per_block"].items():
                k = int(key.lstrip("K"))
                tf, fr = blob["teacher_forced"], blob["free_running"]
                offsets = tf["agree_per_offset"]
                cost = fr["cycle_cost_ms"]
                row = {
                    "arm": arm,
                    "session": session,
                    "checkpoint": entry["checkpoint"],
                    "step": step,
                    "K": k,
                    "val_tv": tf["validation_tv"],
                    "tf_agree_specs": tf["agree_specs"],
                    "tf_mean_accepted_prefix": tf["mean_accepted_prefix"],
                    "tf_full_block_rate": tf["full_block_rate"],
                    "seed_row_agreement": tf["seed_row_agreement"],
                    "free_acceptance_rate": fr["acceptance_rate"],
                    "free_mean_accepted_specs": fr["mean_accepted_specs"],
                    "free_full_block_rate": fr["full_block_rate"],
                    "tpf": fr["tokens_per_forward"],
                    "forwards_per_token": fr["forwards_per_token"],
                    "proposal_ms": cost["proposal_ms"],
                    "verify_ms": cost["verify_ms"],
                    "commit_ms": cost["commit_ms"],
                    "overhead_ms": cost["overhead_ms"],
                    "cycle_ms": cost["total_ms"],
                    "unattributed_ms": fr.get("unattributed_ms_per_cycle", ""),
                    "prefill_ms": fr.get("prefill_ms", ""),
                    "tok_s_median": fr["median_tokens_per_second"],
                    "tok_s_std": fr["std_tokens_per_second"],
                    "tok_s_repeat_noise": fr["repeat_noise_tok_s"],
                    "speedup_vs_ar": fr["speedup_vs_ar"],
                    "predicted_tok_s": fr["predicted_tokens_per_second"],
                    "predicted_tok_s_v1": fr.get("predicted_tokens_per_second_v1", ""),
                    "cost_model_error_pct": fr["cost_model_error_pct"],
                    "cost_model_v1_error_pct": fr.get("cost_model_v1_error_pct", ""),
                    "cost_model_steady_error_pct": fr.get(
                        "cost_model_steady_state_error_pct", ""),
                    "ceiling_speedup": fr["ceiling"]["max_speedup_vs_ar"],
                    "ar_agreement": fr["ar_agreement"],
                    "ar_baseline_tok_s": ar,
                    "peak_memory_gb": entry.get("peak_memory_gb", ""),
                    "backbone_unchanged": entry["backbone_unchanged"],
                }
                for j in range(1, 9):
                    row[f"agree_plus_{j}"] = offsets.get(f"+{j}", "")
                hist = fr["acceptance_histogram"]
                total = sum(hist.values())
                for a in range(0, 9):
                    row[f"p_accept_{a}"] = (
                        round(hist.get(str(a), 0) / total, 5) if a < k else ""
                    )
                survival = fr["survival"]["accepted_specs"]
                for j in range(1, 9):
                    row[f"survive_ge_{j}"] = survival.get(str(j), "")
                for bucket in entry["draftability"]["buckets"]:
                    row[f"draft_q{bucket['bucket'] + 1}_accept"] = bucket["agreement"]
                row["draft_pearson_gap"] = entry["draftability"]["pearson_gap"]
                row["draft_pearson_entropy"] = entry["draftability"]["pearson_entropy"]
                rows.append(row)
    # Both U4-B arms continue from the same K=4 checkpoint. Copy that row into each
    # arm so the two learning curves start from their real common origin instead of
    # appearing to begin mid-training.
    # Only the reference measured in the *same* session: an arm-A row would import a
    # different session's wall-clock numbers into a B arm's curve, and cross-session
    # tok/s is exactly what criteria sec.2.1 says not to compare.
    def is_b_arm(arm: str) -> bool:
        # "B" is the session tag, "B1"/"B2" are the experimental arms.
        return arm.startswith("B") and arm != "B"

    b_sessions = {r["session"] for r in rows if is_b_arm(r["arm"])}
    for session in b_sessions:
        arms = sorted({r["arm"] for r in rows
                       if is_b_arm(r["arm"]) and r["session"] == session})
        shared = [r for r in rows if r["session"] == session
                  and r["arm"] not in arms and r["checkpoint"].startswith("step-")]
        for reference in shared:
            for arm in arms:
                copied = dict(reference)
                copied["arm"] = arm
                copied["checkpoint"] = f"{arm}-start({reference['checkpoint']})"
                rows.append(copied)

    # "x best K=4" is the U4-B result (brief sec.17), and it is only meaningful inside
    # one measurement session -- cross-session tok/s has a between-session sd of ~4%
    # (criteria sec.2.1), which is larger than any effect we are looking for.
    best_k4 = {}
    for row in rows:
        if row["K"] == 4:
            key = row["session"]
            if key not in best_k4 or row["tok_s_median"] > best_k4[key]["tok_s"]:
                best_k4[key] = {"tok_s": row["tok_s_median"], "tpf": row["tpf"],
                                "step": row["step"], "arm": row["arm"]}
    for row in rows:
        reference = best_k4.get(row["session"])
        if reference:
            row["best_k4_tok_s"] = reference["tok_s"]
            row["best_k4_step"] = reference["step"]
            row["speedup_vs_best_k4"] = round(row["tok_s_median"] / reference["tok_s"], 4)
            row["tpf_vs_best_k4"] = round(row["tpf"] / reference["tpf"], 4)
        else:
            row["best_k4_tok_s"] = row["best_k4_step"] = ""
            row["speedup_vs_best_k4"] = row["tpf_vs_best_k4"] = ""

    rows.sort(key=lambda r: (r["arm"], r["K"], r["step"]))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_tables(datasets: list[dict], rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "u4_checkpoints.csv", rows)

    # acceptance histograms, long form
    hist_rows = []
    for data in datasets:
        for entry in data["checkpoints"]:
            for key, blob in entry["per_block"].items():
                hist = blob["free_running"]["acceptance_histogram"]
                total = sum(hist.values())
                for accepted, n in sorted(hist.items(), key=lambda kv: int(kv[0])):
                    hist_rows.append({
                        "arm": checkpoint_arm(entry["checkpoint"], data.get("arm", "A")),
                        "checkpoint": entry["checkpoint"],
                        "step": checkpoint_step(entry["checkpoint"]),
                        "K": int(key.lstrip("K")),
                        "accepted_specs": int(accepted),
                        "committed_tokens": int(accepted) + 2,
                        "cycles": n,
                        "fraction": round(n / total, 5),
                    })
    write_csv(out / "u4_acceptance_histogram.csv", hist_rows)

    # survival curves, long form
    survival_rows = []
    for data in datasets:
        for entry in data["checkpoints"]:
            for key, blob in entry["per_block"].items():
                survival = blob["free_running"]["survival"]
                for j, p in survival["accepted_specs"].items():
                    survival_rows.append({
                        "arm": checkpoint_arm(entry["checkpoint"], data.get("arm", "A")),
                        "checkpoint": entry["checkpoint"],
                        "step": checkpoint_step(entry["checkpoint"]),
                        "K": int(key.lstrip("K")),
                        "j": int(j),
                        "p_accepted_ge_j": p,
                        "mean_accepted_specs": survival["mean_accepted_specs"],
                        "mean_committed_tokens": survival["mean_committed_tokens"],
                    })
    write_csv(out / "u4_survival.csv", survival_rows)

    # marginal slot economics
    marginal_rows = []
    for data in datasets:
        for entry in data["checkpoints"]:
            fit = entry.get("slot_cost_fit") or {}
            for key, blob in entry["per_block"].items():
                for slot in blob["free_running"].get("marginal_slot_value", []):
                    marginal_rows.append({
                        "arm": checkpoint_arm(entry["checkpoint"], data.get("arm", "A")),
                        "checkpoint": entry["checkpoint"],
                        "step": checkpoint_step(entry["checkpoint"]),
                        "K": int(key.lstrip("K")),
                        "ms_per_slot": fit.get("ms_per_slot", ""),
                        "slot_fit_r2": fit.get("r_squared", ""),
                        **slot,
                    })
    write_csv(out / "u4_marginal_slot.csv", marginal_rows)

    # draftability by future offset
    offset_rows = []
    for data in datasets:
        for entry in data["checkpoints"]:
            d = entry["draftability"]
            for blob in d.get("by_offset", []):
                offset_rows.append({
                    "arm": checkpoint_arm(entry["checkpoint"], data.get("arm", "A")),
                    "checkpoint": entry["checkpoint"],
                    "step": checkpoint_step(entry["checkpoint"]),
                    "block_size": d["block_size"],
                    **blob,
                })
    write_csv(out / "u4_draftability_by_offset.csv", offset_rows)

    # cost-model requirements / ceilings
    req_rows = []
    for data in datasets:
        for entry in data["checkpoints"]:
            for key, blob in entry["per_block"].items():
                fr = blob["free_running"]
                for req in fr["speedup_requirements"]:
                    req_rows.append({
                        "arm": checkpoint_arm(entry["checkpoint"], data.get("arm", "A")),
                        "checkpoint": entry["checkpoint"],
                        "K": int(key.lstrip("K")),
                        "ceiling_speedup": fr["ceiling"]["max_speedup_vs_ar"],
                        "actual_speedup": fr["speedup_vs_ar"],
                        "actual_per_slot_accept": fr["acceptance_rate"],
                        **req,
                    })
    write_csv(out / "u4_cost_model.csv", req_rows)

    json.dump(
        {
            "sessions": [
                {
                    "arm": d.get("arm", "A"),
                    "provenance": d["provenance"],
                    "harness": d["harness"],
                    "ar_baseline": d["ar_baseline"],
                    "offset_convention": d.get("offset_convention"),
                }
                for d in datasets
            ],
            "rows": rows,
        },
        open(out / "u4_summary.json", "w"), indent=2,
    )


def write_saturation(rows: list[dict], out: Path, thresholds: dict) -> list[dict]:
    """Apply the pre-registered saturation rule and write the derivation out.

    Written as its own table rather than folded into prose so that the verdict can be
    checked against the numbers it came from.
    """
    checks = []
    for arm in sorted({r["arm"] for r in rows}):
        for k in sorted({r["K"] for r in rows if r["arm"] == arm}):
            series = sorted(
                [r for r in rows if r["arm"] == arm and r["K"] == k],
                key=lambda r: r["step"],
            )
            for index in range(1, len(series)):
                previous, current = series[index - 1], series[index]
                span = current["step"] - previous["step"]
                d_accept = current["free_mean_accepted_specs"] - previous["free_mean_accepted_specs"]
                d_tpf = current["tpf"] - previous["tpf"]
                d_tv = current["val_tv"] - previous["val_tv"]
                scale = span / thresholds["interval"] if thresholds["interval"] else 1.0
                flat = (
                    d_accept < thresholds["accept"] * scale
                    and d_tpf < thresholds["tpf"] * scale
                    and d_tv > -thresholds["tv"] * scale
                )
                checks.append({
                    "arm": arm, "K": k,
                    "from_step": previous["step"], "to_step": current["step"],
                    "span": span,
                    "d_mean_accepted_specs": round(d_accept, 5),
                    "d_tpf": round(d_tpf, 5),
                    "d_val_tv": round(d_tv, 5),
                    "threshold_accept": round(thresholds["accept"] * scale, 5),
                    "threshold_tpf": round(thresholds["tpf"] * scale, 5),
                    "threshold_tv": round(thresholds["tv"] * scale, 5),
                    "interval_flat": flat,
                })
    # saturation needs two consecutive flat intervals
    for index, check in enumerate(checks):
        previous = checks[index - 1] if index else None
        check["saturated"] = bool(
            check["interval_flat"] and previous
            and previous["interval_flat"]
            and previous["arm"] == check["arm"] and previous["K"] == check["K"]
        )
    write_csv(out / "u4_saturation.csv", checks)
    return checks


# -------------------------------------------------------------------- plots


def make_plots(datasets: list[dict], histories: list[dict], rows: list[dict], out: Path):
    out.mkdir(parents=True, exist_ok=True)
    ks = sorted({r["K"] for r in rows})
    ar = rows[0]["ar_baseline_tok_s"] if rows else 0.0

    def series(k, field, arm=None):
        pts = [
            (r["step"], r[field]) for r in rows
            if r["K"] == k and r.get(field, "") != "" and (arm is None or r["arm"] == arm)
        ]
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]

    def line_plot(field, ylabel, title, filename, ks_wanted=None, hline=None, hlabel=None):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for k in (ks_wanted or ks):
            x, y = series(k, field)
            if x:
                ax.plot(x, y, label=f"K={k}", **STYLE.get(k, {}))
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

    # 1 K=4 val TV
    line_plot("val_tv", "held-out TV loss", "1. K=4 validation TV vs training step",
              "plot1_k4_val_tv.png", ks_wanted=[4])

    # 2 K=4 agreement by offset
    _offset_plot(rows, out, k=4, filename="plot2_k4_offsets.png",
                 title="2. K=4 teacher-forced agreement by future offset")

    # 3 K=4 mean accepted prefix
    line_plot("free_mean_accepted_specs", "mean accepted specs / cycle",
              "3. K=4 free-running accepted prefix vs training step",
              "plot3_k4_accepted_prefix.png", ks_wanted=[4])

    # 4 K=4 TPF
    line_plot("tpf", "tokens per forward", "4. K=4 TPF vs training step",
              "plot4_k4_tpf.png", ks_wanted=[4], hline=1.0, hlabel="AR (1.0)")

    # 5 K=4 tok/s
    line_plot("tok_s_median", "tokens / second", "5. K=4 throughput vs training step",
              "plot5_k4_tok_s.png", ks_wanted=[4], hline=ar, hlabel=f"AR ({ar:.1f} tok/s)")

    # 6 K=8 agreement +1..+8
    if 8 in ks:
        _offset_plot(rows, out, k=8, filename="plot6_k8_offsets.png",
                     title="6. K=8 teacher-forced agreement by future offset")
        line_plot("free_mean_accepted_specs", "mean accepted specs / cycle",
                  "7. K=8 free-running accepted prefix vs training step",
                  "plot7_k8_accepted_prefix.png", ks_wanted=[8])
        line_plot("tok_s_median", "tokens / second", "8. K=8 throughput vs training step",
                  "plot8_k8_tok_s.png", ks_wanted=[8], hline=ar,
                  hlabel=f"AR ({ar:.1f} tok/s)")

    # 9 horizon survival curves
    fig, ax = plt.subplots(figsize=(7, 4.4))
    latest = {}
    for row in rows:
        key = (row["arm"], row["K"])
        if key not in latest or row["step"] > latest[key]["step"]:
            latest[key] = row
    for (arm, k), row in sorted(latest.items()):
        js = [j for j in range(1, k) if row.get(f"survive_ge_{j}", "") != ""]
        ax.plot([0] + js, [1.0] + [row[f"survive_ge_{j}"] for j in js],
                label=f"{arm} K={k} @{row['step']} "
                      f"(E={row['free_mean_accepted_specs']:.2f})",
                **STYLE.get(k, {}))
    ax.set_xlabel("j (accepted speculative tokens)")
    ax.set_ylabel("P(accepted >= j)")
    ax.set_ylim(0, 1.02)
    ax.set_title("9. Horizon survival — area under each curve is E[accepted specs]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plot9_survival.png", dpi=140)
    plt.close(fig)

    # 10 draftability gap correlation vs future offset
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for data in datasets:
        entries = sorted(data["checkpoints"], key=lambda e: checkpoint_step(e["checkpoint"]))
        cmap = plt.get_cmap("viridis")
        for index, entry in enumerate(entries):
            by_offset = entry["draftability"].get("by_offset", [])
            if not by_offset:
                continue
            ax.plot([b["offset"] for b in by_offset], [b["pearson_gap"] for b in by_offset],
                    marker="o", color=cmap(index / max(len(entries) - 1, 1)),
                    label=f"{data.get('arm', 'A')} step {checkpoint_step(entry['checkpoint'])}")
    ax.axhline(0, ls="--", lw=1, color="#666")
    ax.set_xlabel("future offset")
    ax.set_ylabel("pearson(draftability gap, agreement)")
    ax.set_title("10. Does predecessor dependence matter more further out?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "plot10_draftability_by_offset.png", dpi=140)
    plt.close(fig)

    # 11 predicted vs measured, v1 and v2
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for k in ks:
        subset = [r for r in rows if r["K"] == k]
        style = STYLE.get(k, {})
        ax.scatter([r["predicted_tok_s"] for r in subset],
                   [r["tok_s_median"] for r in subset],
                   color=style.get("color"), marker=style.get("marker", "o"),
                   s=64, label=f"K={k} (v2)")
        v1 = [r for r in subset if r["predicted_tok_s_v1"] != ""]
        if v1:
            ax.scatter([r["predicted_tok_s_v1"] for r in v1],
                       [r["tok_s_median"] for r in v1],
                       facecolors="none", edgecolors=style.get("color"),
                       marker=style.get("marker", "o"), s=64, label=f"K={k} (v1)")
    values = [r["tok_s_median"] for r in rows] + [r["predicted_tok_s"] for r in rows]
    values += [r["predicted_tok_s_v1"] for r in rows if r["predicted_tok_s_v1"] != ""]
    lo, hi = min(values) * 0.95, max(values) * 1.05
    ax.plot([lo, hi], [lo, hi], ls="--", color="#666", lw=1, label="y = x")
    ax.set_xlabel("cost-model predicted tok/s")
    ax.set_ylabel("measured tok/s")
    ax.set_title("11. Cost model v1 (hollow) vs v2 (filled)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plot11_cost_model.png", dpi=140)
    plt.close(fig)

    # 12 ceiling comparison
    fig, ax = plt.subplots(figsize=(7, 4.2))
    width = 0.35
    best = {}
    for row in rows:
        if row["K"] not in best or row["speedup_vs_ar"] > best[row["K"]]["speedup_vs_ar"]:
            best[row["K"]] = row
    labels = sorted(best)
    positions = range(len(labels))
    ax.bar([p - width / 2 for p in positions],
           [best[k]["ceiling_speedup"] for k in labels], width,
           label="modelled ceiling", color="#cbd5e1", edgecolor="#475569")
    ax.bar([p + width / 2 for p in positions],
           [best[k]["speedup_vs_ar"] for k in labels], width,
           label="best measured", color="#2563eb")
    for index, k in enumerate(labels):
        ax.text(index - width / 2, best[k]["ceiling_speedup"] + 0.02,
                f"{best[k]['ceiling_speedup']:.2f}x", ha="center", fontsize=8)
        ax.text(index + width / 2, best[k]["speedup_vs_ar"] + 0.02,
                f"{best[k]['speedup_vs_ar']:.2f}x", ha="center", fontsize=8)
    ax.axhline(1.0, ls="--", lw=1, color="#666", label="AR")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"K={k}" for k in labels])
    ax.set_ylabel("speedup vs AR")
    ax.set_title("12. Structural ceiling vs measured, by block size")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "plot12_ceilings.png", dpi=140)
    plt.close(fig)

    # 0 training curve
    if histories:
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        for history in histories:
            steps = [h["step"] for h in history["history"]]
            tv = [h["tv"] for h in history["history"]]
            window = 25
            smoothed = [sum(tv[max(0, i - window):i + 1]) / len(tv[max(0, i - window):i + 1])
                        for i in range(len(tv))]
            ax.plot(steps, tv, lw=0.4, alpha=0.25, color="#94a3b8")
            ax.plot(steps, smoothed, lw=1.5, label=f"train TV ({window}-step mean)")
            if history.get("evals"):
                ax.plot([e["step"] for e in history["evals"]],
                        [e["tv"] for e in history["evals"]],
                        marker=".", lw=1, label="held-out TV (in-training)")
            start = (history.get("resume") or {}).get("start_step")
            if start:
                ax.axvline(start, ls=":", color="#dc2626", lw=1)
        ax.set_xlabel("training step")
        ax.set_ylabel("TV loss")
        ax.set_title("0. Training and held-out TV (dotted line = resume point)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "plot0_training_loss.png", dpi=140)
        plt.close(fig)


def _offset_plot(rows, out: Path, k: int, filename: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    entries = sorted([r for r in rows if r["K"] == k], key=lambda r: r["step"])
    if not entries:
        plt.close(fig)
        return
    cmap = plt.get_cmap("viridis")
    for index, row in enumerate(entries):
        offsets = [j for j in range(1, k + 1) if row.get(f"agree_plus_{j}", "") != ""]
        ax.plot(offsets, [row[f"agree_plus_{j}"] for j in offsets], marker="o",
                color=cmap(index / max(len(entries) - 1, 1)),
                label=f"{row['arm']} step {row['step']}")
    ax.set_xlabel("future offset (+1 is the adapter-off seed row)")
    ax.set_ylabel("teacher-forced agreement with the frozen model")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Act IV-U4 tables and plots")
    parser.add_argument("--eval", action="append", default=[],
                        help="u4-eval JSON; repeatable (e.g. the K=4 and K=8 arms)")
    parser.add_argument("--training", action="append", default=[],
                        help="training result.json; repeatable")
    parser.add_argument("--out", default="results/act4u4")
    parser.add_argument("--threshold-accept", type=float, default=0.06)
    parser.add_argument("--threshold-tpf", type=float, default=0.03)
    parser.add_argument("--threshold-tv", type=float, default=0.015)
    parser.add_argument("--threshold-interval", type=int, default=1600)
    args = parser.parse_args()

    datasets = load_evals([REPO / p for p in (args.eval or ["runs/u4a/eval.json"])])
    if not datasets:
        raise SystemExit("no evaluation JSON found")
    histories = [json.loads((REPO / p).read_text()) for p in args.training
                 if (REPO / p).is_file()]

    out = REPO / args.out
    rows = build_rows(datasets)
    write_tables(datasets, rows, out)
    checks = write_saturation(rows, out, {
        "accept": args.threshold_accept, "tpf": args.threshold_tpf,
        "tv": args.threshold_tv, "interval": args.threshold_interval,
    })
    make_plots(datasets, histories, rows, out)

    print(f"[u4] wrote tables and plots to {out}")
    for path in sorted(out.iterdir()):
        print(f"   {path.name}")
    saturated = [c for c in checks if c["saturated"]]
    if saturated:
        first = saturated[0]
        print(f"[u4] pre-registered saturation first met at arm {first['arm']} "
              f"K={first['K']} step {first['to_step']}")
    else:
        print("[u4] pre-registered saturation not met at any checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
