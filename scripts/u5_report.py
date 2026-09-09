#!/usr/bin/env python
"""Turn `u5-eval` output into the Act IV-U5 tables and plots.

Reads the paired-evaluation JSON (and optionally the per-arm training `result.json`
files, for compute accounting) and writes machine-readable CSVs plus the nine
pre-registered plots under `results/act4u5/`.

Nothing here recomputes a measurement. The paired bootstrap intervals come from the
evaluation harness, which is the only place that sees per-prompt and per-row data; this
file reshapes and plots them.

Offset convention, unchanged since U4: **future offset `+j` is supervised slot `j-1`**,
and `+1` is the adapter-off seed row, whose agreement is 1.0 by construction.
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

#: Colour per training horizon, so an arm keeps its identity across all nine plots.
ARM_STYLE = {
    "A_k4": dict(color="#dc2626", marker="s", label="A: K_train=4 (control)"),
    "B_k6": dict(color="#d97706", marker="D", label="B: K_train=6"),
    "C_k8": dict(color="#059669", marker="^", label="C: K_train=8"),
    "D_curr": dict(color="#7c3aed", marker="o", label="D: curriculum 4->6->8"),
    "untrained": dict(color="#64748b", marker="x", label="untrained (control 1)"),
}

#: Nominal training horizon per arm, for the "transfer gain by K_train" views. The
#: curriculum arm has no single K; it is plotted separately rather than assigned a
#: misleading average.
ARM_KTRAIN = {"A_k4": 4, "B_k6": 6, "C_k8": 8}


def style_for(arm: str) -> dict:
    base = ARM_STYLE.get(arm)
    if base:
        return dict(base)
    return dict(color="#0f172a", marker=".", label=arm)


def arm_step(name: str) -> int:
    groups = re.findall(r"\d+", name)
    return int(groups[-1]) if groups else 0


def arm_family(name: str) -> str:
    """`A_k4@13600` -> `A_k4`. Checkpoint labels carry the arm and the step."""
    return name.split("@")[0]


# ------------------------------------------------------------------- tables


def build_rows(datasets: list[dict], training: dict) -> list[dict]:
    rows = []
    for data in datasets:
        ar = data["ar_baseline"]["median_tokens_per_second"]
        session = data["provenance"]["timestamp"]
        primary = data.get("primary_block", 4)
        for entry in data["arms"]:
            family = arm_family(entry["arm"])
            meta = training.get(family, {})
            for key, blob in entry["per_block"].items():
                k_decode = int(key.lstrip("K"))
                tf, fr = blob["teacher_forced"], blob["free_running"]
                cost = fr["cycle_cost_ms"]
                steps = meta.get("steps", "")
                row = {
                    "session": session,
                    "checkpoint": entry["arm"],
                    "arm": family,
                    "step": arm_step(entry["arm"]),
                    "K_train": meta.get("k_train", ARM_KTRAIN.get(family, "")),
                    "curriculum": meta.get("curriculum", ""),
                    "training_steps": steps,
                    "training_tokens": meta.get("tokens", ""),
                    "supervised_positions": meta.get("supervised_positions", ""),
                    "training_seconds": meta.get("wall_seconds", ""),
                    "seconds_per_step": meta.get("seconds_per_step", ""),
                    "K_decode": k_decode,
                    "is_primary": k_decode == primary,
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
                    "tok_s_median": fr["median_tokens_per_second"],
                    "tok_s_mean": fr["mean_tokens_per_second"],
                    "speedup_vs_ar": fr["speedup_vs_ar"],
                    "predicted_tok_s": fr["predicted_tokens_per_second"],
                    "cost_model_error_pct": fr["cost_model_error_pct"],
                    "ar_agreement": fr["ar_agreement"],
                    "ar_baseline_tok_s": ar,
                    "peak_memory_gb": entry.get("peak_memory_gb", ""),
                    "backbone_unchanged": entry["backbone_unchanged"],
                }
                for j in range(1, 9):
                    row[f"agree_plus_{j}"] = tf["agree_per_offset"].get(f"+{j}", "")
                hist = fr["acceptance_histogram"]
                total = sum(hist.values())
                for m in range(0, 9):
                    row[f"p_accept_{m}"] = (
                        round(hist.get(str(m), 0) / total, 5) if m < k_decode else ""
                    )
                for j, p in fr["survival"]["accepted_specs"].items():
                    row[f"survive_ge_{j}"] = p
                for bucket in entry["draftability"]["buckets"]:
                    row[f"draft_q{bucket['bucket'] + 1}_accept"] = bucket["agreement"]
                row["draft_pearson_gap"] = entry["draftability"]["pearson_gap"]
                row["draft_pearson_entropy"] = entry["draftability"]["pearson_entropy"]
                rows.append(row)

    # delta vs the K=4 control, within a session and at the same decode block
    controls = {}
    for data in datasets:
        control = data.get("control_arm")
        for row in rows:
            if row["session"] == data["provenance"]["timestamp"] and row["arm"] == control:
                controls[(row["session"], row["K_decode"], row["step"])] = row
    for row in rows:
        reference = (
            controls.get((row["session"], row["K_decode"], row["step"]))
            or controls.get((row["session"], row["K_decode"], 0))
        )
        if reference and reference is not row:
            row["delta_tok_s_vs_control"] = round(
                row["tok_s_median"] - reference["tok_s_median"], 4)
            row["delta_tok_s_pct_vs_control"] = round(
                100 * (row["tok_s_median"] - reference["tok_s_median"])
                / reference["tok_s_median"], 3)
            row["delta_tpf_vs_control"] = round(row["tpf"] - reference["tpf"], 5)
            row["delta_prefix_vs_control"] = round(
                row["free_mean_accepted_specs"] - reference["free_mean_accepted_specs"], 5)
        else:
            for field in ("delta_tok_s_vs_control", "delta_tok_s_pct_vs_control",
                          "delta_tpf_vs_control", "delta_prefix_vs_control"):
                row[field] = "" if reference is None else 0.0

        # sec.20: transfer efficiency. The arms are compute-matched by construction --
        # every forward is `window-1` tokens wide whatever the block size -- so this
        # ranks the same as the raw delta. It is reported because "per unit compute"
        # is the practical question, and because the near-equality is itself the point.
        seconds = row.get("training_seconds") or 0
        if seconds and row["delta_prefix_vs_control"] not in ("", None):
            hours = seconds / 3600.0
            row["prefix_gain_per_training_hour"] = round(
                row["delta_prefix_vs_control"] / hours, 5)
            row["tok_s_gain_per_training_hour"] = round(
                (row["delta_tok_s_vs_control"] or 0.0) / hours, 4)
        else:
            row["prefix_gain_per_training_hour"] = ""
            row["tok_s_gain_per_training_hour"] = ""

    rows.sort(key=lambda r: (r["K_decode"], r["arm"], r["step"]))
    return rows


def load_training(paths: list[Path]) -> dict:
    """Compute accounting per arm, from each arm's training `result.json`."""
    meta = {}
    for path in paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        config, resume = data["config"], data.get("resume") or {}
        steps = (resume.get("end_step") or config["steps"]) - (resume.get("start_step") or 0)
        curriculum = config.get("block_curriculum") or []
        # Both forwards are `window - 1` wide regardless of block size, so token count
        # is the same for every arm; supervised positions are what differ.
        tokens = steps * config["batch_size"] * (config["window"] - 1) * 2
        if curriculum:
            per_stage = steps / len(curriculum)
            supervised = int(sum(per_stage * config["batch_size"] * (b - 1) for b in curriculum))
            k_train = "/".join(str(b) for b in curriculum)
        else:
            supervised = steps * config["batch_size"] * (config["block_size"] - 1)
            k_train = config["block_size"]
        meta[path.parent.name] = {
            "k_train": k_train,
            "curriculum": ",".join(str(b) for b in curriculum),
            "steps": steps,
            "tokens": tokens,
            "supervised_positions": supervised,
            "wall_seconds": data.get("wall_seconds"),
            "seconds_per_step": round(data["wall_seconds"] / steps, 4)
            if data.get("wall_seconds") else "",
        }
    return meta


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_tables(datasets: list[dict], rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "u5_arms.csv", rows)

    transfer_rows = []
    for data in datasets:
        for blob in data.get("transfer", []):
            flat = {
                "session": data["provenance"]["timestamp"],
                "arm": blob["arm"],
                "control": blob["control"],
                "K_decode": blob["K_decode"],
                "delta_acceptance": blob["delta_acceptance"],
                "delta_tpf": blob["delta_tpf"],
                "delta_tok_s_pct": blob["delta_tok_s_pct"],
            }
            for metric in ("tok_s", "tpf", "accepted_specs",
                           "tf_accepted_prefix", "tf_agreement"):
                stats = blob.get(metric) or {}
                flat[f"{metric}_mean_diff"] = stats.get("mean_diff", "")
                flat[f"{metric}_ci_low"] = stats.get("ci_low", "")
                flat[f"{metric}_ci_high"] = stats.get("ci_high", "")
                flat[f"{metric}_significant"] = stats.get("significant", "")
                flat[f"{metric}_wins"] = stats.get("wins", "")
                flat[f"{metric}_n"] = stats.get("n", "")
            for offset, delta in (blob.get("delta_agreement_by_offset") or {}).items():
                flat[f"delta_agree_{offset}"] = delta
            transfer_rows.append(flat)
    write_csv(out / "u5_transfer.csv", transfer_rows)

    json.dump(
        {
            "sessions": [
                {"provenance": d["provenance"], "harness": d["harness"],
                 "ar_baseline": d["ar_baseline"], "primary_block": d.get("primary_block"),
                 "control_arm": d.get("control_arm"),
                 "offset_convention": d.get("offset_convention")}
                for d in datasets
            ],
            "rows": rows,
            "transfer": transfer_rows,
        },
        open(out / "u5_summary.json", "w"), indent=2,
    )


# -------------------------------------------------------------------- plots


def make_plots(datasets: list[dict], rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    primary = datasets[0].get("primary_block", 4)
    prime = [r for r in rows if r["K_decode"] == primary and r["arm"] != "untrained"]
    if not prime:
        return
    ar = prime[0]["ar_baseline_tok_s"]
    arms = sorted({r["arm"] for r in prime})

    def curve(arm, field):
        points = sorted(
            (r["step"], r[field]) for r in prime
            if r["arm"] == arm and r.get(field, "") != ""
        )
        return [p[0] for p in points], [p[1] for p in points]

    def step_plot(field, ylabel, title, filename, hline=None, hlabel=None):
        figure, axis = plt.subplots(figsize=(7.2, 4.3))
        for arm in arms:
            x, y = curve(arm, field)
            if x:
                axis.plot(x, y, **style_for(arm))
        if hline is not None:
            axis.axhline(hline, ls="--", lw=1, color="#666", label=hlabel)
        axis.set_xlabel("training step (all arms resume from the same checkpoint)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(out / filename, dpi=140)
        plt.close(figure)

    step_plot("free_mean_accepted_specs", f"K={primary} mean accepted specs / cycle",
              f"1. K={primary} acceptance vs training step, by training horizon",
              "plot1_acceptance.png")
    step_plot("tpf", f"K={primary} tokens per forward",
              f"2. K={primary} TPF vs training step", "plot2_tpf.png",
              hline=1.0, hlabel="AR (1.0)")
    step_plot("tok_s_median", f"K={primary} tokens / second",
              f"3. K={primary} throughput vs training step", "plot3_tok_s.png",
              hline=ar, hlabel=f"AR ({ar:.1f} tok/s)")

    # 4 future offset vs agreement, final checkpoint of each arm
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    finals = {}
    for row in prime:
        if row["arm"] not in finals or row["step"] > finals[row["arm"]]["step"]:
            finals[row["arm"]] = row
    for arm, row in sorted(finals.items()):
        offsets = [j for j in range(1, primary + 1) if row.get(f"agree_plus_{j}", "") != ""]
        axis.plot(offsets, [row[f"agree_plus_{j}"] for j in offsets], **style_for(arm))
    axis.set_xlabel("future offset (+1 is the adapter-off seed row)")
    axis.set_ylabel("teacher-forced agreement")
    axis.set_title(f"4. K={primary} agreement by future offset, final checkpoint")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "plot4_offsets.png", dpi=140)
    plt.close(figure)

    # 5 transfer gain by K_train, with paired intervals
    transfer = [t for d in datasets for t in d.get("transfer", [])
                if t["K_decode"] == primary]
    if transfer:
        figure, axis = plt.subplots(figsize=(7.2, 4.3))
        labels, centres, lows, highs = [], [], [], []
        for blob in transfer:
            stats = blob["tok_s"]
            labels.append(blob["arm"])
            centres.append(stats["mean_diff"])
            lows.append(stats["mean_diff"] - (stats["ci_low"] or stats["mean_diff"]))
            highs.append((stats["ci_high"] or stats["mean_diff"]) - stats["mean_diff"])
        positions = range(len(labels))
        axis.bar(positions, centres, yerr=[lows, highs], capsize=5,
                 color=[style_for(arm_family(a))["color"] for a in labels])
        axis.axhline(0, color="#111", lw=1)
        axis.set_xticks(list(positions))
        axis.set_xticklabels(labels)
        axis.set_ylabel(f"paired mean d(tok/s) at K={primary}")
        axis.set_title("5. Transfer gain vs the K=4-trained control (95% paired bootstrap)")
        axis.grid(alpha=0.3, axis="y")
        figure.tight_layout()
        figure.savefig(out / "plot5_transfer_gain.png", dpi=140)
        plt.close(figure)

    # 6 accepted-prefix histogram by arm, final checkpoints
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    width = 0.8 / max(len(finals), 1)
    for index, (arm, row) in enumerate(sorted(finals.items())):
        buckets = [m for m in range(primary) if row.get(f"p_accept_{m}", "") != ""]
        offsets = [m + (index - len(finals) / 2) * width + width / 2 for m in buckets]
        axis.bar(offsets, [row[f"p_accept_{m}"] for m in buckets], width,
                 color=style_for(arm)["color"], label=style_for(arm)["label"])
    axis.set_xlabel("accepted speculative tokens per cycle")
    axis.set_ylabel("fraction of cycles")
    axis.set_title(f"6. K={primary} accepted-prefix distribution by arm")
    axis.set_xticks(list(range(primary)))
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "plot6_prefix_histogram.png", dpi=140)
    plt.close(figure)

    # 7 draftability quintile vs acceptance by arm
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    for arm, row in sorted(finals.items()):
        quintiles = [q for q in range(1, 6) if row.get(f"draft_q{q}_accept", "") != ""]
        if quintiles:
            axis.plot(quintiles, [row[f"draft_q{q}_accept"] for q in quintiles],
                      **style_for(arm))
    axis.set_xlabel("draftability-gap quintile (1 = most draftable)")
    axis.set_ylabel(f"K={primary} teacher-forced agreement")
    axis.set_title("7. Does long-horizon training help the hard-to-draft regions?")
    axis.set_xticks([1, 2, 3, 4, 5])
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "plot7_draftability.png", dpi=140)
    plt.close(figure)

    # 8 training compute vs K=4 speed gain
    figure, axis = plt.subplots(figsize=(6.4, 4.3))
    for arm, row in sorted(finals.items()):
        seconds = row.get("training_seconds")
        delta = row.get("delta_tok_s_vs_control")
        if seconds and delta != "":
            axis.scatter(seconds / 3600.0, delta, s=90, **{
                k: v for k, v in style_for(arm).items() if k != "marker"},
                marker=style_for(arm)["marker"])
    axis.axhline(0, color="#111", lw=1)
    axis.set_xlabel("training wall clock (hours)")
    axis.set_ylabel(f"d(tok/s) at K={primary} vs control")
    axis.set_title("8. Speed gain against the compute that bought it")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "plot8_compute_vs_gain.png", dpi=140)
    plt.close(figure)

    # 9 K_train vs K=4 inference throughput
    figure, axis = plt.subplots(figsize=(6.4, 4.3))
    fixed = [(ARM_KTRAIN[a], r) for a, r in finals.items() if a in ARM_KTRAIN]
    if fixed:
        fixed.sort()
        axis.plot([k for k, _ in fixed], [r["tok_s_median"] for _, r in fixed],
                  marker="o", color="#0f172a", label="fixed K_train")
        for k, row in fixed:
            axis.annotate(f"{row['tok_s_median']:.1f}", (k, row["tok_s_median"]),
                          textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    for arm, row in finals.items():
        if arm not in ARM_KTRAIN:
            axis.axhline(row["tok_s_median"], ls=":", lw=1.4,
                         color=style_for(arm)["color"], label=style_for(arm)["label"])
    axis.axhline(ar, ls="--", lw=1, color="#666", label=f"AR ({ar:.1f})")
    axis.set_xlabel("training horizon K_train")
    axis.set_ylabel(f"K_decode={primary} tokens / second")
    axis.set_title(f"9. Training horizon vs K={primary} inference throughput")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "plot9_ktrain_vs_throughput.png", dpi=140)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Act IV-U5 tables and plots")
    parser.add_argument("--eval", action="append", default=[],
                        help="u5-eval JSON; repeatable")
    parser.add_argument("--training", action="append", default=[],
                        help="per-arm training result.json; repeatable")
    parser.add_argument("--out", default="results/act4u5")
    args = parser.parse_args()

    paths = [REPO / p for p in (args.eval or ["runs/u5/eval.json"])]
    datasets = [json.loads(p.read_text()) for p in paths if p.is_file()]
    if not datasets:
        raise SystemExit(f"no evaluation JSON found at {[str(p) for p in paths]}")

    training = load_training([REPO / p for p in args.training])
    rows = build_rows(datasets, training)
    out = REPO / args.out
    write_tables(datasets, rows, out)
    make_plots(datasets, rows, out)

    print(f"[u5] wrote tables and plots to {out}")
    for path in sorted(out.iterdir()):
        print(f"   {path.name}")

    primary = datasets[0].get("primary_block", 4)
    wins = [t for d in datasets for t in d.get("transfer", [])
            if t["K_decode"] == primary and t["tok_s"].get("significant")
            and t["tok_s"]["mean_diff"] > 0]
    if wins:
        print(f"[u5] arms beating the control on paired K={primary} tok/s: "
              + ", ".join(f"{t['arm']} ({t['tok_s']['mean_diff']:+.2f} tok/s)" for t in wins))
    else:
        print(f"[u5] no arm beat the control on paired K={primary} tok/s beyond noise")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
