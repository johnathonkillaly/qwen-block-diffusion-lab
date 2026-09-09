"""Stage 1 analysis: does denoiser uncertainty predict target acceptance?

Runs exactly the comparisons pre-registered in `RPRM_DIFFUSION_PREREG.md` and prints
the verdict against the frozen thresholds. It reads only the CSV emitted by
`scripts/rprm_stage1_extract.py`; it never touches the model, so re-running it is
cheap and cannot change the measurement.

Estimators are implemented here rather than pulled from scikit-learn (not installed
in `.venv-unsloth`). Logistic regression is plain unregularised maximum likelihood
via L-BFGS-B on the exact NLL -- no penalty, no tuning, so no tuning partition is
needed and TEST stays untouched until the end.

Two things that are easy to get wrong and are done deliberately:

  * **Splits are at the window level.** All 7 speculative positions in a window share
    a context; a token-level split would leak that context across FIT and TEST.
  * **Bootstrap resamples windows, not positions.** Positions within a window are
    correlated, so a token-level bootstrap would report intervals that are too tight.

Usage:
    .venv-unsloth/bin/python scripts/rprm_stage1_analyze.py --tag primary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent.parent

SPLIT_SEED = 20260908
SHUFFLE_SEED = 20260908
BOOTSTRAP_SEED = 20260908
N_SHUFFLE = 20
N_BOOTSTRAP = 2000
SPLIT_SHARES = (0.50, 0.20, 0.30)  # FIT / VAL / TEST

MIN_CELL_N = 200
MIN_BUCKET_N = 40


# ------------------------------------------------------------------ estimators


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUROC with proper tie handling (average ranks)."""
    y = np.asarray(y, dtype=float)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_scores = score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision (step-wise, the standard AP estimator)."""
    order = np.argsort(-score, kind="mergesort")
    y = np.asarray(y, dtype=float)[order]
    if y.sum() == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- well behaved at the small acceptance rates here."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


class Logit:
    """Unregularised logistic regression, standardised features, L-BFGS-B."""

    def __init__(self) -> None:
        self.beta: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Logit":
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        self.sigma[self.sigma < 1e-12] = 1.0
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sigma])

        def nll_and_grad(beta):
            eta = np.clip(Z @ beta, -35, 35)
            p = 1 / (1 + np.exp(-eta))
            nll = -np.mean(y * eta - np.logaddexp(0, eta))
            grad = -(Z.T @ (y - p)) / len(y)
            return nll, grad

        result = minimize(
            nll_and_grad, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10},
        )
        self.beta = result.x
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sigma])
        return 1 / (1 + np.exp(-np.clip(Z @ self.beta, -35, 35)))


# ------------------------------------------------------------------- features


def design(df: pd.DataFrame, spec: list[str], offsets: list[int], schedule: bool):
    """Build a design matrix from a list of feature-group names."""
    cols, names = [], []
    for part in spec:
        if part == "progress":
            for off in offsets[1:]:  # first offset is the reference level
                cols.append((df["offset"].to_numpy() == off).astype(float))
                names.append(f"offset_{off}")
            if schedule:
                for extra in ("timestep", "corruption_level", "n_corrupted_predecessors"):
                    cols.append(df[extra].to_numpy(dtype=float))
                    names.append(extra)
        else:
            cols.append(df[part].to_numpy(dtype=float))
            names.append(part)
    return np.column_stack(cols), names


MODELS = [
    ("1_progress_only", ["progress"]),
    ("2_uncertainty_only", ["normalized_entropy"]),
    ("3_existing_only", ["existing_draftability_score"]),
    ("4_progress_uncertainty", ["progress", "normalized_entropy"]),
    ("5_progress_existing", ["progress", "existing_draftability_score"]),
    ("6_progress_existing_uncertainty",
     ["progress", "existing_draftability_score", "normalized_entropy"]),
    ("8_progress_teacher_entropy", ["progress", "teacher_entropy"]),
]


# -------------------------------------------------------------------- analysis


def split_windows(df: pd.DataFrame) -> dict[str, np.ndarray]:
    windows = np.array(sorted(df["example_id"].unique()))
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(windows))
    n_fit = int(round(SPLIT_SHARES[0] * len(windows)))
    n_val = int(round(SPLIT_SHARES[1] * len(windows)))
    return {
        "FIT": windows[order[:n_fit]],
        "VAL": windows[order[n_fit : n_fit + n_val]],
        "TEST": windows[order[n_fit + n_val :]],
    }


def quintile_table(fit: pd.DataFrame, test: pd.DataFrame, column: str) -> list[dict]:
    """Acceptance by quintile of `column`; edges from FIT, applied to TEST."""
    edges = np.quantile(fit[column].to_numpy(), [0.2, 0.4, 0.6, 0.8])
    idx = np.digitize(test[column].to_numpy(), edges)
    rows = []
    for q in range(5):
        sel = test[idx == q]
        n = len(sel)
        k = int(sel["accepted"].sum())
        lo, hi = wilson(k, n)
        rows.append({
            "quintile": q, "n": n,
            "mean_value": round(float(sel[column].mean()), 5) if n else None,
            "acceptance": round(k / n, 5) if n else None,
            "ci_low": round(lo, 5), "ci_high": round(hi, 5),
        })
    return rows


def within_cell_table(fit: pd.DataFrame, test: pd.DataFrame, offsets: list[int]) -> list[dict]:
    """Per-offset: AUROC of low-entropy-as-acceptance, plus tercile acceptance."""
    rows = []
    for off in offsets:
        f = fit[fit["offset"] == off]
        t = test[test["offset"] == off]
        if len(t) == 0 or len(f) == 0:
            continue
        y = t["accepted"].to_numpy(dtype=float)
        h = t["normalized_entropy"].to_numpy()
        edges = np.quantile(f["normalized_entropy"].to_numpy(), [1 / 3, 2 / 3])
        idx = np.digitize(h, edges)
        low, high = t[idx == 0], t[idx == 2]
        rows.append({
            "offset": off, "n": len(t),
            "acceptance": round(float(y.mean()), 5),
            "auroc_neg_entropy": round(auroc(y, -h), 5) if 0 < y.sum() < len(y) else None,
            "auroc_neg_gap": round(
                auroc(y, -t["existing_draftability_score"].to_numpy()), 5)
            if 0 < y.sum() < len(y) else None,
            "n_low": len(low), "acc_low": round(float(low["accepted"].mean()), 5) if len(low) else None,
            "n_high": len(high), "acc_high": round(float(high["accepted"].mean()), 5) if len(high) else None,
            "adequate": len(t) >= MIN_CELL_N and min(len(low), len(high)) >= MIN_BUCKET_N,
        })
    return rows


def cluster_bootstrap_delta(test: pd.DataFrame, pa: np.ndarray, pb: np.ndarray) -> dict:
    """95% CI on AUROC(b) − AUROC(a), resampling *windows*."""
    y = test["accepted"].to_numpy(dtype=float)
    ids = test["example_id"].to_numpy()
    uniq, inverse = np.unique(ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = []
    for _ in range(N_BOOTSTRAP):
        pick = rng.integers(0, len(groups), len(groups))
        take = np.concatenate([groups[g] for g in pick])
        ys = y[take]
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        deltas.append(auroc(ys, pb[take]) - auroc(ys, pa[take]))
    deltas = np.array(deltas)
    return {
        "delta_mean": round(float(deltas.mean()), 5),
        "ci_low": round(float(np.percentile(deltas, 2.5)), 5),
        "ci_high": round(float(np.percentile(deltas, 97.5)), 5),
        "n_resamples": len(deltas),
    }


def cluster_bootstrap_logloss(test, pa, pb) -> dict:
    y = test["accepted"].to_numpy(dtype=float)
    uniq, inverse = np.unique(test["example_id"].to_numpy(), return_inverse=True)
    groups = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    deltas = []
    for _ in range(N_BOOTSTRAP):
        pick = rng.integers(0, len(groups), len(groups))
        take = np.concatenate([groups[g] for g in pick])
        deltas.append(log_loss(y[take], pa[take]) - log_loss(y[take], pb[take]))
    deltas = np.array(deltas)
    return {
        "reduction_mean": round(float(deltas.mean()), 6),
        "ci_low": round(float(np.percentile(deltas, 2.5)), 6),
        "ci_high": round(float(np.percentile(deltas, 97.5)), 6),
    }


def run_stratum(df: pd.DataFrame, stratum: str, splits: dict) -> dict:
    schedule = stratum == "SCHEDULE"
    offsets = sorted(df["offset"].unique())
    fit = df[df["example_id"].isin(splits["FIT"])].reset_index(drop=True)
    test = df[df["example_id"].isin(splits["TEST"])].reset_index(drop=True)
    y_fit = fit["accepted"].to_numpy(dtype=float)
    y_test = test["accepted"].to_numpy(dtype=float)

    out: dict = {
        "stratum": stratum,
        "n_fit": len(fit), "n_test": len(test),
        "windows_fit": int(fit["example_id"].nunique()),
        "windows_test": int(test["example_id"].nunique()),
        "acceptance_fit": round(float(y_fit.mean()), 5),
        "acceptance_test": round(float(y_test.mean()), 5),
    }

    # ---- the seven pre-registered models
    preds, models = {}, {}
    for name, spec in MODELS:
        Xf, _ = design(fit, spec, offsets, schedule)
        Xt, _ = design(test, spec, offsets, schedule)
        model = Logit().fit(Xf, y_fit)
        p = model.predict(Xt)
        preds[name], models[name] = p, model
    out["models"] = {
        name: {
            "auroc": round(auroc(y_test, p), 5),
            "auprc": round(auprc(y_test, p), 5),
            "log_loss": round(log_loss(y_test, p), 5),
        }
        for name, p in preds.items()
    }

    # ---- 7: shuffled control, entropy permuted within cell
    cell_cols = ["offset"] if not schedule else ["offset", "t_bucket"]
    shuffled_auroc, shuffled_ll = [], []
    for rep in range(N_SHUFFLE):
        rng = np.random.default_rng(SHUFFLE_SEED + rep)
        f2, t2 = fit.copy(), test.copy()
        for frame in (f2, t2):
            values = frame["normalized_entropy"].to_numpy().copy()
            for _, index in frame.groupby(cell_cols).indices.items():
                values[index] = rng.permutation(values[index])
            frame["normalized_entropy"] = values
        Xf, _ = design(f2, ["progress", "normalized_entropy"], offsets, schedule)
        Xt, _ = design(t2, ["progress", "normalized_entropy"], offsets, schedule)
        p = Logit().fit(Xf, y_fit).predict(Xt)
        shuffled_auroc.append(auroc(y_test, p))
        shuffled_ll.append(log_loss(y_test, p))
    shuffled_auroc = np.array(shuffled_auroc)
    out["models"]["7_shuffled_control"] = {
        "auroc": round(float(shuffled_auroc.mean()), 5),
        "auroc_p2_5": round(float(np.percentile(shuffled_auroc, 2.5)), 5),
        "auroc_p97_5": round(float(np.percentile(shuffled_auroc, 97.5)), 5),
        "log_loss": round(float(np.mean(shuffled_ll)), 5),
        "n_permutations": N_SHUFFLE,
    }

    # ---- descriptive
    out["quintiles_entropy"] = quintile_table(fit, test, "normalized_entropy")
    out["quintiles_gap"] = quintile_table(fit, test, "existing_draftability_score")
    out["quintiles_teacher_entropy"] = quintile_table(fit, test, "teacher_entropy")
    out["within_offset"] = within_cell_table(fit, test, offsets)
    out["spearman_test"] = {
        col: round(spearman(test[col].to_numpy(), y_test), 5)
        for col in ("normalized_entropy", "top1_probability", "top1_top2_margin",
                    "existing_draftability_score", "teacher_entropy", "corrupt_entropy")
    }

    # ---- diagnostics: alternative uncertainty measures (NOT routes to a pass)
    out["diagnostic_alternatives"] = {}
    for alt in ("top1_probability", "top1_top2_margin", "effective_support"):
        Xf, _ = design(fit, ["progress", alt], offsets, schedule)
        Xt, _ = design(test, ["progress", alt], offsets, schedule)
        p = Logit().fit(Xf, y_fit).predict(Xt)
        out["diagnostic_alternatives"][alt] = {
            "auroc": round(auroc(y_test, p), 5),
            "delta_vs_progress": round(
                auroc(y_test, p) - out["models"]["1_progress_only"]["auroc"], 5),
        }

    # ---- pre-registered comparisons
    out["comparisons"] = {
        "4_vs_1_auroc_delta": round(
            out["models"]["4_progress_uncertainty"]["auroc"]
            - out["models"]["1_progress_only"]["auroc"], 5),
        "6_vs_5_auroc_delta": round(
            out["models"]["6_progress_existing_uncertainty"]["auroc"]
            - out["models"]["5_progress_existing"]["auroc"], 5),
        "4_vs_1_bootstrap": cluster_bootstrap_delta(
            test, preds["1_progress_only"], preds["4_progress_uncertainty"]),
        "6_vs_5_bootstrap": cluster_bootstrap_delta(
            test, preds["5_progress_existing"], preds["6_progress_existing_uncertainty"]),
        "6_vs_5_logloss": cluster_bootstrap_logloss(
            test, preds["5_progress_existing"], preds["6_progress_existing_uncertainty"]),
    }
    out["_preds"] = {k: v.tolist() for k, v in preds.items()}
    return out


def verdict(res: dict) -> dict:
    """Apply the frozen thresholds from RPRM_DIFFUSION_PREREG.md §6."""
    q = [r for r in res["quintiles_entropy"] if r["acceptance"] is not None]
    acc = [r["acceptance"] for r in q]
    # criterion 1: acceptance should FALL as entropy rises; count bad inversions
    inversions = [acc[i + 1] - acc[i] for i in range(len(acc) - 1)]
    bad = [d for d in inversions if d > 0]
    c1 = len(bad) <= 1 and (max(bad) <= 0.02 if bad else True)

    adequate = [r for r in res["within_offset"] if r["adequate"]]
    signed = [r for r in adequate if r["auroc_neg_entropy"] and r["auroc_neg_entropy"] > 0.5]
    c2 = len(signed) >= 4

    c3 = res["comparisons"]["4_vs_1_auroc_delta"] >= 0.05
    c4 = (res["models"]["4_progress_uncertainty"]["auroc"]
          > res["models"]["7_shuffled_control"]["auroc_p97_5"])
    ll = res["comparisons"]["6_vs_5_logloss"]
    c5 = (res["comparisons"]["6_vs_5_auroc_delta"] >= 0.02
          or (ll["reduction_mean"] >= 0.01 and ll["ci_low"] > 0))
    c6 = res["comparisons"]["4_vs_1_bootstrap"]["ci_low"] > 0

    criteria = {
        "c1_monotone": c1, "c2_within_cell": c2, "c3_auroc_gain_over_progress": c3,
        "c4_beats_shuffled": c4, "c5_adds_to_existing": c5, "c6_bootstrap_nontrivial": c6,
    }
    if all(criteria.values()):
        label = "STRONG PASS"
    elif c3 and c4:
        label = "WEAK PASS"
    else:
        label = "FAIL"
    return {
        "criteria": criteria,
        "n_adequate_cells": len(adequate),
        "n_cells_signed": len(signed),
        "quintile_inversions": [round(d, 5) for d in inversions],
        "verdict": label,
    }


def make_plots(frames: dict, results: dict, out_dir: Path, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full = results["FULL"]

    # 1: acceptance vs entropy quintile, globally (+ gap and teacher entropy)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label, style in (
        ("quintiles_entropy", "denoiser normalized entropy", "-o"),
        ("quintiles_gap", "existing draftability gap", "-s"),
        ("quintiles_teacher_entropy", "teacher entropy (already tested)", "--^"),
    ):
        rows = full[key]
        xs = [r["quintile"] for r in rows]
        ys = [r["acceptance"] for r in rows]
        err = [[y - r["ci_low"] for y, r in zip(ys, rows)],
               [r["ci_high"] - y for y, r in zip(ys, rows)]]
        ax.errorbar(xs, ys, yerr=err, fmt=style, capsize=3, label=label)
    ax.set_xlabel("quintile (0 = lowest value)")
    ax.set_ylabel("held-out acceptance")
    ax.set_title(f"Acceptance by predictor quintile — FULL stratum, TEST ({tag})")
    ax.set_xticks(range(5))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"plot1_quintiles_{tag}.png", dpi=150)
    plt.close(fig)

    # 2: acceptance vs entropy within offset
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = [r for r in full["within_offset"] if r["acc_low"] is not None]
    xs = [r["offset"] for r in rows]
    ax.plot(xs, [r["acc_low"] for r in rows], "-o", label="low-entropy tercile")
    ax.plot(xs, [r["acc_high"] for r in rows], "-s", label="high-entropy tercile")
    ax.plot(xs, [r["acceptance"] for r in rows], "--", color="grey", label="all positions")
    for r in rows:
        ax.annotate(f"n={r['n']}", (r["offset"], r["acceptance"]), fontsize=6,
                    textcoords="offset points", xytext=(0, -12), ha="center")
    ax.set_xlabel("future offset (+j)")
    ax.set_ylabel("held-out acceptance")
    ax.set_title(f"Within-offset acceptance by denoiser entropy — FULL, TEST ({tag})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"plot2_within_offset_{tag}.png", dpi=150)
    plt.close(fig)

    # 3: real vs within-cell shuffled control
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    m = full["models"]
    labels = ["progress\nonly", "progress +\nentropy", "shuffled\ncontrol"]
    vals = [m["1_progress_only"]["auroc"], m["4_progress_uncertainty"]["auroc"],
            m["7_shuffled_control"]["auroc"]]
    bars = ax.bar(labels, vals, color=["#888", "#2b6cb0", "#c05621"])
    ax.errorbar(
        2, m["7_shuffled_control"]["auroc"],
        yerr=[[m["7_shuffled_control"]["auroc"] - m["7_shuffled_control"]["auroc_p2_5"]],
              [m["7_shuffled_control"]["auroc_p97_5"] - m["7_shuffled_control"]["auroc"]]],
        fmt="none", ecolor="black", capsize=5)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.4f}", (bar.get_x() + bar.get_width() / 2, v), ha="center",
                    va="bottom", fontsize=9)
    ax.set_ylabel("held-out AUROC")
    ax.set_ylim(min(vals) - 0.02, max(vals) + 0.02)
    ax.set_title(f"Real entropy vs within-offset shuffled control ({tag})")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / f"plot3_shuffled_control_{tag}.png", dpi=150)
    plt.close(fig)

    # 4: ROC and PR for the primary models
    test = frames["FULL"]["test"]
    y = test["accepted"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in ("1_progress_only", "4_progress_uncertainty", "5_progress_existing",
                 "6_progress_existing_uncertainty"):
        p = np.array(full["_preds"][name])
        order = np.argsort(-p)
        ys = y[order]
        tpr = np.cumsum(ys) / max(ys.sum(), 1)
        fpr = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
        axes[0].plot(np.r_[0, fpr], np.r_[0, tpr], label=f"{name} ({full['models'][name]['auroc']:.4f})")
        prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
        axes[1].plot(tpr, prec, label=f"{name} ({full['models'][name]['auprc']:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC (AUROC)"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].axhline(y.mean(), color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall (AP)"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.suptitle(f"Primary model comparison — FULL stratum, TEST ({tag})")
    fig.tight_layout()
    fig.savefig(out_dir / f"plot4_roc_pr_{tag}.png", dpi=150)
    plt.close(fig)

    # 5: calibration
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ("1_progress_only", "5_progress_existing", "6_progress_existing_uncertainty"):
        p = np.array(full["_preds"][name])
        edges = np.quantile(p, np.linspace(0, 1, 11))
        edges[-1] += 1e-9
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() > 0]
        ys = [y[idx == b].mean() for b in range(10) if (idx == b).sum() > 0]
        ax.plot(xs, ys, "-o", ms=4, label=name)
    ax.plot([0, max(y.mean() * 3, 0.5)], [0, max(y.mean() * 3, 0.5)], "k--", lw=0.8,
            label="perfect calibration")
    ax.set_xlabel("predicted acceptance"); ax.set_ylabel("observed acceptance")
    ax.set_title(f"Calibration — FULL stratum, TEST ({tag})")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"plot5_calibration_{tag}.png", dpi=150)
    plt.close(fig)
    print(f"[rprm] wrote 5 plots to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="primary")
    parser.add_argument("--dir", default="results/rprm_stage1")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    out_dir = REPO / args.dir
    df = pd.read_csv(out_dir / f"positions_{args.tag}.csv")
    df["t_bucket"] = pd.qcut(df["timestep"], 5, labels=False, duplicates="drop")
    splits = split_windows(df)
    print(f"[rprm] {len(df)} positions, {df['example_id'].nunique()} windows -> "
          f"FIT {len(splits['FIT'])} / VAL {len(splits['VAL'])} / TEST {len(splits['TEST'])}")

    results, frames = {}, {}
    for stratum in ("FULL", "SCHEDULE"):
        sub = df[df["stratum"] == stratum].reset_index(drop=True)
        if sub.empty:
            continue
        res = run_stratum(sub, stratum, splits)
        res["verdict"] = verdict(res)
        results[stratum] = res
        frames[stratum] = {
            "fit": sub[sub["example_id"].isin(splits["FIT"])].reset_index(drop=True),
            "test": sub[sub["example_id"].isin(splits["TEST"])].reset_index(drop=True),
        }
        print(f"\n===== {stratum} =====")
        print(f"  n_test={res['n_test']} acceptance={res['acceptance_test']}")
        for name, m in res["models"].items():
            print(f"  {name:34s} AUROC {m['auroc']:.4f}  logloss {m['log_loss']:.4f}")
        print(f"  4 vs 1 AUROC delta: {res['comparisons']['4_vs_1_auroc_delta']:+.4f} "
              f"CI {res['comparisons']['4_vs_1_bootstrap']['ci_low']:+.4f}..."
              f"{res['comparisons']['4_vs_1_bootstrap']['ci_high']:+.4f}")
        print(f"  6 vs 5 AUROC delta: {res['comparisons']['6_vs_5_auroc_delta']:+.4f} "
              f"CI {res['comparisons']['6_vs_5_bootstrap']['ci_low']:+.4f}..."
              f"{res['comparisons']['6_vs_5_bootstrap']['ci_high']:+.4f}")
        print(f"  VERDICT: {res['verdict']['verdict']}  {res['verdict']['criteria']}")

    if not args.no_plots and "FULL" in results:
        make_plots(frames, results, out_dir, args.tag)

    # entropy quintile table is the headline descriptive artefact -> its own CSV
    rows = []
    for stratum, res in results.items():
        for key in ("quintiles_entropy", "quintiles_gap", "quintiles_teacher_entropy"):
            for r in res[key]:
                rows.append({"stratum": stratum, "predictor": key.replace("quintiles_", ""), **r})
    pd.DataFrame(rows).to_csv(out_dir / f"quintiles_{args.tag}.csv", index=False)
    pd.DataFrame([
        {"stratum": s, **r} for s, res in results.items() for r in res["within_offset"]
    ]).to_csv(out_dir / f"within_offset_{args.tag}.csv", index=False)

    for res in results.values():
        res.pop("_preds", None)
    (out_dir / f"analysis_{args.tag}.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n")
    print(f"\n[rprm] wrote analysis_{args.tag}.json, quintiles_{args.tag}.csv, "
          f"within_offset_{args.tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
