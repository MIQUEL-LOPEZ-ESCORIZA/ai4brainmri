"""Paired, participant-level bootstrap comparison of models evaluated on the same test cohort.

This is the statistical procedure described in the paper's "Statistical analysis" section:

1. Every model is scored on the *same* scans (per-scan predictions written by
   `downstream/evaluate.py`, `downstream/linear_probe.py`, `baselines/*`, all with the same
   CSV schema: `scan, label, prediction[, <subject_col>][, <cohort_col>]`).
2. For B bootstrap replicates, participants are resampled with replacement (all scans of a
   resampled participant are kept together) and *the same* resample is used for every model, so
   the between-model difference is computed on paired data.
3. For each pair of models we report the observed difference, its 95% percentile CI, and a
   two-sided bootstrap p-value (twice the smaller tail mass of the difference distribution
   around 0).
4. p-values within a cohort are adjusted for multiple comparisons with Holm's step-down method.

A difference whose CI includes 0 is *not* evidence of equivalence -- only that the data cannot
distinguish the models at this sample size. The paper's wording ("competitive under the tested
conditions") follows that interpretation.

Usage:
    python -m analysis.compare_models \
        --predictions asp=preds/ad_asp.csv mae=preds/ad_mae.csv anat=preds/ad_anat.csv cnn=preds/ad_cnn.csv \
        --metric ad_macro_auc --subject-col subject_id --cohort-col source_dataset \
        --reference asp --out results/ad_paired_comparison.csv
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

from downstream.metrics import _AD_DIAGNOSIS_PAIRS, bootstrap_indices


def ad_macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Macro average of the AD-vs-CN, AD-vs-MCI and MCI-vs-CN AUCs (labels 0 / 0.5 / 1)."""
    aucs = []
    for pos, neg in _AD_DIAGNOSIS_PAIRS.values():
        mask = (y == pos) | (y == neg)
        aucs.append(roc_auc_score((y[mask] == pos).astype(int), p[mask]))
    return float(np.mean(aucs))


def dice_mean(y: np.ndarray, p: np.ndarray) -> float:
    """For segmentation: `prediction` already holds the per-scan Dice; average it."""
    return float(np.mean(p))


METRICS = {
    "auroc": (roc_auc_score, "higher"),
    "ad_macro_auc": (ad_macro_auc, "higher"),
    "mae": (mean_absolute_error, "lower"),
    "dice": (dice_mean, "higher"),
}


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def load_predictions(specs: list[str], subject_col: str, cohort_col: str) -> tuple[pd.DataFrame, list[str]]:
    """Inner-join every model's predictions on `scan` so all models are scored on identical rows."""
    merged, names = None, []
    for spec in specs:
        name, _, path = spec.partition("=")
        df = pd.read_csv(path)
        keep = ["scan", "label"] + [c for c in (subject_col, cohort_col) if c and c in df.columns]
        df = df[keep + ["prediction"]].rename(columns={"prediction": name})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df[["scan", name]], on="scan", how="inner")
        names.append(name)
    return merged, names


def compare(df: pd.DataFrame, names: list[str], metric: str, subject_col: str, reference: str | None,
            n_bootstrap: int, seed: int) -> pd.DataFrame:
    metric_fn, direction = METRICS[metric]
    y = df["label"].to_numpy()
    preds = {n: df[n].to_numpy() for n in names}
    groups = df[subject_col].to_numpy() if subject_col else None

    pairs = [(reference, n) for n in names if n != reference] if reference else list(itertools.combinations(names, 2))
    observed = {n: metric_fn(y, preds[n]) for n in names}

    rng = np.random.RandomState(seed)
    diffs = {pair: [] for pair in pairs}
    for _ in range(n_bootstrap):
        idx = bootstrap_indices(len(y), rng, groups)
        try:
            scores = {n: metric_fn(y[idx], preds[n][idx]) for n in names}
        except ValueError:  # a resample without both classes
            continue
        for a, b in pairs:
            diffs[(a, b)].append(scores[a] - scores[b])

    rows = []
    for a, b in pairs:
        d = np.asarray(diffs[(a, b)])
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        rows.append({
            "model_a": a, "model_b": b,
            f"{metric}_a": observed[a], f"{metric}_b": observed[b],
            "difference": observed[a] - observed[b],
            "ci_lower": np.percentile(d, 2.5), "ci_upper": np.percentile(d, 97.5),
            "p_bootstrap": p, "n_scans": len(y),
            "n_participants": len(np.unique(groups)) if groups is not None else len(y),
            "better_is": direction, "valid_replicates": len(d),
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_bootstrap"].tolist())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", nargs="+", required=True, metavar="NAME=CSV")
    parser.add_argument("--metric", choices=sorted(METRICS), required=True)
    parser.add_argument("--subject-col", default="", help="participant ID column (participant-level bootstrap)")
    parser.add_argument("--cohort-col", default="", help="run the comparison separately per cohort")
    parser.add_argument("--reference", default=None, help="compare every model against this one only")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    df, names = load_predictions(args.predictions, args.subject_col, args.cohort_col)
    cohorts = [("overall", df)]
    if args.cohort_col:
        cohorts = [(str(c), df[df[args.cohort_col] == c]) for c in sorted(df[args.cohort_col].unique())]

    results = []
    for cohort, sub in cohorts:
        res = compare(sub, names, args.metric, args.subject_col, args.reference, args.n_bootstrap, args.seed)
        res.insert(0, "cohort", cohort)
        results.append(res)
    results = pd.concat(results, ignore_index=True)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(results.round(4).to_string(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
