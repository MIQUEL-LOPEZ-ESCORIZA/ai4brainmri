"""Linear models on FreeSurfer anatomical features (the paper's "Anatomical" baseline).

- Diagnostic tasks: logistic regression on standardized features.
- Alzheimer's diagnosis: soft labels CN=0, MCI=0.5, AD=1 trained with log loss (paper Methods
  and Supplementary "label encoding" analysis). scikit-learn's LogisticRegression needs hard
  labels, so every sample with a fractional label y is duplicated as a positive with weight y and
  a negative with weight 1-y -- this minimizes exactly the soft-label binary cross-entropy.
- Brain age: ordinary least squares on the same features.
- `--model random_forest` reproduces the supplementary linear-vs-nonlinear comparison.

Hyperparameters are chosen by 5-fold cross-validation on the training set only (grids below,
as in the Supplementary preliminary analyses); test cohorts are never used for selection.

Writes per-scan predictions in the shared schema (`scan, label, prediction, ...`) so the
anatomical model can be compared with the ViTs and CNN by `analysis/compare_models.py`.

Usage:
    python -m baselines.anatomical.train_linear --cfg baselines/configs/anatomical_ad.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from downstream.metrics import bootstrap_ci, format_result, pairwise_ad_aucs, regression_metrics


def expand_soft_labels(x: np.ndarray, y: np.ndarray, groups: np.ndarray | None):
    """Duplicate fractional-label rows into weighted hard-label rows (see module docstring)."""
    hard = np.isin(y, (0.0, 1.0))
    soft = ~hard
    xs = np.concatenate([x[hard], x[soft], x[soft]])
    ys = np.concatenate([y[hard], np.ones(soft.sum()), np.zeros(soft.sum())]).astype(int)
    ws = np.concatenate([np.ones(hard.sum()), y[soft], 1 - y[soft]])
    gs = None if groups is None else np.concatenate([groups[hard], groups[soft], groups[soft]])
    return xs, ys, ws, gs


def build_search(task: str, model: str, groups_available: bool) -> GridSearchCV:
    cv = GroupKFold(n_splits=5) if groups_available else 5
    if task == "regression":
        if model == "linear":
            # OLS has no hyperparameters; GridSearchCV with a single candidate keeps one code path.
            return GridSearchCV(make_pipeline(StandardScaler(), LinearRegression()), {}, cv=cv,
                                scoring="neg_mean_absolute_error")
        return GridSearchCV(RandomForestRegressor(random_state=0),
                            {"n_estimators": [10, 50, 100], "max_depth": [5, 10, 20]},
                            cv=cv, scoring="neg_mean_absolute_error")
    if model == "linear":
        return GridSearchCV(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
            # no regularization vs. L2 with C=1 (scikit-learn default), as in the Supplement
            {"logisticregression__penalty": [None, "l2"], "logisticregression__C": [1.0]},
            cv=cv, scoring="neg_log_loss",
        )
    return GridSearchCV(RandomForestClassifier(random_state=0),
                        {"n_estimators": [10, 50, 100], "max_depth": [5, 10, 20],
                         "criterion": ["gini", "entropy"]},
                        cv=cv, scoring="neg_log_loss")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    features = [l.strip() for l in Path(cfg["features"]).read_text().splitlines() if l.strip() and not l.startswith("#")]
    label, subject, cohort = cfg["label_col"], cfg.get("subject_col", ""), cfg.get("cohort_col", "")
    task = cfg["task"]  # "classification" | "soft_classification" | "regression"

    train = pd.read_csv(cfg["train_csv"]).dropna(subset=features + [label])
    test = pd.read_csv(cfg["test_csv"]).dropna(subset=features + [label])
    if cfg.get("train_fraction", 1.0) < 1.0:  # Supplementary 10% / 30% / 100% experiments
        keep = train[subject].drop_duplicates().sample(frac=cfg["train_fraction"], random_state=cfg.get("seed", 42))
        train = train[train[subject].isin(keep)]

    x, y = train[features].to_numpy(float), train[label].to_numpy(float)
    groups = train[subject].to_numpy() if subject else None
    search = build_search("regression" if task == "regression" else "classification", cfg.get("model", "linear"),
                          groups is not None)

    if task == "soft_classification":
        x, y_fit, w, groups = expand_soft_labels(x, y, groups)
        weight_key = "logisticregression__sample_weight" if cfg.get("model", "linear") == "linear" else "sample_weight"
        search.fit(x, y_fit, groups=groups, **{weight_key: w})
    else:
        search.fit(x, y if task == "regression" else y.astype(int), groups=groups)
    print(f"Selected by 5-fold CV: {search.best_params_}")

    x_test, y_test = test[features].to_numpy(float), test[label].to_numpy(float)
    pred = search.predict(x_test) if task == "regression" else search.predict_proba(x_test)[:, 1]

    test_groups = test[subject].to_numpy() if subject else None
    for name, mask in [("overall", np.ones(len(test), bool))] + (
        [(c, (test[cohort] == c).to_numpy()) for c in sorted(test[cohort].unique())] if cohort else []
    ):
        g = None if test_groups is None else test_groups[mask]
        if task == "regression":
            r = regression_metrics(y_test[mask], pred[mask], groups=g)
            print(f"[{name}] MAE={format_result(r['mae'])}  r={format_result(r['pearson_r'])}")
        elif task == "soft_classification":
            r = pairwise_ad_aucs(y_test[mask], pred[mask], groups=g)
            print(f"[{name}] macro-AUC={format_result(r['macro'])}")
        else:
            from sklearn.metrics import roc_auc_score
            print(f"[{name}] AUROC={format_result(bootstrap_ci(roc_auc_score, y_test[mask], pred[mask], groups=g))}")

    if cfg.get("predictions_csv"):
        out = pd.DataFrame({"scan": test[cfg.get("scan_col", "img_path_T1_mni152")], "label": y_test, "prediction": pred})
        for col in (subject, cohort):
            if col:
                out[col] = test[col].to_numpy()
        Path(cfg["predictions_csv"]).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cfg["predictions_csv"], index=False)


if __name__ == "__main__":
    main()
