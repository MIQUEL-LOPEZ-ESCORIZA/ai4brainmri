"""Final-evaluation metrics, shared by `evaluate.py` and `linear_probe.py`.

Ported from the original repo's analysis notebooks (`notebooks/confidence_interval.ipynb`,
`notebooks/eval_vit_AD_results (1).ipynb`, `notebooks/healthy_aging_evaluation.ipynb`), which is
where the paper's actual reported numbers come from -- not from `main_finetune_final.py`'s
inline torchmetrics, which only picked the best checkpoint during training.

One deliberate deviation: the original `calc_aucs` (in both `eval_vit_AD_results (1).ipynb` and
`example_get_metrics_AD.ipynb`) computes the three pairwise AD-diagnosis AUCs by filtering rows
of the *smoothed* 2-column target (`[1-label/2, label/2]`) on inconsistent thresholds -- e.g. the
"AD_CN" pair filters out `col0 == 0.5`, which is the AD rows, not the MCI rows its own comment
implies it wants to exclude. Working through the arithmetic, that filter leaves `y_true` taking
values `{0, 0.25}` for the CN/MCI rows it keeps, which isn't a valid binary label for
`sklearn.roc_curve` and would raise `ValueError: Data is not binary` on any evaluation run that
actually contains MCI cases. `pairwise_ad_aucs` below instead filters directly on the raw
0/0.5/1 label for each pair -- unambiguous, and matches what the "AD_CN"/"AD_MCI"/"MCI_CN"
naming actually intends.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

# (positive label, negative label) for each pairwise AD-diagnosis comparison.
# Raw label convention: healthy (CN) = 0, MCI = 0.5, AD = 1.
_AD_DIAGNOSIS_PAIRS = {
    "AD_CN": (1.0, 0.0),
    "AD_MCI": (1.0, 0.5),
    "MCI_CN": (0.5, 0.0),
}


def bootstrap_indices(n: int, rng: np.random.RandomState, groups=None) -> np.ndarray:
    """One bootstrap resample of row indices. With `groups` (e.g. participant IDs), whole groups
    are resampled with replacement and all their rows kept (cluster bootstrap), so repeated scans
    of one participant never count as independent observations."""
    if groups is None:
        return rng.randint(0, n, n)
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    rows_by_group = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    picked = rng.randint(0, len(unique), len(unique))
    return np.concatenate([rows_by_group[g] for g in picked])


def bootstrap_ci(metric_fn, y_true, y_pred, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42,
                 groups=None) -> dict:
    """`metric_fn(y_true, y_pred) -> float`, resampled `n_bootstrap` times with replacement.
    Mirrors the original repo's `add_bootstrap_CI` (`notebooks/confidence_interval.ipynb`,
    `notebooks/healthy_aging_evaluation.ipynb`): percentile-method CI, same defaults.
    Pass `groups` (participant IDs) for a participant-level bootstrap. Resamples where the metric
    is undefined (e.g. a single class for AUROC) are skipped.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_bootstrap):
        idx = bootstrap_indices(n, rng, groups)
        try:
            stats.append(metric_fn(y_true[idx], y_pred[idx]))
        except ValueError:
            continue
    return {
        "value": metric_fn(y_true, y_pred),
        "ci_lower": np.percentile(stats, (1 - ci) / 2 * 100),
        "ci_upper": np.percentile(stats, (1 + ci) / 2 * 100),
    }


def pairwise_ad_aucs(raw_label, prob_ad, n_bootstrap: int = 1000, groups=None) -> dict:
    """Three pairwise AD-diagnosis AUCs (AD-vs-CN, AD-vs-MCI, MCI-vs-CN) plus their macro
    average, each with a bootstrap CI. `raw_label` is 0 (CN) / 0.5 (MCI) / 1 (AD);
    `prob_ad` is the predicted P(AD). See the module docstring for how this differs from the
    original notebooks' version.
    """
    raw_label, prob_ad = np.asarray(raw_label), np.asarray(prob_ad)
    results = {}
    for name, (pos, neg) in _AD_DIAGNOSIS_PAIRS.items():
        mask = (raw_label == pos) | (raw_label == neg)
        y_true = (raw_label[mask] == pos).astype(int)
        g = None if groups is None else np.asarray(groups)[mask]
        results[name] = bootstrap_ci(roc_auc_score, y_true, prob_ad[mask], n_bootstrap, groups=g)
    results["macro"] = {
        "value": float(np.mean([results[k]["value"] for k in _AD_DIAGNOSIS_PAIRS])),
        "ci_lower": float(np.mean([results[k]["ci_lower"] for k in _AD_DIAGNOSIS_PAIRS])),
        "ci_upper": float(np.mean([results[k]["ci_upper"] for k in _AD_DIAGNOSIS_PAIRS])),
    }
    return results


def regression_metrics(y_true, y_pred, n_bootstrap: int = 1000, groups=None) -> dict:
    """Pearson correlation + MAE + MSE, each with a bootstrap CI -- matches
    `notebooks/healthy_aging_evaluation.ipynb` (used for age estimation)."""
    pearson_r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    return {
        "pearson_r": bootstrap_ci(pearson_r, y_true, y_pred, n_bootstrap, groups=groups),
        "mae": bootstrap_ci(mean_absolute_error, y_true, y_pred, n_bootstrap, groups=groups),
        "mse": bootstrap_ci(mean_squared_error, y_true, y_pred, n_bootstrap, groups=groups),
    }


def format_result(result: dict) -> str:
    return f"{result['value']:.4f} ({result['ci_lower']:.4f}, {result['ci_upper']:.4f})"
