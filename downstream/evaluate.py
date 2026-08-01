"""CLI entry point: the final evaluation for a fine-tuned checkpoint.

Deliberately separate from train.py. train.py's own end-of-epoch validation uses plain
torchmetrics purely to pick the best checkpoint -- it is not the paper's reported metric. This
script is what the original repo's evaluation notebooks did by hand, after a separate eval-only
run of `main_finetune_final.py` had saved raw predictions to `.npy` files:
- `notebooks/eval_vit_AD_results (1).ipynb` / `notebooks/example_get_metrics_AD.ipynb`: the
  3-way pairwise AD-diagnosis AUC (AD-vs-CN, AD-vs-MCI, MCI-vs-CN) plus macro average, broken
  down per test cohort.
- `notebooks/confidence_interval.ipynb`: bootstrap CIs around every metric.
- `notebooks/healthy_aging_evaluation.ipynb`: bootstrap-CI'd Pearson correlation / MAE / MSE for
  age estimation.

This script runs inference once and reports all of that directly -- see `metrics.py` for the
actual metric implementations (including one deliberate correctness fix vs. the original
notebooks' pairwise-AUC filter; see its module docstring).

Usage:
    python -m downstream.evaluate --cfg downstream/configs/eval/ad_diagnosis.yaml
    python -m downstream.evaluate --cfg downstream/configs/eval/epilepsy_episurg.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from downstream.config import Config, load_config
from downstream.data import build_loader
from downstream.metrics import bootstrap_ci, format_result, pairwise_ad_aucs, regression_metrics
from downstream.model import build_model, load_finetuned_checkpoint
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", required=True, help="path to an eval YAML config")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a config field, e.g. --set data.test_csv=path/to/other_test.csv",
    )
    return parser.parse_args()


@torch.no_grad()
def predict(encoder, head, loader, config: Config, device: torch.device) -> tuple:
    """Run inference over `loader`, in dataset order. Returns `(raw_labels, predictions)` --
    `predictions` is P(positive class) for classification, the raw regression output for
    regression. Positional order matches the underlying CSV's rows (the loader doesn't shuffle
    in "test" mode), so callers can index both against that CSV for cohort breakdowns.
    """
    encoder.eval()
    head.eval()
    all_labels, all_preds = [], []
    for batch in loader:
        logits = head(encoder(batch["image"].to(device)))
        if config.model.num_classes == 1:
            preds = logits.squeeze(1).cpu().numpy()
        else:
            preds = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_labels.append(batch["pred_label"].numpy())
        all_preds.append(preds)
    return np.concatenate(all_labels), np.concatenate(all_preds)


def report(name: str, raw_labels: np.ndarray, preds: np.ndarray, config: Config) -> None:
    if len(raw_labels) < 2:
        print(f"[{name}] n={len(raw_labels)} -- too few samples to score")
        return

    if config.model.num_classes == 1:
        result = regression_metrics(raw_labels, preds)
        print(
            f"[{name}] n={len(raw_labels)}  "
            f"pearson_r={format_result(result['pearson_r'])}  "
            f"MAE={format_result(result['mae'])}  MSE={format_result(result['mse'])}"
        )
    elif config.train.smooth_mci_label:
        result = pairwise_ad_aucs(raw_labels, preds)
        parts = ", ".join(f"{k}={format_result(v)}" for k, v in result.items())
        print(f"[{name}] n={len(raw_labels)}  {parts}")
    else:
        result = bootstrap_ci(roc_auc_score, raw_labels, preds)
        print(f"[{name}] n={len(raw_labels)}  AUROC={format_result(result)}")


def main() -> None:
    args = parse_args()
    config = load_config(args.cfg, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{config.task_name}] device={device}")

    encoder, head = build_model(config)
    load_finetuned_checkpoint(encoder, head, config.model.pretrained)
    encoder, head = encoder.to(device), head.to(device)

    loader = build_loader(config, config.data.test_csv, "test")
    raw_labels, preds = predict(encoder, head, loader, config, device)

    report("overall", raw_labels, preds, config)

    if config.data.cohort_col:
        df = pd.read_csv(config.data.test_csv)
        assert len(df) == len(raw_labels), "test CSV row count changed between load and inference"
        for cohort in sorted(df[config.data.cohort_col].dropna().unique()):
            mask = (df[config.data.cohort_col] == cohort).to_numpy()
            report(str(cohort), raw_labels[mask], preds[mask], config)


if __name__ == "__main__":
    main()
