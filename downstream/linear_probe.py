"""CLI entry point: linear probing on a frozen ASP-pretrained encoder.

Extracts CLS-token embeddings with a frozen encoder, then fits a scikit-learn linear model on
top of them -- the paper's "linear probing" strategy, used for tasks with very little labeled
data (Parkinson's, epilepsy, glioma/IDH mutation). Two ways a task gets its embeddings, both
traced from the original repo's `notebooks/downstream_linprob_pipeline (1).ipynb`:

- **Self-contained** (glioma): `data.train_csv`/`test_csv` carry `data.img_path_col` directly;
  embeddings are extracted fresh from each task's own images. `data.group_col`, if set, averages
  predictions per group (e.g. patient ID) before scoring -- UCSF-PDGM has multiple scans/patient.
- **Shared pool** (Parkinson's, epilepsy): `data.pool_csv` is a large pool of scans embedded
  once; each task's train/test CSV carries only `data.join_col` (e.g. "subject_session") + a
  label, and is matched against the pool by that key. This avoids re-embedding the same scans
  for every disease that happens to draw from the same cohort.

The final metric is bootstrap-CI'd the same way as evaluate.py (see metrics.py) -- matches
`notebooks/confidence_interval.ipynb`, which was applied uniformly regardless of adaptation
strategy in the original repo.

Usage:
    python -m downstream.linear_probe --cfg downstream/configs/linear_probe/glioma_registered.yaml
    python -m downstream.linear_probe --cfg downstream/configs/linear_probe/parkinsons.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch.utils.data import DataLoader

from downstream.config import Config, load_config
from downstream.data import build_image_dataset, build_transforms
from downstream.metrics import bootstrap_ci, format_result
from downstream.model import build_model, load_asp_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", required=True, help="path to a task YAML config")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a config field, e.g. --set data.train_csv=path/to/large_train.csv",
    )
    return parser.parse_args()


@torch.no_grad()
def embed(encoder, image_paths: list, config: Config, device: torch.device) -> np.ndarray:
    """Frozen-encoder CLS embedding for each path, in order. Center-crop, no augmentation --
    embeddings should be deterministic since the encoder is only ever run once per image."""
    encoder.eval()
    dataset = build_image_dataset(image_paths, build_transforms(config.roi, "test"))
    loader = DataLoader(dataset, batch_size=config.data.batch_size, num_workers=config.data.num_workers)
    embeddings = [encoder(batch["image"].to(device)).cpu().numpy() for batch in loader]
    return np.concatenate(embeddings)


def build_embedding_pool(config: Config, encoder, device: torch.device) -> dict:
    """Embed every scan in `data.pool_csv` once, keyed by `data.join_col`."""
    df = pd.read_csv(config.data.pool_csv)
    embeddings = embed(encoder, df[config.data.img_path_col].tolist(), config, device)
    return dict(zip(df[config.data.join_col], embeddings))


def load_split(config: Config, encoder, device: torch.device, pool: dict | None, csv_path: str) -> tuple:
    """Return `(X, y, groups)` for one train/test split.

    `groups` is `None` unless `data.group_col` is set (only needed for scoring, not fitting).
    """
    df = pd.read_csv(csv_path)
    if pool is not None:
        x = np.stack([pool[key] for key in df[config.data.join_col]])
    else:
        x = embed(encoder, df[config.data.img_path_col].tolist(), config, device)

    y = df[config.data.label_col]
    if config.data.label_map:
        y = y.map(config.data.label_map)
    y = y.to_numpy()

    groups = df[config.data.group_col] if config.data.group_col else None
    return x, y, groups


def score(y_true: np.ndarray, y_pred: np.ndarray, groups, num_classes: int) -> dict:
    if groups is not None:
        grouped = pd.DataFrame({"y": y_true, "pred": y_pred}).groupby(groups.to_numpy()).mean()
        y_true, y_pred = grouped["y"].to_numpy(), grouped["pred"].to_numpy()
    metric_fn = mean_absolute_error if num_classes == 1 else roc_auc_score
    return bootstrap_ci(metric_fn, y_true, y_pred)


def main() -> None:
    args = parse_args()
    config = load_config(args.cfg, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{config.task_name}] device={device}")

    encoder, _ = build_model(config)
    if config.model.pretrained:
        result = load_asp_checkpoint(encoder, config.model.pretrained)
        print(f"Loaded ASP checkpoint {config.model.pretrained}: {result}")
    encoder = encoder.to(device)

    pool = build_embedding_pool(config, encoder, device) if config.data.pool_csv else None

    train_x, train_y, _ = load_split(config, encoder, device, pool, config.data.train_csv)
    test_x, test_y, test_groups = load_split(config, encoder, device, pool, config.data.test_csv)

    if config.model.num_classes == 1:
        model = LinearRegression().fit(train_x, train_y)
        preds = model.predict(test_x)
    else:
        model = LogisticRegression(max_iter=1000).fit(train_x, train_y)
        preds = model.predict_proba(test_x)[:, 1]

    result = score(test_y, preds, test_groups, config.model.num_classes)
    metric_name = "MAE" if config.model.num_classes == 1 else "AUROC"
    print(f"test {metric_name}: {format_result(result)}")


if __name__ == "__main__":
    main()
