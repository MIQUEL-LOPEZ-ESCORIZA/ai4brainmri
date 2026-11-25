"""Supervised 3D CNN baseline (the paper's "CNN"): ResNet-18 trained from scratch.

Paper Methods, "Convolutional Neural Networks": 3D ResNet-18 (selected over ResNet-50 and a
160^3 input in the Supplementary preliminary analysis), up to 60 epochs, batch size 16, SGD with
initial learning rate 0.01, early stopping on validation loss. Single-channel 128^3 crops of the
MNI-registered T1 volume, same augmentation as the ViT pipeline (random crop + intensity shift
at train time, center crop at test time).

Losses:
- binary tasks and Alzheimer's diagnosis (soft labels CN=0, MCI=0.5, AD=1): one logit,
  `BCEWithLogitsLoss` -- the soft-label binary cross-entropy used for every AD model;
- brain age: one output, `MSELoss`.

The best-validation checkpoint is then scored on the test CSV with the same bootstrap metrics
as the ViTs, and per-scan predictions are written for `analysis/compare_models.py`.

Usage:
    python -m baselines.cnn.train_resnet --cfg baselines/configs/cnn_ad.yaml
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from monai.networks.nets import resnet18, resnet50
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from downstream.data import build_dataset, build_transforms
from downstream.metrics import bootstrap_ci, format_result, pairwise_ad_aucs, regression_metrics

ARCHS = {"resnet18": resnet18, "resnet50": resnet50}


def loader(cfg: dict, csv: str, mode: str) -> DataLoader:
    roi = tuple(cfg.get("roi", (128, 128, 128)))
    ds = build_dataset(csv, cfg.get("img_path_col", "img_path_T1_mni152"), cfg["label_col"], build_transforms(roi, mode))
    return DataLoader(ds, batch_size=cfg.get("batch_size", 16), shuffle=(mode == "train"),
                      num_workers=cfg.get("num_workers", 4), pin_memory=True)


def run_epoch(model, dl, criterion, device, optimizer=None) -> tuple[float, np.ndarray, np.ndarray]:
    model.train(optimizer is not None)
    total, labels, outputs = 0.0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for batch in dl:
            x, y = batch["image"].to(device), batch["pred_label"].float().to(device).view(-1, 1)
            out = model(x)
            loss = criterion(out, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(x)
            labels.append(y.cpu().numpy().ravel())
            outputs.append(out.detach().cpu().numpy().ravel())
    return total / len(dl.dataset), np.concatenate(labels), np.concatenate(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = cfg["task"]  # classification | soft_classification | regression
    model = ARCHS[cfg.get("arch", "resnet18")](spatial_dims=3, n_input_channels=1, num_classes=1).to(device)
    criterion = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.get("lr", 0.01), momentum=cfg.get("momentum", 0.0))

    train_dl, val_dl = loader(cfg, cfg["train_csv"], "train"), loader(cfg, cfg["val_csv"], "val")
    best, best_state, patience, waited = float("inf"), None, cfg.get("early_stopping_patience", 10), 0
    for epoch in range(cfg.get("max_epochs", 60)):
        tr_loss, _, _ = run_epoch(model, train_dl, criterion, device, optimizer)
        val_loss, _, _ = run_epoch(model, val_dl, criterion, device)
        print(f"epoch {epoch + 1}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best:
            best, best_state, waited = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    model.load_state_dict(best_state)
    Path(cfg["save_path"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, cfg["save_path"])

    test_df = pd.read_csv(cfg["test_csv"])
    _, y, out = run_epoch(model, loader(cfg, cfg["test_csv"], "test"), criterion, device)
    pred = out if task == "regression" else 1 / (1 + np.exp(-out))
    subject, cohort = cfg.get("subject_col", ""), cfg.get("cohort_col", "")
    groups = test_df[subject].to_numpy() if subject else None
    splits = [("overall", np.ones(len(y), bool))]
    if cohort:
        splits += [(c, (test_df[cohort] == c).to_numpy()) for c in sorted(test_df[cohort].unique())]
    for name, m in splits:
        g = None if groups is None else groups[m]
        if task == "regression":
            r = regression_metrics(y[m], pred[m], groups=g)
            print(f"[{name}] MAE={format_result(r['mae'])}")
        elif task == "soft_classification":
            print(f"[{name}] macro-AUC={format_result(pairwise_ad_aucs(y[m], pred[m], groups=g)['macro'])}")
        else:
            print(f"[{name}] AUROC={format_result(bootstrap_ci(roc_auc_score, y[m], pred[m], groups=g))}")

    if cfg.get("predictions_csv"):
        res = pd.DataFrame({"scan": test_df[cfg.get("img_path_col", "img_path_T1_mni152")], "label": y, "prediction": pred})
        for col in (subject, cohort):
            if col:
                res[col] = test_df[col].to_numpy()
        Path(cfg["predictions_csv"]).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(cfg["predictions_csv"], index=False)


if __name__ == "__main__":
    main()
