"""Train and evaluate a segmentation model (BraTS 2023 whole tumor, ATLAS R2.0 stroke lesion).

Paper protocol (Methods + Supplementary):
- Single-channel T1-weighted input. BraTS provides a native T1 and a contrast-enhanced T1 (T1-CE)
  per participant: both are used as separate training samples; evaluation uses T1-CE. Neither
  BraTS nor ATLAS is re-registered (both are already standardized), so the manual masks are
  never interpolated.
- BraTS target: whole tumor = union of labels 1 (necrotic core), 2 (edema), 3 (enhancing).
  ATLAS target: lesion mask.
- Splits are at the participant level for BraTS (1,000 train / 251 test participants; optional
  10% training subset) and at the *site* level for ATLAS (21 training sites / 12 test sites), so
  no participant or site contributes to both sets. See `data/README.md`.
- Pretrained ViTs: UNETR decoder + LoRA-adapted ViT-B encoder (`model.adaptation: lora`).
  Baselines: UNETR and U-Net trained from scratch.
- AdamW, 150 epochs, random 128^3 crops (positive/negative-balanced around the lesion) for
  training, sliding-window inference over the full volume for validation and testing.
- Metric: per-scan Dice, mean with participant-level bootstrap 95% CI.

Usage:
    python -m downstream.segmentation.train_segmentation --cfg downstream/configs/segmentation/brats_asp.yaml
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from monai import data, transforms
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from downstream.config import Config, _merge_dict_into_dataclass
from downstream.metrics import bootstrap_ci, format_result
from downstream.segmentation.model import build_segmentation_model


def seg_transforms(roi: tuple, mode: str, binarize: bool) -> transforms.Compose:
    keys = ["image", "label"]
    t = [transforms.LoadImaged(keys=keys), transforms.EnsureChannelFirstd(keys=keys)]
    if binarize:  # whole tumor / any lesion
        t.append(transforms.Lambdad(keys="label", func=lambda x: (x > 0).astype(np.float32) if isinstance(x, np.ndarray) else (x > 0).float()))
    t.append(transforms.ScaleIntensityd(keys="image", minv=0.0, maxv=1.0))
    if mode == "train":
        t += [
            transforms.SpatialPadd(keys=keys, spatial_size=roi),
            transforms.RandCropByPosNegLabeld(keys=keys, label_key="label", spatial_size=roi, pos=1, neg=1, num_samples=2),
            transforms.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ]
    return transforms.Compose(t)


def make_loader(csv: str, cfg: dict, mode: str) -> data.DataLoader:
    df = pd.read_csv(csv)
    if mode == "train" and cfg.get("train_fraction", 1.0) < 1.0:  # e.g. the 10% BraTS experiment
        subjects = df[cfg["subject_col"]].drop_duplicates().sample(frac=cfg["train_fraction"], random_state=cfg.get("seed", 42))
        df = df[df[cfg["subject_col"]].isin(subjects)]
    items = [{"image": r[cfg["img_path_col"]], "label": r[cfg["mask_path_col"]]} for r in df.to_dict("records")]
    ds = data.Dataset(items, seg_transforms(tuple(cfg["roi"]), mode, cfg.get("binarize_label", True)))
    return data.DataLoader(ds, batch_size=cfg.get("batch_size", 2) if mode == "train" else 1,
                           shuffle=(mode == "train"), num_workers=cfg.get("num_workers", 4))


@torch.no_grad()
def per_scan_dice(model, loader, roi, device) -> np.ndarray:
    model.eval()
    metric = DiceMetric(include_background=False, reduction="none")
    scores = []
    for batch in loader:
        logits = sliding_window_inference(batch["image"].to(device), roi, 4, model, overlap=0.5)
        pred = torch.argmax(logits, dim=1, keepdim=True)
        onehot_pred = torch.cat([1 - pred, pred], dim=1).float()
        label = batch["label"].to(device).long()
        onehot_label = torch.cat([1 - label, label], dim=1).float()
        scores.append(metric(onehot_pred, onehot_label).cpu().numpy().ravel()[0])
    return np.nan_to_num(np.asarray(scores), nan=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config()
    config.roi = tuple(cfg["roi"])
    _merge_dict_into_dataclass(config.model, cfg.get("model", {}))
    model = build_segmentation_model(config, cfg["kind"]).to(device)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 1e-5))
    epochs = cfg.get("max_epochs", 150)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_dl = make_loader(cfg["train_csv"], cfg, "train")
    val_dl = make_loader(cfg["val_csv"], cfg, "val")
    best, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            loss = loss_fn(model(batch["image"].to(device)), batch["label"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % cfg.get("val_every", 10) == 0 or epoch == epochs - 1:
            dice = per_scan_dice(model, val_dl, config.roi, device).mean()
            print(f"epoch {epoch + 1}: loss={loss.item():.4f} val_dice={dice:.4f}")
            if dice > best:
                best, best_state = dice, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    Path(cfg["save_path"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, cfg["save_path"])

    test_df = pd.read_csv(cfg["test_csv"])
    dice = per_scan_dice(model, make_loader(cfg["test_csv"], cfg, "test"), config.roi, device)
    groups = test_df[cfg["subject_col"]].to_numpy() if cfg.get("subject_col") else None
    result = bootstrap_ci(lambda y, p: float(np.mean(p)), np.zeros_like(dice), dice, groups=groups)
    print(f"[test] Dice={format_result(result)}  (n={len(dice)} scans)")
    if cfg.get("predictions_csv"):
        out = pd.DataFrame({"scan": test_df[cfg["img_path_col"]], "label": 1.0, "prediction": dice})
        if cfg.get("subject_col"):
            out[cfg["subject_col"]] = groups
        Path(cfg["predictions_csv"]).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cfg["predictions_csv"], index=False)


if __name__ == "__main__":
    main()
