"""CSV-driven MRI dataset and transforms for the ASP downstream pipeline.

Mirrors `mae/src/data/datasets.py` + the `centercrop` branch of
`mae/src/data/transforms.py::resnet_transforms` in the original repo, trimmed to the single-crop
classification/regression case (no segmentation masks, no multi-patch grids — every ASP
downstream config used a single 128^3 crop, `VIT.NUM_PATCHES: 1`).
"""
from __future__ import annotations

import pandas as pd
from monai import data, transforms
from torch.utils.data import DataLoader

from downstream.config import Config


def build_transforms(roi: tuple, mode: str) -> transforms.Compose:
    """`train`: pad to 256^3 then random-crop `roi` (+ intensity jitter), for augmentation.
    `val`/`test`: center-crop `roi` directly, no padding, no augmentation.
    """
    load = [
        transforms.LoadImaged(keys=["image"], image_only=False, reader="NibabelReader"),
        transforms.EnsureChannelFirstd(keys=["image"]),
    ]
    if mode == "train":
        return transforms.Compose(
            load
            + [
                transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=[256, 256, 256]),
                transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                transforms.RandSpatialCropd(keys=["image"], roi_size=roi),
                transforms.ToTensord(keys=["image"]),
            ]
        )
    return transforms.Compose(
        load
        + [
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.CenterSpatialCropd(keys=["image"], roi_size=roi),
            transforms.ToTensord(keys=["image"]),
        ]
    )


def build_dataset(csv_path: str, img_path_col: str, label_col: str, transform: transforms.Compose, cache_dir: str = ""):
    df = pd.read_csv(csv_path)
    items = [{"image": row[img_path_col], "pred_label": float(row[label_col])} for row in df.to_dict("records")]
    if cache_dir:
        return data.PersistentDataset(data=items, transform=transform, cache_dir=cache_dir)
    return data.Dataset(data=items, transform=transform)


def build_image_dataset(image_paths: list, transform: transforms.Compose):
    """Same as `build_dataset`, but for unlabeled images (used by `linear_probe.py` to embed a
    shared pool of scans that isn't tied to any single task's labels)."""
    items = [{"image": path} for path in image_paths]
    return data.Dataset(data=items, transform=transform)


def build_loader(config: Config, csv_path: str, mode: str) -> DataLoader:
    """`mode` is `"train"` (enables augmentation + shuffling) or `"val"`/`"test"`."""
    d = config.data
    dataset = build_dataset(csv_path, d.img_path_col, d.label_col, build_transforms(config.roi, mode))
    return DataLoader(
        dataset,
        batch_size=d.batch_size,
        shuffle=(mode == "train"),
        num_workers=d.num_workers,
        pin_memory=True,
    )
