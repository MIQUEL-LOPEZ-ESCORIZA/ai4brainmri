"""Linear-probing sweep across pretraining checkpoints -- paper Figure 4(b): how anatomically/
demographically predictive the encoder's embedding becomes as pretraining progresses.

Merged from the original repo's `validation.py` and `validation_mae_seg_combined.py`, which were
~500-line near-duplicates differing only in which checkpoint files they'd probe and which key of
the checkpoint dict held the encoder weights. That's now the single `--checkpoint_source`
choice below; everything else (data loading, transforms, feature extraction, the sklearn probes,
the output plot) was already identical between the two and is unchanged here.

Not training -- this is entirely decoupled from seg_mae_combined_train.py. It scans
`--checkpoints_dir` for every checkpoint file that matches the chosen source's convention, and
for each one: loads that epoch's encoder, extracts CLS embeddings over the train/test CSVs, fits
a fresh `sklearn` `LogisticRegression` (for `gender`) or `Ridge` (for continuous features like
age, CortexVol) per `--features_to_validate`, and records train/test accuracy or R^2. After
sweeping every qualifying epoch it saves one plot (`results_with_full_image.png` in
`--output_dir`) showing each metric's trend across epochs -- no structured output is saved
separately (that's also unchanged from the original).

Usage:
    python -m pretraining.linear_probing.linear_probe_sweep \\
        --checkpoint_source mae \\
        --train_dataset_csv ... --test_dataset_csv ... \\
        --features_to_validate age gender CortexVol \\
        --checkpoints_dir ... --cache_directory ... \\
        --vit_params '{"in_chans": 1, "img_size": [128, 128, 128], "patch_size": [16, 16, 16], ...}' \\
        --output_dir ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from monai import transforms
from monai.data import PersistentDataset
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from src.models.vit import ViT


# -----------------------------------------------------------------------------
# Checkpoint sources -- the one real difference between the two original scripts
# -----------------------------------------------------------------------------

def _extract_epoch_general(name: str) -> Optional[int]:
    """Multiple fallback patterns: ..._epoch_999.pt, checkpoint123.pth, or any trailing digits
    before the extension. Was `validation.py::extract_epoch_from_checkpoint`."""
    for pattern in (
        r"epoch[_-]?(\d+)(?=\.(?:pt|pth)$)",
        r"checkpoint[_-]?(\d+)(?=\.(?:pt|pth)$)",
        r"(\d+)(?=\.(?:pt|pth)$)",
    ):
        m = re.search(pattern, name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _extract_epoch_asp(name: str) -> Optional[int]:
    """Was `validation_mae_seg_combined.py::extract_epoch_from_checkpoint`."""
    m = re.search(r"checkpoint_seg_(\d+)\.pth", name)
    return int(m.group(1)) if m else None


@dataclass
class CheckpointSource:
    state_key: str  # key inside the checkpoint dict holding the encoder's state_dict
    matches: Callable[[str], bool]  # which filenames in --checkpoints_dir to consider at all
    extract_epoch: Callable[[str], Optional[int]]
    epoch_filter: Callable[[int], bool] = field(default=lambda epoch: True)  # extra per-epoch rule


CHECKPOINT_SOURCES = {
    # Raw MAE-pretrained checkpoints (single top-level `state_dict` key) -- e.g. the plain-MAE
    # snapshots or a frozen-encoder segmentation-decoder run's own checkpoints. Was
    # validation.py, used by val_job_mae.sh and val_job_mae_seg.sh in the original repo.
    "mae": CheckpointSource(
        state_key="state_dict",
        matches=lambda name: name.endswith(".pt"),
        extract_epoch=_extract_epoch_general,
    ),
    # seg_mae_combined_train.py's own checkpoints (`model_mae`/`model_seg` sub-keys). Was
    # validation_mae_seg_combined.py, used by val_job_mae_seg_combined.sh and
    # val_job_mae_seg_combined_all.sh. Only every 5th epoch is probed, matching the original.
    "asp": CheckpointSource(
        state_key="model_mae",
        matches=lambda name: "seg" in name and name.endswith(".pth"),
        extract_epoch=_extract_epoch_asp,
        epoch_filter=lambda epoch: epoch % 5 == 0,
    ),
}


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_source", choices=sorted(CHECKPOINT_SOURCES), required=True,
                        help="Which checkpoint convention --checkpoints_dir holds -- see CHECKPOINT_SOURCES above.")
    parser.add_argument("--train_dataset_csv", type=str, required=True,
                        help="Path to the training dataset CSV file.")
    parser.add_argument("--test_dataset_csv", type=str, required=True,
                        help="Path to the test dataset CSV file.")
    parser.add_argument("--features_to_validate", type=str, nargs="+", required=True,
                        help="Features to validate (e.g., age, gender, CortexVol).")
    parser.add_argument("--cache_directory", type=str, required=True, default=None,
                        help="Directory to cache the processed dataset.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for the dataloader.")
    parser.add_argument("--transform", type=int, default=2,
                        help="Loading transform to be applied.")
    parser.add_argument("--checkpoints_dir", type=str, required=True,
                        help="Directory containing the model checkpoints to sweep over.")
    parser.add_argument("--vit_params", type=str, required=True,
                        help="JSON string representing the ViT model parameters.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the results plot.")
    parser.add_argument("--cut_off_epoch", type=int, default=-1,
                        help="Only probe checkpoints after this epoch.")

    args = parser.parse_args()
    args.vit_params = json.loads(args.vit_params)
    return args


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_and_preprocess_data(csv_path: str, features_to_validate: list) -> pd.DataFrame:
    """Load and preprocess the dataset."""
    df = pd.read_csv(csv_path)
    df.fillna(-99, inplace=True)
    print("Dataframe has size:", len(df))
    print("First row of dataframe is:", df[features_to_validate].head(1))
    return df


def create_transform(img_size, transform_number: int) -> transforms.Compose:
    """Define preprocessing transforms."""
    if transform_number == 1:
        return transforms.Compose([
            transforms.LoadImaged(keys=["image"], reader="NibabelReader", image_only=True, allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=["image"], allow_missing_keys=True),
            transforms.Orientationd(keys=["image"], axcodes="RAS", allow_missing_keys=True),
            transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=3, allow_missing_keys=True),
            transforms.CenterSpatialCropd(keys=["image"], roi_size=img_size, allow_missing_keys=True),
            transforms.CastToTyped(keys=["image"], dtype=np.float16),
        ])
    if transform_number == 4:
        return transforms.Compose([
            transforms.LoadImaged(keys=["image"], reader="NibabelReader", image_only=True, allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=["image"], allow_missing_keys=True),
            transforms.Orientationd(keys=["image"], axcodes="RAS", allow_missing_keys=True),
            transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=3, allow_missing_keys=True),
            transforms.CenterSpatialCropd(keys=["image"], roi_size=img_size, allow_missing_keys=True),
            transforms.ToTensord(keys=["image"], allow_missing_keys=True),
        ])
    if transform_number == 2:
        return transforms.Compose([
            transforms.LoadImaged(keys=["image"], reader="NibabelReader", image_only=True, allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=["image"], allow_missing_keys=True),
            transforms.CenterSpatialCropd(keys=["image"], roi_size=img_size, allow_missing_keys=True),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True, allow_missing_keys=True,
            ),
            transforms.ToTensord(keys=["image"], allow_missing_keys=True),
        ])
    if transform_number == 3:
        return transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"], image_only=False, allow_missing_keys=True, reader="NibabelReader"),
            transforms.EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True, allow_missing_keys=True,
            ),
            transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=img_size, allow_missing_keys=True),
            transforms.ToTensord(keys=["image", "label"], allow_missing_keys=True),
        ])
    if transform_number == 5:
        return transforms.Compose([
            transforms.LoadImaged(keys=["image"], reader="NibabelReader", image_only=True, allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=["image"], allow_missing_keys=True),
            transforms.CenterSpatialCropd(keys=["image"], roi_size=img_size, allow_missing_keys=True),
            transforms.NormalizeIntensityd(keys=["image"], allow_missing_keys=True),
            transforms.ToTensord(keys=["image"], allow_missing_keys=True),
        ])
    raise ValueError(f"Unknown --transform {transform_number}")


class ImageDataset(Dataset):
    def __init__(self, cache_dir, validation_dataset_df, features_to_validate, transform_function):
        self.cache_dir = cache_dir
        self.features_to_validate = features_to_validate
        os.makedirs(self.cache_dir, exist_ok=True)

        data_list = [
            {"image": row["img_path_T1_mni152"], **{col: row[col] for col in features_to_validate}}
            for _, row in validation_dataset_df.iterrows()
        ]
        self.cache_dataset = PersistentDataset(data=data_list, transform=transform_function, cache_dir=cache_dir)

    def __len__(self):
        return len(self.cache_dataset)

    def __getitem__(self, idx):
        data = self.cache_dataset[idx]
        img_tensor = data["image"].float()
        metadata = {col: data[col] for col in self.features_to_validate}
        return img_tensor, metadata


# -----------------------------------------------------------------------------
# Model loading + feature extraction
# -----------------------------------------------------------------------------

def load_model(checkpoint_path: str, vit_params: dict, state_key: str) -> ViT:
    """Load a ViT encoder's weights from one pretraining checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vit_encoder = ViT(**vit_params)

    state_dict = {
        k.replace("module.", ""): v
        for k, v in checkpoint[state_key].items()
        if k.startswith("module.")
    }
    missing_keys, unexpected_keys = vit_encoder.load_state_dict(state_dict, strict=False)
    vit_encoder.eval()

    if not missing_keys and not unexpected_keys:
        print("ViT encoder loaded successfully without missing or unexpected keys.")
    else:
        print("ViT encoder loaded with warnings:")
        if missing_keys:
            print(f" - Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f" - Unexpected keys: {unexpected_keys}")

    return vit_encoder


def extract_features(vit_model: ViT, dataloader: DataLoader, device: torch.device) -> pd.DataFrame:
    """Extract features from images using the ViT model."""
    vit_model.to(device)
    features_list = []
    total_batches = len(dataloader)

    for it, (img_tensors, metadata) in enumerate(dataloader):
        img_tensors = img_tensors.to(device).float()
        with torch.no_grad():
            features_batch = vit_model(img_tensors).cpu().numpy()
        for i in range(len(features_batch)):
            features_list.append({
                "features": features_batch[i],
                **{
                    key: metadata[key][i].item() if isinstance(metadata[key][i], torch.Tensor) else metadata[key][i]
                    for key in metadata
                },
            })
        progress = (it + 1) / total_batches * 100
        if int(progress) % 10 == 0:
            print(f"Validation Progress: {int(progress)}% completed")

    return pd.DataFrame(features_list)


def train_and_evaluate(train_features_df: pd.DataFrame, test_features_df: pd.DataFrame, features_to_validate: list) -> dict:
    """Fit a linear probe (LogisticRegression for gender, Ridge otherwise) per feature."""
    X_train = np.array(train_features_df["features"].tolist())
    X_test = np.array(test_features_df["features"].tolist())

    results = {}
    for feature in features_to_validate:
        y_train = train_features_df[feature].values
        y_test = test_features_df[feature].values

        # -99 is the load_and_preprocess_data() sentinel for missing values
        cond_train = y_train != -99
        cond_test = y_test != -99
        X_train_filtered, y_train_filtered = X_train[cond_train], y_train[cond_train]
        X_test_filtered, y_test_filtered = X_test[cond_test], y_test[cond_test]

        if feature == "gender":
            model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000))])
            model.fit(X_train_filtered, y_train_filtered)
            results[f"{feature}_train_accuracy"] = accuracy_score(y_train_filtered, model.predict(X_train_filtered))
            results[f"{feature}_test_accuracy"] = accuracy_score(y_test_filtered, model.predict(X_test_filtered))
        else:
            model = Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(alpha=10))])
            model.fit(X_train_filtered, y_train_filtered)
            results[f"{feature}_r2_train"] = r2_score(y_train_filtered, model.predict(X_train_filtered))
            results[f"{feature}_r2_test"] = r2_score(y_test_filtered, model.predict(X_test_filtered))

    return results


def visualize_df(features_df: pd.DataFrame, features_to_validate: list) -> None:
    print(features_df[features_to_validate].head())


def plot_results(results: dict, epochs, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for metric, values in results.items():
        plt.plot(epochs, values, marker="o", linestyle="-", label=metric)
    plt.legend()
    plt.grid(True)
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Metrics Across Checkpoints")
    plt.savefig(os.path.join(output_dir, "results_with_full_image"))
    plt.close()


def main() -> None:
    args = parse_arguments()
    source = CHECKPOINT_SOURCES[args.checkpoint_source]

    print(f"Checkpoint source: {args.checkpoint_source}")
    print(f"Training dataset CSV: {args.train_dataset_csv}")
    print(f"Test dataset CSV: {args.test_dataset_csv}")
    print(f"Features to validate: {args.features_to_validate}")
    print(f"Checkpoints directory: {args.checkpoints_dir}")
    print(f"Output directory: {args.output_dir}")

    train_transform = create_transform(args.vit_params["img_size"], args.transform)
    train_dataset = ImageDataset(
        args.cache_directory, load_and_preprocess_data(args.train_dataset_csv, args.features_to_validate),
        args.features_to_validate, train_transform,
    )
    test_transform = create_transform(args.vit_params["img_size"], args.transform)
    test_dataset = ImageDataset(
        args.cache_directory, load_and_preprocess_data(args.test_dataset_csv, args.features_to_validate),
        args.features_to_validate, test_transform,
    )

    loader_kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_dataloader = DataLoader(train_dataset, **loader_kwargs)
    test_dataloader = DataLoader(test_dataset, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for feature in args.features_to_validate:
        if feature == "gender":
            results[f"{feature}_train_accuracy"] = []
            results[f"{feature}_test_accuracy"] = []
        else:
            results[f"{feature}_r2_train"] = []
            results[f"{feature}_r2_test"] = []
    epochs = []

    for checkpoint in sorted(os.listdir(args.checkpoints_dir)):
        if not source.matches(checkpoint):
            continue
        epoch = source.extract_epoch(checkpoint)
        if epoch is not None and not (epoch > args.cut_off_epoch and source.epoch_filter(epoch)):
            continue

        epochs.append(epoch)
        print("Validating epoch:", epoch)
        vit_model = load_model(os.path.join(args.checkpoints_dir, checkpoint), args.vit_params, source.state_key)
        train_features_df = extract_features(vit_model, train_dataloader, device)
        test_features_df = extract_features(vit_model, test_dataloader, device)
        visualize_df(train_features_df, args.features_to_validate)
        metrics = train_and_evaluate(train_features_df, test_features_df, args.features_to_validate)
        print(metrics)
        for key in results:
            results[key].append(metrics[key])

    sorted_indices = np.argsort(epochs)
    epochs = np.array(epochs)[sorted_indices]
    for key in results:
        results[key] = np.array(results[key])[sorted_indices]

    plot_results(results, epochs, args.output_dir)
    print("Image containing metrics saved in:", args.output_dir)


if __name__ == "__main__":
    main()
