


import os
import argparse
import torch
import wandb  # W&B Integration


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np

from monai.data.utils import list_data_collate
from monai import    transforms
from monai.data import PersistentDataset

import random




import json
import matplotlib.pyplot as plt
import shutil


from src.utils import utils_seg as utils




def extract_image_segmentation_datalist(csv_path, dataset_percentage):
    """
    Extracts image and segmentation paths from the given CSV file.

    Args:
        csv_path (str): Path to the CSV file containing the dataset information.

    Returns:
        list: A list of dictionaries where each entry contains 'image' and 'label' keys.
    """
    # Read the CSV file
    full_data = pd.read_csv(csv_path)

    num_samples = int(len(full_data) * dataset_percentage)
    data = full_data.sample(n=num_samples, random_state=42)

    # Extract image and segmentation paths
    datalist = []
    for _, row in data.iterrows():
        # Only include rows with valid image and segmentation paths
        if row["is_img_path_T1_mni152"] and row["is_aparc+aseg"]:
            sample_dict = {
                "image": row["img_path_T1_mni152"],
                "label": row["img_seg_path_T1"],
            }
            datalist.append(sample_dict)
    
    return datalist

def grouping():
    return [[2,41], [16], [17,53]]  # 2 groups

def grouping_total():
    return [
        [26, 58],
        [18, 54],
        [16],
        [251, 252, 253, 254, 255],
        [11, 50],
        [8, 47],
        [7, 46],
        [2, 41],
        [17, 53],
        [4, 43],
        [13, 52],
        [12, 51],
        [10, 49],
        [77],
        [1006, 2006],
        [1007, 2007],
        [1008, 2008],
        [1009, 2009],
        [1035, 2035],
        [1011, 2011],
        [1012, 2012],
        [1013, 2013],
        [1014, 2014],
        [1015, 2015],
        [1016, 2016],
        [1018, 2018],
        [1019, 2019],
        [1020, 2020],
        [1022, 2022],
        [1023, 2023],
        [1024, 2024],
        [1025, 2025],
        [1026, 2026],
        [1027, 2027],
        [1028, 2028],
        [1029, 2029],
        [1030, 2030],
        [1031, 2031],
        [1033, 2033],
    ]

class TargetMaskTransform_grouped(transforms.MapTransform):
    def __init__(self, keys, groups, allow_missing_keys=False):
        """
        Randomly selects ONE group from `groups` and creates a binary mask.

        Args:
            keys (list): Keys in the dictionary to apply the transform.
            groups (list of lists): A list of groups, where each group contains labels to be merged into a single mask.
            allow_missing_keys (bool): If True, allow keys to be missing in the data dictionary.
        """
        super().__init__(keys, allow_missing_keys)
        self.groups = [torch.tensor(group) for group in groups]  # Convert groups to tensor lists

    def __call__(self, data):
        """
        Args:
            data (dict): The input data dictionary.

        Returns:
            dict: The data dictionary with:
                - "label": Binary mask for the **randomly chosen group**
                - "group_index": The index of the selected group
        """
        d = dict(data)
        mask = d["label"]  # Get the segmentation mask

        # Ensure mask is a PyTorch tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)

        original_dtype = mask.dtype  # Store original dtype

        # Randomly select ONE group
        group_idx = random.randint(0, len(self.groups) - 1)  # Pick a random group index
        selected_group = self.groups[group_idx]  # Get the selected group labels

        # Create a binary mask for the selected group
        binary_mask = torch.isin(mask, selected_group).to(original_dtype)  # Convert to original dtype

        # Update data dictionary
        d["label"] = binary_mask # Shape: (1, H, W, D)
        d["group_index"] = torch.tensor(group_idx, dtype=torch.long)  # Save the group index

        return d


def image_label_transform_total(groups, vit_img_size):
    """
    Create a joint transform for images and labels.

    Args:
        groups (list of lists): The grouped labels.
        vit_img_size (tuple): Size of the input image for ViT.

    Returns:
        Compose: A transform pipeline for both images and labels.
    """

    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"], reader="NibabelReader", image_only=True, allow_missing_keys=False),
        transforms.EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=False),

        transforms.RandSpatialCropd(
            keys=["image", "label"],
            roi_size=vit_img_size,
            random_center=True,
            random_size=False,
            allow_missing_keys=True,
        ),

        transforms.ScaleIntensityRanged(
            keys=["image"],
            a_min=0, 
            a_max=255,
            b_min=0.0,
            b_max=1.0,
            clip=True,
            allow_missing_keys=True,
        ),

        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0, allow_missing_keys=True),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1, allow_missing_keys=True),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2, allow_missing_keys=True),

        transforms.ToTensord(keys=["image", "label"], allow_missing_keys=True),

        # Apply new transform to randomly extract one group
        TargetMaskTransform_grouped(keys=["label"], groups=groups),
    ])


class ImageDataset:
    """
    A dataset class that provides access to images and segmentation labels.
    """
    def __init__(self, data_list, cache_dir, transform):
        self.clear_cache(cache_dir)
        self.dataset = PersistentDataset(data=data_list, transform=transform, cache_dir=cache_dir)

    @staticmethod
    def clear_cache(cache_dir):
        if cache_dir is not None and os.path.exists(cache_dir):
            print(f"Clearing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return data["image"], data["label"], data["group_index"]  # Return the group index





def get_loaders(config):

    data_list = extract_image_segmentation_datalist(config.DATA.TRAIN_CSV_PATH, config.DATA.DATA_PERCENTAGE)


    # Initialize the dataset
    # Initialize the dataset and split into training/validation
    if config.DATA.GROUPING == "local":
        groups = grouping()
    else:
        groups = grouping_total()
    
    transformation = image_label_transform_total(groups, (config.VIT.INPUT_SIZE,config.VIT.INPUT_SIZE,config.VIT.INPUT_SIZE ))

    dataset = ImageDataset(data_list=data_list, cache_dir=config.DATA.CACHE_DIR, transform=transformation)


    train_size = int(config.DATA.TRAINING_PERCENTAGE * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    #sanity_visualization(train_dataset)

    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=utils.get_world_size(),
                                                        rank=utils.get_rank(), shuffle=True)
    
    train_loader = torch.utils.data.DataLoader(
    train_dataset, sampler=train_sampler, batch_size=config.TRAIN.BATCH_SIZE,
    num_workers=config.DATA.NUM_WORKERS, pin_memory=False, drop_last=True
    )

    val_sampler = torch.utils.data.DistributedSampler(val_dataset, num_replicas=utils.get_world_size(),
                                                        rank=utils.get_rank(), shuffle=False)
    
    val_loader = torch.utils.data.DataLoader(
    val_dataset, sampler=val_sampler, batch_size=config.TRAIN.BATCH_SIZE,
    num_workers=config.DATA.NUM_WORKERS, pin_memory=False, drop_last=True
    )

    return train_loader, val_loader


# -----------------------------------------------------------------------------
# MAE-only loaders -- stage-1 pretraining (pretraining/mae_train.py). No segmentation labels or
# region groups are needed here, unlike the loaders above, so this is a separate, simpler path
# rather than reusing get_loaders() with the mask/group machinery just discarded.
# -----------------------------------------------------------------------------

def extract_image_datalist(csv_path, dataset_percentage, img_path_col="img_path_T1_mni152",
                            valid_col="is_img_path_T1_mni152"):
    """Same CSV convention as extract_image_segmentation_datalist(), minus the label column."""
    full_data = pd.read_csv(csv_path)
    num_samples = int(len(full_data) * dataset_percentage)
    data = full_data.sample(n=num_samples, random_state=42)

    return [
        {"image": row[img_path_col]}
        for _, row in data.iterrows()
        if row[valid_col]
    ]


def image_only_transform(vit_img_size):
    """Stage-1 MAE augmentation: random-crop + intensity scaling/shift + flips, no labels."""
    return transforms.Compose([
        transforms.LoadImaged(keys=["image"], reader="NibabelReader", image_only=True, allow_missing_keys=False),
        transforms.EnsureChannelFirstd(keys=["image"], allow_missing_keys=False),

        transforms.RandSpatialCropd(
            keys=["image"], roi_size=vit_img_size, random_center=True, random_size=False,
            allow_missing_keys=True,
        ),

        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True,
            allow_missing_keys=True,
        ),

        transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=0, allow_missing_keys=True),
        transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=1, allow_missing_keys=True),
        transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=2, allow_missing_keys=True),
        # The one augmentation stage 1 uses that stage 2's image_label_transform_total() above
        # doesn't -- paper Methods describes stage-1 augmentation as "random flip + intensity
        # shift" specifically. Exact offset isn't stated; 0.1 is a modest, commonly-used default.
        transforms.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5, allow_missing_keys=True),

        transforms.ToTensord(keys=["image"], allow_missing_keys=True),
    ])


class MAEImageDataset:
    """Image-only dataset for stage-1 MAE pretraining -- no segmentation labels/groups."""

    def __init__(self, data_list, cache_dir, transform):
        ImageDataset.clear_cache(cache_dir)
        self.dataset = PersistentDataset(data=data_list, transform=transform, cache_dir=cache_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]["image"]


def get_mae_loaders(config):
    """Single-process loaders (no DistributedSampler) for stage-1 MAE pretraining."""
    data_list = extract_image_datalist(config.DATA.TRAIN_CSV_PATH, config.DATA.DATA_PERCENTAGE)
    transform = image_only_transform((config.VIT.INPUT_SIZE,) * 3)
    dataset = MAEImageDataset(data_list=data_list, cache_dir=config.DATA.CACHE_DIR, transform=transform)

    train_size = int(config.DATA.TRAINING_PERCENTAGE * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True,
        num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False,
        num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY, drop_last=True,
    )
    return train_loader, val_loader

