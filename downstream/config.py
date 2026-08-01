"""Typed configuration for the ASP downstream pipeline.

Replaces the old repo's YACS `CfgNode` with plain dataclasses. A config is loaded from a YAML
file and can be tweaked from the command line with repeated `--set section.field=value` flags,
e.g. `--set train.max_epochs=5 --set data.batch_size=4`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    train_csv: str = ""
    val_csv: str = ""
    test_csv: str = ""
    img_path_col: str = "img_path_T1_mni152"
    label_col: str = "pred_label"
    batch_size: int = 8
    num_workers: int = 4
    # evaluate.py only: break results down by this column's unique values in addition to the
    # overall metric (e.g. "source_dataset" -> per-cohort AD-diagnosis AUCs). Empty = no breakdown.
    cohort_col: str = ""

    # --- linear_probe.py only, below ---
    # Some linear-probe tasks (Parkinson's, epilepsy) don't have their own image folder -- they
    # reuse embeddings extracted once from a shared pool of scans, matched to each task's
    # train/test CSV by `join_col` (e.g. "subject_session"). Leave both blank for a
    # self-contained task (e.g. glioma) whose train/test CSVs carry `img_path_col` directly.
    pool_csv: str = ""
    join_col: str = ""
    # Average predictions over this column before scoring (e.g. a patient ID, when a test CSV
    # has multiple scans per subject -- used for glioma/UCSF-PDGM).
    group_col: str = ""
    # Map a categorical label column to 0/1 before fitting (e.g. epilepsy's {"hc": 0, "fcd": 1,
    # ...} group column). Leave empty if label_col is already numeric.
    label_map: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    # Path to a checkpoint to initialize from.
    # - train.py / linear_probe.py: the raw external ASP-pretraining checkpoint (model_mae
    #   sub-key, DDP module. prefix) -- see model.py::load_asp_checkpoint.
    # - evaluate.py: a checkpoint train.py produced (encoder + head) -- see
    #   model.py::load_finetuned_checkpoint.
    pretrained: str = ""
    save_path: str = "checkpoints/model.pt"
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    patch_size: int = 16
    # 1 => regression head (MSELoss), 2 => binary classification head (CrossEntropyLoss)
    num_classes: int = 2
    # Freeze the whole encoder, only train the head (used together with linear_probe.py).
    lock_backbone: bool = False
    # Freeze everything except the last transformer block + head ("partial fine-tuning" in the
    # paper, used for the epilepsy task).
    lock_last_block_only: bool = False


@dataclass
class TrainConfig:
    max_epochs: int = 100
    val_every: int = 2
    base_lr: float = 1.5e-4
    min_lr: float = 1.5e-8
    weight_decay: float = 0.005
    warmup_ratio: float = 0.01
    grad_clip: float = 3.0
    # Alzheimer's label smoothing: healthy=0, AD=1, MCI=0.5 -> soft 2-way target.
    # See paper Methods, "Linear Models with Anatomical Features".
    smooth_mci_label: bool = False


@dataclass
class Config:
    task_name: str = "task"
    seed: int = 42
    roi: tuple = (128, 128, 128)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _merge_dict_into_dataclass(obj: Any, updates: dict) -> None:
    """Recursively overwrite dataclass fields in-place from a nested dict."""
    for key, value in updates.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config field: {key!r}")
        current = getattr(obj, key)
        if isinstance(value, dict) and is_dataclass(current):
            _merge_dict_into_dataclass(current, value)
        else:
            setattr(obj, key, value)


def _set_dotted(obj: Any, dotted_key: str, raw_value: str) -> None:
    """Apply a single `--set section.field=value` override, with basic type coercion."""
    *path, leaf = dotted_key.split(".")
    target = obj
    for part in path:
        target = getattr(target, part)

    current = getattr(target, leaf)
    value: Any = raw_value
    if isinstance(current, bool):
        value = raw_value.lower() in ("1", "true", "yes")
    elif isinstance(current, int):
        value = int(raw_value)
    elif isinstance(current, float):
        value = float(raw_value)
    setattr(target, leaf, value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config file into a `Config`, then apply `key=value` CLI overrides."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    config = Config()
    _merge_dict_into_dataclass(config, raw)

    for item in overrides or []:
        key, _, value = item.partition("=")
        _set_dotted(config, key.strip(), value.strip())

    return config
