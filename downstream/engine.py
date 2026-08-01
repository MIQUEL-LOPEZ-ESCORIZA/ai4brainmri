"""Training and evaluation loop for the ASP downstream pipeline.

Single-GPU, task-agnostic over classification (`num_classes=2`, `CrossEntropyLoss`,
BinaryAccuracy/BinaryAUROC) vs regression (`num_classes=1`, `MSELoss`,
MeanAbsoluteError/MeanSquaredError), matching the loss/metric selection in the original repo's
`main_finetune_final.py` + `engine_finetune.py` for this pipeline (their `QuantileLoss`/
`GaussianNLLLoss`/multiclass branches are unused by every ASP downstream config and are dropped).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from downstream.config import Config


def build_criterion(config: Config) -> nn.Module:
    return nn.MSELoss() if config.model.num_classes == 1 else nn.CrossEntropyLoss()


def build_metrics(config: Config, device: torch.device) -> MetricCollection:
    if config.model.num_classes == 1:
        return MetricCollection([MeanAbsoluteError(), MeanSquaredError()]).to(device)
    return MetricCollection([BinaryAccuracy(), BinaryAUROC()]).to(device)


def safe_compute(metrics: MetricCollection) -> dict:
    """`metrics.compute()`, but returns `{}` instead of raising if no batch ever had a scorable
    sample. Can happen on a very small split combined with the MCI-exclusion filter below (an AD
    label-smoothed split with 0-1 CN/AD samples has nothing to score against)."""
    try:
        return metrics.compute()
    except ValueError:
        return {}


def scale_lr_for_batch_size(config: Config) -> None:
    """Linear LR scaling rule, exactly as `main_finetune_final.py` applies it right before
    building the optimizer: `TRAIN.BASE_LR` in the YAML is a reference value for an effective
    batch size of 256, scaled down for the actually configured (typically much smaller) batch
    size. `TRAIN.MIN_LR` is then *recomputed* as `scaled_base_lr * 1e-3` -- the YAML's own MIN_LR
    field is never used by the original code either. Mutates `config.train` in place; call once,
    right after loading the config (so it sees any `--set` overrides to `data.batch_size` or
    `train.base_lr`).
    """
    config.train.base_lr = config.train.base_lr * config.data.batch_size / 256
    config.train.min_lr = config.train.base_lr * 1e-3


def build_optimizer(config: Config, encoder_params, head_params) -> torch.optim.Optimizer:
    """The head is randomly initialized and the encoder is pretrained, so the head gets a 100x
    higher learning rate (same ratio as the original repo's separate encoder/classifier
    optimizers). Note: unlike the original, the cosine schedule's LR floor scales with this 100x
    factor too (both param groups share one `LambdaLR`) rather than sharing one fixed absolute
    floor — a deliberate simplification that only affects the very end of the LR schedule.
    """
    return torch.optim.AdamW(
        [
            {"params": list(encoder_params), "lr": config.train.base_lr},
            {"params": list(head_params), "lr": config.train.base_lr * 100},
        ],
        weight_decay=config.train.weight_decay,
    )


def cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _loss_target(raw_labels: torch.Tensor, config: Config) -> torch.Tensor:
    """Regression: float column vector. Classification: class indices, optionally smoothed for
    AD diagnosis (healthy=0, MCI=0.5, AD=1 -> a soft 2-way target instead of a hard class index —
    see paper Methods, "Linear Models with Anatomical Features")."""
    if config.model.num_classes == 1:
        return raw_labels.float().view(-1, 1)
    if config.train.smooth_mci_label:
        return torch.stack((1 - raw_labels / 2, raw_labels / 2), dim=1)
    return raw_labels.long()


def _update_metrics(metrics: MetricCollection, logits: torch.Tensor, raw_labels: torch.Tensor, config: Config) -> None:
    if config.model.num_classes == 1:
        metrics.update(logits, raw_labels.float().view(-1, 1))
        return
    probs_ad = torch.softmax(logits, dim=1)[:, 1]
    if config.train.smooth_mci_label:
        # MCI (label == 0.5) has no hard ground truth for a binary CN-vs-AD metric; only
        # cognitively-normal (0) and AD (1) samples are scored.
        keep = raw_labels != 0.5
        if keep.sum() > 1:
            metrics.update(probs_ad[keep], raw_labels[keep].int())
    else:
        metrics.update(probs_ad, raw_labels.int())


def _forward(encoder, head, batch, device):
    images = batch["image"].to(device)
    raw_labels = batch["pred_label"].to(device)
    logits = head(encoder(images))
    return logits, raw_labels


def train_one_epoch(encoder, head, loader, optimizer, scheduler, criterion, metrics, config, device) -> float:
    encoder.train()
    head.train()
    metrics.reset()
    total_loss = 0.0
    for batch in loader:
        logits, raw_labels = _forward(encoder, head, batch, device)
        loss = criterion(logits, _loss_target(raw_labels, config))

        optimizer.zero_grad()
        loss.backward()
        # Matches the original: only the encoder's gradients are clipped, not the head's.
        nn.utils.clip_grad_norm_(encoder.parameters(), config.train.grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * logits.shape[0]
        _update_metrics(metrics, logits.detach(), raw_labels.detach(), config)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(encoder, head, loader, criterion, metrics, config, device) -> float:
    encoder.eval()
    head.eval()
    metrics.reset()
    total_loss = 0.0
    for batch in loader:
        logits, raw_labels = _forward(encoder, head, batch, device)
        loss = criterion(logits, _loss_target(raw_labels, config))
        total_loss += loss.item() * logits.shape[0]
        _update_metrics(metrics, logits, raw_labels, config)

    return total_loss / len(loader.dataset)
