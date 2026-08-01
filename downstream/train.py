"""CLI entry point: full or partial fine-tuning of an ASP-pretrained ViT on a downstream task.

Only handles training + checkpoint selection (best validation loss). It does not report final
metrics -- see evaluate.py for that (a separate script, on purpose: this loop's own validation
metrics exist only to pick a checkpoint, not to be the reported numbers).

Usage:
    python -m downstream.train --cfg downstream/configs/finetune/ad_diagnosis_train.yaml
"""
from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch

from downstream.config import load_config
from downstream.data import build_loader
from downstream.engine import (
    build_criterion,
    build_metrics,
    build_optimizer,
    cosine_schedule_with_warmup,
    evaluate,
    safe_compute,
    scale_lr_for_batch_size,
    train_one_epoch,
)
from downstream.model import build_model, freeze_all_but_last_block, freeze_backbone, load_asp_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", required=True, help="path to a finetune YAML config")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a config field, e.g. --set train.max_epochs=5",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config = load_config(args.cfg, args.overrides)
    set_seed(config.seed)
    scale_lr_for_batch_size(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{config.task_name}] device={device}")
    print(f"Effective base_lr={config.train.base_lr:.2e}  min_lr={config.train.min_lr:.2e}  (batch_size={config.data.batch_size})")

    encoder, head = build_model(config)
    if config.model.pretrained:
        result = load_asp_checkpoint(encoder, config.model.pretrained)
        print(f"Loaded ASP checkpoint {config.model.pretrained}: {result}")
    encoder, head = encoder.to(device), head.to(device)

    if config.model.lock_backbone:
        freeze_backbone(encoder)
    elif config.model.lock_last_block_only:
        freeze_all_but_last_block(encoder)

    criterion = build_criterion(config)
    metrics = build_metrics(config, device)

    train_loader = build_loader(config, config.data.train_csv, "train")
    val_loader = build_loader(config, config.data.val_csv, "val")

    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    head_params = [p for p in head.parameters() if p.requires_grad]
    optimizer = build_optimizer(config, encoder_params, head_params)
    total_steps = len(train_loader) * config.train.max_epochs
    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=int(config.train.warmup_ratio * total_steps),
        total_steps=total_steps,
        min_lr_ratio=config.train.min_lr / config.train.base_lr,
    )

    best_val_loss = float("inf")
    best_state = None
    for epoch in range(config.train.max_epochs):
        train_loss = train_one_epoch(
            encoder, head, train_loader, optimizer, scheduler, criterion, metrics, config, device
        )
        print(f"epoch {epoch + 1}/{config.train.max_epochs}  train_loss={train_loss:.4f}  {safe_compute(metrics)}")

        is_last_epoch = epoch == config.train.max_epochs - 1
        if (epoch + 1) % config.train.val_every == 0 or is_last_epoch:
            val_loss = evaluate(encoder, head, val_loader, criterion, metrics, config, device)
            print(f"  val_loss={val_loss:.4f}  {safe_compute(metrics)}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "encoder": copy.deepcopy(encoder.state_dict()),
                    "head": copy.deepcopy(head.state_dict()),
                }

    if best_state is None:
        raise RuntimeError("no validation pass ran -- check train.max_epochs and train.val_every")

    save_path = Path(config.model.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, save_path)
    print(f"Saved best checkpoint (val_loss={best_val_loss:.4f}) to {save_path}")
    print(f"Run evaluate.py against this checkpoint for the final reported metrics.")


if __name__ == "__main__":
    main()
