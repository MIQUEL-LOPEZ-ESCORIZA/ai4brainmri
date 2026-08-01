"""
MAE_TRAIN.PY
============
Stage-1 pretraining: masked-autoencoder training of the ViT encoder that stage 2 (ASP, see
seg_mae_combined_train.py) starts from. Unlike that script, there's no original implementation
to port here -- this stage's real training code lives in an inaccessible external cluster repo
(see pretraining/README.md) -- so this is new code, written to mirror the shape of
seg_mae_combined_train.py's MAE half (build_mae, train_one_epoch_mae, val_one_epoch_mae) but
simplified: single-process, no DDP/torchrun, since (unlike stage 2) there's no original
multi-GPU implementation this needs to stay faithful to.
"""

import os
import argparse
import time

import torch
import wandb

from monai.config import print_config

from src.data.dataloaders import get_mae_loaders
from src.data.config import get_config
from src.data.logger import create_logger
from src.utils.misc import MetricLogger, all_reduce_mean
from src.utils.optimizers import get_optimizer
from src.models.mae import MaskedAutoencoderViT
from src.utils import utils_seg as utils

print_config()


def parse_option():
    parser = argparse.ArgumentParser("MAE pretraining (stage 1)", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="Path to config YAML")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    config = get_config(args.cfg)
    return config, args


def build_mae(config):
    return MaskedAutoencoderViT(
        input_size=config.VIT.INPUT_SIZE,
        patch_size=config.VIT.PATCH_SIZE,
        mask_ratio=config.MAE.MASK_RATIO,
        in_chans=config.VIT.IN_CHANS,
        dropout_rate=config.VIT.DROPOUT_RATE,
        spatial_dims=config.VIT.SPATIAL_DIMS,
        patch_embed=config.VIT.PATCH_EMBED,
        pos_embed=config.VIT.POS_EMBED,
        use_bias=config.VIT.USE_BIAS,
        use_flash_attn=config.VIT.USE_FLASH_ATTN,
    )


def train_one_epoch(config, model, loader, optimizer, lr_schedule, epoch, logger, device,
                     use_amp, scaler, wandb_run):
    model.train()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)

    for it, images in enumerate(loader):
        global_it = epoch * len(loader) + it
        lr = lr_schedule[global_it]
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        images = images.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, _, _ = model(images)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if config.TRAIN.GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        loss_val = all_reduce_mean(loss)
        metric_logger.update(loss=loss_val, lr=lr)
        if wandb_run is not None:
            wandb_run.log({"train_loss": float(loss_val)}, step=global_it)

    metric_logger.synchronize_between_processes()
    logger.info(f"Train epoch {epoch + 1}: {metric_logger}")
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


def val_one_epoch(model, loader, epoch, logger, device, use_amp):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    with torch.no_grad():
        for images in loader:
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, _, _ = model(images)
            metric_logger.update(loss=all_reduce_mean(loss))

    metric_logger.synchronize_between_processes()
    logger.info(f"Val epoch {epoch + 1}: {metric_logger}")
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


def save_checkpoint(config, model, optimizer, epoch):
    os.makedirs(config.MODEL.CHECKPOINT_DIR, exist_ok=True)

    # Resume file: plain (unprefixed) keys, loaded straight back into this same non-DDP
    # model/optimizer by restart_from_checkpoint() below.
    utils.save_on_master(
        {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
        os.path.join(config.MODEL.CHECKPOINT_DIR, "checkpoint.pth"),
    )

    if config.TRAIN.SAVECKP_FREQ and epoch % config.TRAIN.SAVECKP_FREQ == 0:
        # Distributable snapshot: consumed by seg_mae_combined_train.py's build_mae()/build_seg()
        # (MODEL.MAE_PRETRAINED / SSL_PRETRAINED) and by linear_probing/linear_probe_sweep.py's
        # --checkpoint_source mae. Both expect a top-level "state_dict" key; the sweep script
        # additionally requires "module."-prefixed parameter names (it was written against
        # DDP-trained checkpoints, where that prefix is automatic). Added by hand here even
        # though this script itself isn't distributed, purely so its checkpoints stay loadable
        # by that existing code unmodified.
        state_dict = {f"module.{k}": v for k, v in model.state_dict().items()}
        utils.save_on_master(
            {"epoch": epoch, "state_dict": state_dict},
            os.path.join(config.MODEL.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:04d}.pt"),
        )


def train_loop(config, model, optimizer, lr_schedule, train_loader, val_loader, device, logger,
                wandb_run):
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)
    max_epochs = config.TRAIN.MAX_EPOCHS
    val_every = config.TRAIN.VAL_EVERY

    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        logger, os.path.join(config.MODEL.CHECKPOINT_DIR, "checkpoint.pth"),
        run_variables=to_restore, model=model, optimizer=optimizer,
    )
    start_epoch = to_restore["epoch"]

    for epoch in range(start_epoch, max_epochs):
        epoch_time = time.time()
        train_stats = train_one_epoch(config, model, train_loader, optimizer, lr_schedule, epoch,
                                       logger, device, config.USE_AMP, scaler, wandb_run)
        logger.info(
            f"Epoch {epoch + 1}/{max_epochs} train loss {train_stats['loss']:.4f}, "
            f"time {time.time() - epoch_time:.1f}s"
        )

        if (epoch + 1) % val_every == 0:
            val_stats = val_one_epoch(model, val_loader, epoch, logger, device, config.USE_AMP)
            logger.info(f"Epoch {epoch + 1}/{max_epochs} val loss {val_stats['loss']:.4f}")
            if wandb_run is not None:
                wandb_run.log({"val_loss": val_stats["loss"], "epoch": epoch + 1})

            save_checkpoint(config, model, optimizer, epoch + 1)


def main():
    config, args = parse_option()
    utils.fix_random_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=config.LOG.FILENAME)

    os.makedirs(config.MODEL.CHECKPOINT_DIR, exist_ok=True)
    with open(os.path.join(config.MODEL.CHECKPOINT_DIR, "config_dump.yaml"), "w") as f:
        f.write(config.dump())

    wandb_run = None
    if config.WANDB.WANDB_ENABLE:
        wandb_run = wandb.init(
            name=config.WANDB.RUN, project=config.WANDB.PROJECT,
            config={"lr": config.TRAIN.BASE_LR, "epochs": config.TRAIN.MAX_EPOCHS},
        )

    model = build_mae(config).to(device)
    optimizer = get_optimizer(config, config.TRAIN.BASE_LR, [model])
    train_loader, val_loader = get_mae_loaders(config)
    lr_schedule = utils.cosine_scheduler(
        config.TRAIN.BASE_LR, config.TRAIN.MIN_LR, config.TRAIN.MAX_EPOCHS,
        len(train_loader), config.TRAIN.WARMUP_EPOCHS,
    )

    train_loop(config, model, optimizer, lr_schedule, train_loader, val_loader, device, logger,
               wandb_run)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
