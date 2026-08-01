"""
TRAIN_MAE_SEG_FIXED.PY
======================
End‑to‑end script that alternates one epoch of segmentation training with one
epoch of MAE pre‑training. The original version contained a handful of syntax
errors, undefined names, device/AMP issues, and DDP misuse.  All those issues
are fixed here.  **Every place that diverges from the original is marked with
`# ---- CHANGED ----` so you can quickly diff.**
"""

# -----------------------------------------------------------------------------
# Imports & basic setup
# -----------------------------------------------------------------------------
import os
import argparse
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader

import wandb
import matplotlib.pyplot as plt
import pandas as pd

from monai.config import print_config

from src.data.dataloaders import grouping, grouping_total, get_loaders
from src.data.config import get_config
from src.data.logger import create_logger
from src.utils.misc import MetricLogger, all_reduce_mean
from src.utils.optimizers import get_optimizer
from src.utils.lr_sched import get_scheduler
from src.models.model import ConvDecoder, PromptSegmentationModel
from src.models.vit import ViT
from src.models.mae import MaskedAutoencoderViT
from src.utils.pos_embed import interpolate_pos_embed
from src.utils import utils_seg as utils

print_config()

# The 39-region lookup table ships with the repo at pretraining/files/ -- resolved relative
# to this file so it works regardless of the caller's working directory.
# ---- CHANGED: was a hardcoded absolute path to the original cluster.
_FILES_DIR = Path(__file__).resolve().parent / "files"

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_option():
    parser = argparse.ArgumentParser("Segmentation + MAE training", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE",
                        help="Path to config YAML")
    parser.add_argument("--local_rank", type=int, required=True,
                        help="DDP local rank")
    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--use_amp", action="store_true")
    args = parser.parse_args()

    config = get_config(args.cfg)
    return config, args


def create_region_bar_plot(region_losses_avg, epoch=None):
    """
    Loads the region mapping CSV, creates a bar plot of region losses, 
    and returns a matplotlib Figure.

    Args:
        region_losses_avg (dict): {region_id: average_loss, ...}
        csv_path (str): path to selected_brain_regions.csv
        epoch (int, optional): if provided, used in the plot title

    Returns:
        matplotlib.figure.Figure: A matplotlib Figure object
    """

    csv_path = _FILES_DIR / "selected_brain_regions.csv"
    # 1. Load region mapping from CSV
    region_df = pd.read_csv(csv_path)
    # Example CSV columns: "Normalized_ID", "Normalized_Label"
    region_map = dict(zip(region_df.index, region_df["Normalized_Label"]))

    # 2. Sort your region IDs so the bars appear in a consistent order
    regions = sorted(region_losses_avg.keys())
    losses = [region_losses_avg[r] for r in regions]

    # 3. Map IDs to labels (fallback to "Region X" if missing)
    labels = [region_map.get(r, f"Region {r}") for r in regions]

    # 4. Create the bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, losses)
    plt.xticks(rotation=45, ha="right")

    # Optional: add epoch number in title
    title = "Validation Loss by Region"
    if epoch is not None:
        title += f" (Epoch {epoch})"
    ax.set_title(title)
    ax.set_ylabel("Loss")

    plt.tight_layout()  # reduce label clipping
    return fig



# Function to visualize a few slices from the 3D tensor
def visualize_slices(tensor, title):
    if len(tensor.shape) == 3:  # If the tensor is 3D (H, W, D)
        tensor = tensor.unsqueeze(0)
    fig, axes = plt.subplots(1, 10, figsize=(15, 3))
    fig.suptitle(title)
    for i, ax in enumerate(axes):
        # Select slices from different depths
        slice_idx = tensor.shape[3] // 10 * i  # Select along depth
        ax.imshow(tensor[0, :, :, slice_idx].cpu().numpy(), cmap="gray")
        ax.axis("off")
    plt.show()




def create_image_grid(volume, title):
    """
    Create an image grid with 5 evenly spaced slices from a 3D volume.

    Args:
        volume (numpy array): The 3D volume (H, W, D).
        title (str): Title for the grid.

    Returns:
        fig (matplotlib figure): A figure containing the image grid.
    """
    if len(volume.shape) == 4:  # If the tensor is 3D (H, W, D)
        volume = volume.squeeze()
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i, ax in enumerate(axes):
        slice_idx = volume.shape[2] // 5 * i  # Select evenly spaced slices
        ax.imshow(volume[:, :, slice_idx], cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    #plt.show()

    return fig

def freeze_encoder(model):
    """
    Disables gradient updates for the encoder parameters.
    """
    for param in model.module.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model):
    """
    Enables gradient updates for the encoder parameters.
    """
    for param in model.module.encoder.parameters():
        param.requires_grad = True




def val_one_epoch_seg(
    config,
    model,
    loader,
    epoch,
    max_epoch,
    logger=None,
    device=None,
    use_amp=False,
    scaler=None,
    wandb_run=None
):
    """
    Validate the model on the validation set.
    Now we also compute and plot IoU by region instead of plotting BCE loss.
    """

    # If you still want BCE for logging, keep it, but for IoU we won't use it for the bar plot
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    model.eval()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)

    # Select a random batch index for logging sample images
    random_batch_idx = random.randint(0, len(loader) - 1)
    model = model.to(device, non_blocking=True)  

    # For BCE loss stats (if you still want them):
    region_loss_sums = defaultdict(float)
    region_loss_counts = defaultdict(int)

    # For IoU stats:
    region_iou_sums = defaultdict(float)
    region_iou_counts = defaultdict(int)

    with torch.no_grad():
        for idx, (images, masks, group_indices) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            group_indices = group_indices.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images, group_indices)

            # 1) Per-sample BCE loss (optional)
            bce_per_sample = criterion(outputs, masks)  # (B, C, H, W, D)
            bce_per_sample = bce_per_sample.mean(dim=[1,2,3,4])  # (B,)

            # 2) Compute IoU per sample
            #    threshold predictions
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()  # (B, C, H, W, D)

            intersection = (pred_masks * masks).sum(dim=[2,3,4])  # (B, C)
            union = pred_masks.sum(dim=[2,3,4]) + masks.sum(dim=[2,3,4]) - intersection  # (B, C)
            iou_vals = intersection / (union + 1e-7)  # (B, C)
            iou_vals = iou_vals.mean(dim=1)  # average across channels if C>1 => shape (B,)

            torch.cuda.synchronize()

            # 3) Accumulate into region-based sums
            for i in range(len(bce_per_sample)):
                region_idx = group_indices[i].item()

                # BCE loss accumulation
                region_loss_sums[region_idx] += bce_per_sample[i].item()
                region_loss_counts[region_idx] += 1

                # IoU accumulation
                region_iou_sums[region_idx] += iou_vals[i].item()
                region_iou_counts[region_idx] += 1

            # 4) Log an example image from the random batch
            if idx == random_batch_idx:
                group_idx = group_indices[0].item()
                if dist.get_rank() == 0:
                    logger.info(f"Logging validation image from batch {random_batch_idx}, region {group_idx}")

                # Just log the first item in that batch
                input_image = images[0].cpu().numpy().squeeze(0)
                ground_truth_mask = masks[0].cpu().numpy().squeeze(0)
                reconstructed_mask = torch.sigmoid(outputs[0]).cpu().numpy().squeeze(0)

                # create_image_grid is presumably your custom function
                original_fig = create_image_grid(input_image, f"Input Image")
                gt_fig = create_image_grid(ground_truth_mask, f"Ground Truth Mask - Region {group_idx}")
                recon_fig = create_image_grid(reconstructed_mask, f"Reconstructed Mask - Region {group_idx}")

                if wandb_run is not None and dist.get_rank() == 0:
                    wandb_run.log({
                        "epoch": epoch,
                        "original_image_grid": wandb.Image(original_fig, caption=f"Epoch {epoch}"),
                        "ground_truth_grid": wandb.Image(gt_fig, caption=f"Epoch {epoch} - Region {group_idx}"),
                        "validation_reconstruction_grid": wandb.Image(recon_fig, caption=f"Epoch {epoch} - Region {group_idx}"),
                    })

                plt.close(original_fig)
                plt.close(gt_fig)
                plt.close(recon_fig)

            # 5) For your overall scalar logging (e.g., average BCE for the entire batch)
            #    This is optional if you only care about IoU from now on.
            local_loss = bce_per_sample.mean()
            loss_value = all_reduce_mean(local_loss)
            metric_logger.update(loss=loss_value)

            if dist.get_rank() == 0:
                logger.info(f"Epoch {epoch+1}/{max_epoch} [{idx+1}/{len(loader)}], BCE Loss: {loss_value:.4f}")

        # 6) After the loop, compute region-wise averages
        region_losses_avg = {
            region: region_loss_sums[region] / region_loss_counts[region]
            for region in region_loss_sums
        }
        region_ious_avg = {
            region: region_iou_sums[region] / region_iou_counts[region]
            for region in region_iou_sums
        }

        # 7) [Optional] Log the region-wise BCE loss (if you still want it)
        # wandb_run.log({ "epoch": epoch, "region_loss": region_losses_avg })

        # 8) Create a bar plot for IoU, not BCE
        if wandb_run is not None and dist.get_rank() == 0:
            fig_iou = create_region_bar_plot(region_ious_avg, epoch)
            wandb_run.log({
                "epoch": epoch,
                "val_iou_by_region_barplot": wandb.Image(fig_iou)
            })
            plt.close(fig_iou)

        metric_logger.synchronize_between_processes()
        if dist.get_rank() == 0:
            logger.info(f"Averaged stats: {metric_logger}")

    return {   "metrics": {k: meter.global_avg for k, meter in metric_logger.meters.items()},   "region_ious": region_ious_avg,}



# -----------------------------------------------------------------------------
# Utility: weight transfer between MAE ↔ SEG
# -----------------------------------------------------------------------------


def update_model_weights(models: Dict[str, nn.Module], to_update_model: str, logger):
    """
    Copy encoder/decoder backbone weights between the MAE and SEG networks
    without touching any other parameters.

    Parameters
    ----------
    models : dict
        Dictionary holding the two PyTorch models, e.g. {"mae": mae_net, "seg": seg_net}.
    to_update_model : str
        Which model to update: "seg" (copy MAE → SEG) or "mae" (copy SEG → MAE).
    logger : logging.Logger
        Logger where missing / unexpected keys will be reported.
    """
    if to_update_model not in {"seg", "mae"}:
        raise ValueError('`to_update_model` must be either "seg" or "mae"')

    source_key, target_key = ("mae", "seg") if to_update_model == "seg" else ("seg", "mae")
    src_sd   = models[source_key].state_dict()
    tgt_sd   = models[target_key].state_dict()
    patched  = {}

    # Build a remapped state-dict containing only the overlapping weights
    for k, v in src_sd.items():
        if to_update_model == "seg":                # MAE → SEG
            if k.startswith("module."):
                new_k = "module.encoder." + k[len("module."):]
                if new_k in tgt_sd:
                    patched[new_k] = v
        else:                                       # SEG → MAE
            if k.startswith("module.encoder."):
                new_k = "module." + k[len("module.encoder."):]
                if new_k in tgt_sd:
                    patched[new_k] = v

    # Load the subset (strict=False avoids touching the rest)
    incompatible = models[target_key].load_state_dict(patched, strict=False)

    # Logging
    logger.info(f"Copied {len(patched):>4} tensors  ({source_key} ➜ {target_key})")
    if incompatible.missing_keys:
        logger.info(f"[{target_key}] missing keys:     {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        logger.info(f"[{target_key}] unexpected keys:  {incompatible.unexpected_keys}")

# -----------------------------------------------------------------------------
# MAE helpers (training & validation)
# -----------------------------------------------------------------------------

def train_one_epoch_mae(config, model, loader, optimizer, lr_schedule, epoch,
                        max_epoch, logger, device, use_amp, scaler, wandb_run):
    model.train()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)

    for idx, (images, _, _) in enumerate(loader):


        global_it = epoch * len(loader) + idx              # ---- CHANGED ----
        lr = lr_schedule[global_it]                             # ---- CHANGED ----
        for pg in optimizer.param_groups:                       # ---- CHANGED ----
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
        if dist.get_rank() == 0 and wandb_run is not None:
            wandb_run.log({"mae_train_loss": float(loss_val)}, step=global_it)

    metric_logger.synchronize_between_processes()
    logger.info(f"MAE train epoch {epoch+1}: {metric_logger}")
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


def val_one_epoch_mae(config, model, loader, epoch, max_epoch, logger, device,
                      use_amp):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, _, _ = model(images)
            metric_logger.update(loss=all_reduce_mean(loss))
    metric_logger.synchronize_between_processes()
    logger.info(f"MAE val epoch {epoch+1}: {metric_logger}")
    return {k: m.global_avg for k, m in metric_logger.meters.items()}

# -----------------------------------------------------------------------------
# Segmentation helpers (training & validation)
# -----------------------------------------------------------------------------

def train_one_epoch_seg(config, model, loader, optimizer, scheduler_list, criterion,
                        epoch, max_epoch, logger, device, use_amp, scaler, wandb_run):
    model.train()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    for it, (images, masks, group_idx) in enumerate(loader):
        global_it = epoch * len(loader) + it
        lr = scheduler_list[global_it]
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        images, masks, group_idx = (t.to(device, non_blocking=True) for t in (images, masks, group_idx))
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(images, group_idx)
            loss = criterion(out, masks)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if config.TRAIN.GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.GRAD_CLIP)
        scaler.step(optimizer) 
        scaler.update()

        metric_logger.update(loss=all_reduce_mean(loss), lr=lr)
        if dist.get_rank() == 0 and wandb_run is not None:
            wandb_run.log({"seg_train_loss": float(loss)}, step=global_it)

    metric_logger.synchronize_between_processes()
    logger.info(f"SEG train epoch {epoch+1}: {metric_logger}")
    return {k: m.global_avg for k, m in metric_logger.meters.items()}

# (val_one_epoch_seg remains identical to fixed version earlier — omitted for brevity)

# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------

def build_mae(config, logger):
    mae_model = MaskedAutoencoderViT(
        input_size=config.VIT.INPUT_SIZE,
        patch_size=config.VIT.PATCH_SIZE,
        mask_ratio=config.MAE.MASK_RATIO,  # ---- CHANGED ---- added comma
        in_chans=config.VIT.IN_CHANS,
        dropout_rate=config.VIT.DROPOUT_RATE,
        spatial_dims=config.VIT.SPATIAL_DIMS,
        patch_embed=config.VIT.PATCH_EMBED,
        pos_embed=config.VIT.POS_EMBED,
        use_bias=config.VIT.USE_BIAS,
        use_flash_attn=config.VIT.USE_FLASH_ATTN,
    )

    if config.MODEL.MAE_PRETRAINED:  # ---- CHANGED ---- fixed namespace
        sd = torch.load(config.MODEL.MAE_PRETRAINED, map_location="cpu")["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        interpolate_pos_embed(mae_model, sd)
        msg = mae_model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded SSL ViT: {msg}")
    return mae_model
    



def build_seg(config, logger):
    vit_enc = ViT(
        img_size=config.VIT.INPUT_SIZE,
        patch_size=config.VIT.PATCH_SIZE,
        hidden_size=config.VIT.HIDDEN_SIZE,
        mlp_dim=config.VIT.MLP_DIM,
        num_layers=config.VIT.NUM_LAYERS,
        num_heads=config.VIT.NUM_HEADS,
        in_chans=config.VIT.IN_CHANS,
        dropout_rate=config.VIT.DROPOUT_RATE,
        spatial_dims=config.VIT.SPATIAL_DIMS,
        patch_embed=config.VIT.PATCH_EMBED,
        pos_embed=config.VIT.POS_EMBED,
        #classification=config.VIT.CLASSIFICATION,
        #num_classes=config.VIT.NUM_CLASSES,
        post_activation="Tanh",
        qkv_bias=config.VIT.USE_BIAS,
        use_flash_attn=config.VIT.USE_FLASH_ATTN,
        #pooling=config.VIT.POOLING,
    )

    if config.MODEL.SSL_PRETRAINED:  # ---- CHANGED ---- fixed namespace
        sd = torch.load(config.MODEL.SSL_PRETRAINED, map_location="cpu")["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        interpolate_pos_embed(vit_enc, sd)
        msg = vit_enc.load_state_dict(sd, strict=False)
        logger.info(f"Loaded SSL ViT: {msg}")

    groups = grouping() if config.DATA.GROUPING == "local" else grouping_total()
    decoder = ConvDecoder(num_regions=len(groups))
    return PromptSegmentationModel(encoder=vit_enc, decoder=decoder)

# -----------------------------------------------------------------------------
# Training orchestrator
# -----------------------------------------------------------------------------

def train_loop(config, models, optims, scheduler, criterion, device, logger, wandb_run):
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)

    max_epochs = config.TRAIN.MAX_EPOCHS
    val_every = config.TRAIN.VAL_EVERY

    val_losses_mae = []

    val_losses_seg = []

    start_epoch = 0
    resume_path = os.path.join(config.MODEL.CHECKPOINT_DIR, "checkpoint.pth")
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(logger,
        os.path.join(config.MODEL.CHECKPOINT_DIR, "checkpoint.pth"),
        run_variables=to_restore,
        model_seg= models["seg"],
        model_mae= models["mae"],
        optim_seg= optims["seg"],
        optim_mae= optims["mae"],
    )
    start_epoch = to_restore["epoch"]

    for epoch in range(start_epoch,max_epochs):
        tr_loader, val_loader = get_loaders(config)

        epoch_time = time.time()

        # SEG epoch ------------------------------------------------------------
        train_stats = train_one_epoch_seg(config, models["seg"], tr_loader, optims["seg"],
                            scheduler, criterion, epoch, max_epochs, logger,
                            device, config.USE_AMP, scaler, wandb_run)

        logger.info(
            f"Final training  for SEGMENTATION {epoch+1}/{max_epochs}, loss: {train_stats['loss']}, \
                time {time.time() - epoch_time}s"
        )

        update_model_weights(models, "mae", logger)

        epoch_time = time.time()

        # MAE epoch ------------------------------------------------------------
        train_stats=train_one_epoch_mae(config, models["mae"], tr_loader, optims["mae"],
                            scheduler, epoch, max_epochs, logger, device,
                            config.USE_AMP, scaler, wandb_run)

        logger.info(
            f"Final training for MAE  {epoch+1}/{max_epochs}, loss: {train_stats['loss']}, \
                time {time.time() - epoch_time}s"
        )

        # update_model_weights(models, "seg", logger)

        # -------- Validation --------------------------------------------------
        if (epoch + 1) % val_every == 0:

            epoch_time = time.time()
            val_stats = val_one_epoch_seg(config, models["seg"], val_loader, epoch, max_epochs,
                               logger, device, config.USE_AMP, wandb_run = wandb_run)

            logger.info(
                f"Final validation for SEGMENTATION {epoch+1}/{max_epochs} "
                f"loss: {val_stats['metrics'].get('loss', 'N/A'):.4f}, time {time.time() - epoch_time:.1f}s"
            )
            
            if wandb_run != None and dist.get_rank() == 0:
                wandb_run.log({'Validation Loss for SEGMENTATION': float(val_stats['metrics']['loss'])})


            epoch_time = time.time()

            val_stats = val_one_epoch_mae(config, models["mae"], val_loader, epoch, max_epochs,
                               logger, device, config.USE_AMP)

            logger.info(
                f"Final validation for MAE {epoch+1}/{max_epochs} \
                    loss: {val_stats['loss']}, time {time.time() - epoch_time}s"
            )
            
            if wandb_run != None and dist.get_rank() == 0:
                wandb_run.log({'Validation Loss for MAE': (float)(val_stats['loss'])})

            # -------- Checkpoint --------------------------------------------------
            if dist.get_rank() == 0:
                save_dict = {
                "epoch": epoch + 1,
                "model_seg": models["seg"].state_dict(),
                "model_mae": models["mae"].state_dict(),
                "optim_seg": optims["seg"].state_dict(),
                "optim_mae": optims["mae"].state_dict(),
                }

                utils.save_on_master(save_dict, os.path.join(config.MODEL.CHECKPOINT_DIR, 'checkpoint.pth'))
                if config.TRAIN.SAVECKP_FREQ and epoch % config.TRAIN.SAVECKP_FREQ == 0:
                    utils.save_on_master(save_dict, os.path.join(config.MODEL.CHECKPOINT_DIR, f'checkpoint_seg_{epoch:04}.pth'))

            update_model_weights(models, "seg", logger)

# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main():
    config, args = parse_option()

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(device)

    logger = create_logger(output_dir="logs", dist_rank=dist.get_rank(),
                           name=config.LOG.FILENAME)

    if dist.get_rank() == 0:
        os.makedirs(config.MODEL.CHECKPOINT_DIR, exist_ok=True)
        with open(os.path.join(config.MODEL.CHECKPOINT_DIR, "config_dump.yaml"), "w") as f:
            f.write(config.dump())

    wandb_run = None
    if config.WANDB.WANDB_ENABLE and dist.get_rank() == 0:
        wandb_run = wandb.init(name=config.WANDB.RUN,
                               project=config.WANDB.PROJECT,
                               config={"lr": config.TRAIN.BASE_LR,
                                       "epochs": config.TRAIN.MAX_EPOCHS})

    # ---------------- Model instantiation -----------------------------------
    seg_net = build_seg(config, logger).to(device)
    mae_net = build_mae(config,logger).to(device)

    # DDP ---------------------------------------------------------------------
    seg_net = torch.nn.parallel.DistributedDataParallel(seg_net, device_ids=[device], broadcast_buffers=False)
    mae_net = torch.nn.parallel.DistributedDataParallel(mae_net, device_ids=[device], broadcast_buffers=False)

    models = {"seg": seg_net, "mae": mae_net}

    # Optimizers & schedulers -------------------------------------------------
    world_size = dist.get_world_size()
    eff_bs = config.TRAIN.BATCH_SIZE * world_size
    base_lr = config.TRAIN.BASE_LR * eff_bs / 256
    min_lr = base_lr * 1e-3

    for k in ("BASE_LR", "MIN_LR"):
        config.defrost(); setattr(config.TRAIN, k, locals()[k.lower()]); config.freeze()  # ---- CHANGED ---- update cfg

    optim_seg = get_optimizer(config, base_lr, [seg_net])
    optim_mae = get_optimizer(config, base_lr, [mae_net])

    steps_per_epoch = len(get_loaders(config)[0])  # one call just to know length
    scheduler = utils.cosine_scheduler(base_lr, min_lr, config.TRAIN.MAX_EPOCHS,
                                       steps_per_epoch, config.TRAIN.WARMUP_EPOCHS)
    

    criterion = nn.BCEWithLogitsLoss()

    train_loop(config, models,
               {"seg": optim_seg, "mae": optim_mae},
               scheduler,
               criterion, device, logger, wandb_run)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
