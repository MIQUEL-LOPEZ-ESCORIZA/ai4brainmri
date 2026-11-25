"""ViT encoder + task head for the ASP downstream pipeline.

The architecture and checkpoint-loading convention mirror the original repo's
`mae/src/models/vit.py` and the `'mae_seg_combined' in config.MODEL.SAVE_NAME` branch of
`main_finetune_final.py`, trimmed to the one configuration every ASP downstream task actually
used: conv patch embedding, sin-cos position embedding, CLS-token pooling, single 128^3 crop
(no multi-patch aggregation, no flash attention, no UNETR/segmentation decoder).
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers import Conv

from downstream.config import Config


def _build_sincos_position_embedding(grid_size: tuple, embed_dim: int, temperature: float = 10000.0) -> torch.Tensor:
    """3D sin-cos position embedding, one entry per patch (no CLS-token slot)."""
    assert embed_dim % 6 == 0, "embed_dim must be divisible by 6 for a 3D sin-cos embedding"
    h, w, d = grid_size
    grid_h, grid_w, grid_d = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        torch.arange(d, dtype=torch.float32),
        indexing="ij",
    )
    pos_dim = embed_dim // 6
    omega = 1.0 / (temperature ** (torch.arange(pos_dim, dtype=torch.float32) / pos_dim))
    out_h = torch.einsum("m,d->md", grid_h.flatten(), omega)
    out_w = torch.einsum("m,d->md", grid_w.flatten(), omega)
    out_d = torch.einsum("m,d->md", grid_d.flatten(), omega)
    return torch.cat(
        [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w), torch.sin(out_d), torch.cos(out_d)],
        dim=1,
    )[None, :, :]


class PatchEmbedding(nn.Module):
    """Conv patch embedding + fixed sin-cos position embedding, 3D."""

    def __init__(self, img_size: tuple, patch_size: int, hidden_size: int):
        super().__init__()
        self.grid_size = tuple(s // patch_size for s in img_size)
        self.n_patches = int(np.prod(self.grid_size))
        self.proj = Conv["conv", 3](1, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(
            _build_sincos_position_embedding(self.grid_size, hidden_size), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(-1, -2)  # (B, n_patches, hidden_size)
        return x + self.position_embeddings


class ViTEncoder(nn.Module):
    """3D Vision Transformer encoder. Returns the CLS-token embedding."""

    def __init__(self, config: Config):
        super().__init__()
        m = config.model
        self.patch_embedding = PatchEmbedding(config.roi, m.patch_size, m.hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, m.hidden_size))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(m.hidden_size, m.mlp_dim, m.num_heads, dropout_rate=0.0, qkv_bias=True)
                for _ in range(m.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        for block in self.blocks:
            x = block(x)
        return x[:, 0, :]

    def forward_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Patch tokens (CLS token dropped) after the last block, plus the patch tokens after every
        block -- the interface the UNETR decoder in `downstream/segmentation/` expects."""
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        hidden = []
        for block in self.blocks:
            x = block(x)
            hidden.append(x[:, 1:, :])
        return x[:, 1:, :], hidden


class Head(nn.Module):
    """LayerNorm -> Linear on the CLS embedding. `num_classes=1` makes this a regression head."""

    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


def build_model(config: Config) -> tuple[ViTEncoder, Head]:
    encoder = ViTEncoder(config)
    head = Head(config.model.hidden_size, config.model.num_classes)
    return encoder, head


def freeze_backbone(encoder: nn.Module) -> None:
    """Used for linear probing: only the head is trained."""
    for p in encoder.parameters():
        p.requires_grad = False


def freeze_all_but_last_block(encoder: ViTEncoder) -> None:
    """Paper's "partial fine-tuning": only the last transformer block (+ head) is trained."""
    last_block = f"blocks.{len(encoder.blocks) - 1}."
    for name, p in encoder.named_parameters():
        p.requires_grad = name.startswith(last_block)


def prepare_encoder(encoder: ViTEncoder, config: Config) -> str:
    """Apply the configured adaptation strategy (see `ModelConfig.adaptation`) to `encoder` in
    place, after the pretrained weights are loaded. Returns a short description for logging.

    Must also be called by `evaluate.py` *before* loading a LoRA checkpoint, so that the module
    tree contains the adapter parameters the checkpoint expects.
    """
    m = config.model
    adaptation = m.adaptation
    if m.lock_backbone:
        adaptation = "frozen"
    elif m.lock_last_block_only:
        adaptation = "partial"

    if adaptation == "full":
        return "full fine-tuning"
    if adaptation == "frozen":
        freeze_backbone(encoder)
        return "frozen encoder (head only)"
    if adaptation == "partial":
        freeze_all_but_last_block(encoder)
        return "partial fine-tuning (last block)"
    if adaptation == "lora":
        from downstream.lora import apply_lora, count_trainable

        n = apply_lora(encoder, m.lora_rank, m.lora_alpha, m.lora_dropout, tuple(m.lora_targets))
        trainable, total = count_trainable(encoder)
        return f"LoRA r={m.lora_rank} alpha={m.lora_alpha} on {n} layers ({trainable:,}/{total:,} encoder params trainable)"
    raise ValueError(f"unknown model.adaptation {adaptation!r} (expected lora/full/partial/frozen)")


def _interpolate_pos_embed(encoder: ViTEncoder, state_dict: dict) -> None:
    """Resize a checkpoint's position embedding to this model's grid size, if they differ.

    Needed if the ASP checkpoint was pretrained at a different resolution than `config.roi`.
    """
    key = "patch_embedding.position_embeddings"
    if key not in state_dict:
        return
    checkpoint_pos_embed = state_dict[key]
    embed_dim = checkpoint_pos_embed.shape[-1]
    checkpoint_n_patches = checkpoint_pos_embed.shape[-2]
    target_n_patches = encoder.patch_embedding.n_patches
    if checkpoint_n_patches == target_n_patches:
        return

    checkpoint_grid_size = round(checkpoint_n_patches ** (1 / 3))
    target_grid_size = round(target_n_patches ** (1 / 3))
    print(f"Interpolating position embedding: grid {checkpoint_grid_size}^3 -> {target_grid_size}^3")

    pos_tokens = checkpoint_pos_embed.reshape(1, checkpoint_grid_size, checkpoint_grid_size, checkpoint_grid_size, embed_dim)
    pos_tokens = pos_tokens.permute(0, 4, 1, 2, 3)
    pos_tokens = F.interpolate(
        pos_tokens, size=(target_grid_size,) * 3, mode="trilinear", align_corners=False
    )
    state_dict[key] = pos_tokens.permute(0, 2, 3, 4, 1).flatten(1, 3)


def load_asp_checkpoint(encoder: ViTEncoder, checkpoint_path: str) -> nn.modules.module._IncompatibleKeys:
    """Load an ASP-pretrained encoder checkpoint into `encoder`.

    The raw ASP checkpoint stores encoder weights under a `model_mae` sub-key (alongside
    decoder/segmentation-head weights this repo has no use for), saved from a
    `DistributedDataParallel`-wrapped model (hence the `module.` prefix on every key). Loaded
    with `strict=False` since the decoder/segmentation-head keys don't exist in this encoder-only
    model.
    """
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict = raw["model_mae"] if "model_mae" in raw else raw
    state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    _interpolate_pos_embed(encoder, state_dict)
    return encoder.load_state_dict(state_dict, strict=False)


def load_finetuned_checkpoint(encoder: ViTEncoder, head: Head, checkpoint_path: str) -> None:
    """Load a checkpoint produced by this repo's own `train.py` (encoder + head together).

    Used exclusively by `evaluate.py`, which always scores an already fine-tuned checkpoint --
    e.g. `configs/eval/ad_diagnosis.yaml` scoring the checkpoint `configs/finetune/
    ad_diagnosis_train.yaml` produced, broken down by cohort.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder.load_state_dict(checkpoint["encoder"])
    head.load_state_dict(checkpoint["head"])
