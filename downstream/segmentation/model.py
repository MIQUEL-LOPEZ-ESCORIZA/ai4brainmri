"""Segmentation models for the BraTS (whole tumor) and ATLAS R2.0 (stroke lesion) tasks.

- `vit_unetr`: the paper's pretrained-ViT segmentation model. A UNETR decoder (Hatamizadeh et
  al., 2022) on top of *our* 3D ViT-B encoder (`downstream.model.ViTEncoder`), initialized from
  an ASP or MAE checkpoint and adapted with LoRA (paper Methods, "Downstream Application of
  Pretrained ViTs"). Hidden states after blocks 3, 6, 9 and 12 feed the decoder skip paths.
- `unetr`: the same architecture trained from scratch (MONAI UNETR), the "UNETR" baseline.
- `unet`: a 3D U-Net trained from scratch (MONAI), the "U-Net" baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import UNETR, UNet

from downstream.config import Config
from downstream.model import ViTEncoder, load_asp_checkpoint, prepare_encoder


class _EncoderAdapter(nn.Module):
    """Makes `ViTEncoder` look like MONAI's `ViT` (returns `(tokens, hidden_states)`)."""

    def __init__(self, encoder: ViTEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor):
        return self.encoder.forward_tokens(x)


def build_segmentation_model(config: Config, kind: str, out_channels: int = 2, feature_size: int = 16) -> nn.Module:
    roi = tuple(config.roi)
    if kind == "unet":
        return UNet(spatial_dims=3, in_channels=1, out_channels=out_channels,
                    channels=(32, 64, 128, 256, 320), strides=(2, 2, 2, 2), num_res_units=2)

    m = config.model
    model = UNETR(in_channels=1, out_channels=out_channels, img_size=roi, feature_size=feature_size,
                  hidden_size=m.hidden_size, mlp_dim=m.mlp_dim, num_heads=m.num_heads,
                  norm_name="instance", res_block=True, dropout_rate=0.0)
    if kind == "unetr":
        return model
    if kind != "vit_unetr":
        raise ValueError(f"unknown segmentation model {kind!r} (vit_unetr / unetr / unet)")

    encoder = ViTEncoder(config)
    if m.pretrained:
        print(f"Loaded pretrained encoder {m.pretrained}: {load_asp_checkpoint(encoder, m.pretrained)}")
    print(f"Encoder adaptation: {prepare_encoder(encoder, config)}")
    model.vit = _EncoderAdapter(encoder)
    return model
