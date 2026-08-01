
import os
import sys
import argparse
import torch
import wandb  # W&B Integration
import math

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
from monai.config import print_config




class ConvDecoder(nn.Module):
    """
    From a 768-dim input plus an embedding lookup of the same size,
    decode a (128,128,128) volume via 3D transposed convolutions.
    Output has shape (batch, 1, 128, 128, 128).
    """
    def __init__(self, num_regions, embedding_dim=768, out_channels=1):
        super(ConvDecoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_regions = num_regions

        # Embedding table for region IDs (size: [num_regions, 768])
        self.region_embedding = nn.Embedding(num_regions, embedding_dim)

        # --- 6 Transposed-Conv layers ---
        # Each deconv is followed by (BatchNorm + ReLU) except the final one.
        #
        # Shapes:
        #   Input  shape: (batch, 768,   1,   1,   1)
        #   Deconv1 out: (batch, 256,   4,   4,   4)
        #   Deconv2 out: (batch, 128,   8,   8,   8)
        #   Deconv3 out: (batch, 64,   16,  16,  16)
        #   Deconv4 out: (batch, 32,   32,  32,  32)
        #   Deconv5 out: (batch, 16,   64,  64,  64)
        #   Deconv6 out: (batch, 1,   128, 128, 128)

        self.deconv1 = nn.ConvTranspose3d(
            in_channels=768,
            out_channels=256,
            kernel_size=4,  # expand spatial dim from 1 to 4
            stride=4,
            padding=0
        )
        self.bn1 = nn.BatchNorm3d(256)

        self.deconv2 = nn.ConvTranspose3d(
            in_channels=256,
            out_channels=128,
            kernel_size=4,  # double spatial dim 4->8
            stride=2,
            padding=1
        )
        self.bn2 = nn.BatchNorm3d(128)

        self.deconv3 = nn.ConvTranspose3d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,  # 8->16
            stride=2,
            padding=1
        )
        self.bn3 = nn.BatchNorm3d(64)

        self.deconv4 = nn.ConvTranspose3d(
            in_channels=64,
            out_channels=32,
            kernel_size=4,  # 16->32
            stride=2,
            padding=1
        )
        self.bn4 = nn.BatchNorm3d(32)

        self.deconv5 = nn.ConvTranspose3d(
            in_channels=32,
            out_channels=16,
            kernel_size=4,  # 32->64
            stride=2,
            padding=1
        )
        self.bn5 = nn.BatchNorm3d(16)

        self.deconv6 = nn.ConvTranspose3d(
            in_channels=16,
            out_channels=out_channels,
            kernel_size=4,  # 64->128
            stride=2,
            padding=1
        )
        # No BatchNorm/ReLU on the last layer by default
        # (you could add a final activation like Sigmoid or Tanh if desired)

    def forward(self, x, region_ids):
        """
        x:          (batch, 768)
        region_ids: (batch,) with each ID in [0, num_regions-1]
        Returns:    (batch, out_channels, 128, 128, 128)
        """
        bsz = x.size(0)

        # 1) Look up region embedding => (batch, 768)
        region_emb = self.region_embedding(region_ids)

        # 2) Sum the two embeddings => (batch, 768)
        combined = x + region_emb

        # 3) Reshape => (batch, 768, 1, 1, 1)
        out = combined.view(bsz, self.embedding_dim, 1, 1, 1)

        # 4) Pass through transposed conv layers
        #
        # Deconv1 => (batch,256,4,4,4)
        out = self.deconv1(out)
        out = self.bn1(out)
        out = F.relu(out)

        # Deconv2 => (batch,128,8,8,8)
        out = self.deconv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # Deconv3 => (batch,64,16,16,16)
        out = self.deconv3(out)
        out = self.bn3(out)
        out = F.relu(out)

        # Deconv4 => (batch,32,32,32,32)
        out = self.deconv4(out)
        out = self.bn4(out)
        out = F.relu(out)

        # Deconv5 => (batch,16,64,64,64)
        out = self.deconv5(out)
        out = self.bn5(out)
        out = F.relu(out)

        # Deconv6 => (batch,1,128,128,128)
        out = self.deconv6(out)
        # typically no BN or ReLU on last layer
        return out
    


    def forward_avg_regions(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 768) - image embedding

        Returns:
            Tensor of shape (batch_size, 256) - intermediate embedding after averaging
        """
        bsz = x.size(0)

        # (num_regions, 768)
        region_ids = torch.arange(self.num_regions, device=x.device)
        region_embs = self.region_embedding(region_ids)  # (num_regions, 768)

        # (batch_size, num_regions, 768)
        combined = x.unsqueeze(1) + region_embs.unsqueeze(0)

        # (batch_size * num_regions, 768, 1, 1, 1)
        combined = combined.view(-1, self.embedding_dim, 1, 1, 1)

        # Deconv1 + BN + ReLU → (batch_size * num_regions, 256, 4, 4, 4)
        out = self.deconv1(combined)
        out = self.bn1(out)
        out = F.relu(out)

        # Reshape → (batch_size, num_regions, 256, 4, 4, 4)
        out = out.view(bsz, self.num_regions, 256, 4, 4, 4)

        # Average across regions → (batch_size, 256, 4, 4, 4)
        out = out.mean(dim=1)

        # Spatial average → (batch_size, 256)
        out = out.mean(dim=[2, 3, 4])

        return out

    def forward_avg_regions_deep(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 768) - image embedding

        Returns:
            Tensor of shape (batch_size, 256) - intermediate embedding after 3 decoder layers
        """
        bsz = x.size(0)

        # (num_regions, 768)
        region_ids = torch.arange(self.num_regions, device=x.device)
        region_embs = self.region_embedding(region_ids)  # (num_regions, 768)

        # (batch_size, num_regions, 768)
        combined = x.unsqueeze(1) + region_embs.unsqueeze(0)

        # (batch_size * num_regions, 768, 1, 1, 1)
        combined = combined.view(-1, self.embedding_dim, 1, 1, 1)

        # ------------------
        # First three decoder layers
        # ------------------
        # Deconv1
        out = self.deconv1(combined)  # → (B*num_regions, 256, 4, 4, 4)
        out = self.bn1(out)
        out = F.relu(out)

        # Deconv2
        out = self.deconv2(out)       # → (B*num_regions, 128, 8, 8, 8)
        out = self.bn2(out)
        out = F.relu(out)

        # Deconv3
        out = self.deconv3(out)       # → (B*num_regions, 64, 16, 16, 16)
        out = self.bn3(out)
        out = F.relu(out)

        # Reshape back to (batch_size, num_regions, C, D, H, W)
        out = out.view(bsz, self.num_regions, 64, 16, 16, 16)

        # Average across regions → (batch_size, 64, 16, 16, 16)
        out = out.mean(dim=1)

        # Final spatial average to get shape (batch_size, 256)
        # Flatten spatial dims, then do one linear projection to get to 256-dim
        out = out.view(bsz, 64, -1).mean(dim=2)  # → (batch_size, 64)


        return out
    def forward_intermidiate_hippo(self, x, region_id=8):
        bsz = x.size(0)

        region_emb = self.region_embedding(torch.full((bsz,), region_id, device=x.device))
        combined = x + region_emb
        out = combined.view(bsz, self.embedding_dim, 1, 1, 1)

        out = self.deconv1(out)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.deconv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = out.view(bsz, -1)  # flatten to (batch, 65536)

        # Option: truncate
        #out = out[:, :768]

        # Option: mean-pool to 768 without params
        factor = 65536 // 768
        out = out[:, :factor * 768].view(bsz, 768, factor).mean(dim=2)

        return out





class PromptSegmentationModel(nn.Module):
    def __init__(self, encoder, decoder):
        """
        Segmentation model with multiple decoders.
        
        Args:
            encoder (nn.Module): The shared encoder (e.g., ViT).
            num_decoders (int): The number of separate decoders (one per labeled mask).
            output_shape (tuple): Shape of the output segmentation masks.
        """
        super(PromptSegmentationModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x, brain_region):
        embedding = self.encoder(x)  # Shared image embedding

        output = self.decoder(embedding,brain_region)

        return output # List of outputs, one per decoder


class PromptSegmentationModelIntermidiate(nn.Module):
    def __init__(self, encoder, decoder):
        """
        Segmentation model with multiple decoders.
        
        Args:
            encoder (nn.Module): The shared encoder (e.g., ViT).
            num_decoders (int): The number of separate decoders (one per labeled mask).
            output_shape (tuple): Shape of the output segmentation masks.
        """
        super(PromptSegmentationModelIntermidiate, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x):
        embedding = self.encoder(x)  # Shared image embedding

        # output = self.decoder.forward_avg_regions(embedding)
        output = self.decoder.forward_avg_regions_deep(embedding)

        return output # List of outputs, one per decoder

class PromptSegmentationModelIntermidiateHippo(nn.Module):
    def __init__(self, encoder, decoder):
        """
        Segmentation model with multiple decoders.
        
        Args:
            encoder (nn.Module): The shared encoder (e.g., ViT).
            num_decoders (int): The number of separate decoders (one per labeled mask).
            output_shape (tuple): Shape of the output segmentation masks.
        """
        super(PromptSegmentationModelIntermidiateHippo, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x):
        embedding = self.encoder(x)  # Shared image embedding

        # output = self.decoder.forward_avg_regions(embedding)
        output = self.decoder.forward_intermidiate_hippo(embedding)

        return output # List of outputs, one per decoder