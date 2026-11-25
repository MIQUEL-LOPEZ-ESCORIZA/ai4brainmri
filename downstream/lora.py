"""Low-Rank Adaptation (LoRA) for the 3D ViT encoder.

The paper's default adaptation strategy for every downstream task that is not linear-probed
(Alzheimer's diagnosis, epilepsy, brain age, and the ViT encoders of the segmentation models):
the pretrained encoder weights stay frozen and trainable low-rank matrices are added to the
attention layers (Hu et al., 2022; MeLo, Zhu et al., 2024). Only the LoRA matrices and the task
head are optimized.

MONAI's `TransformerBlock` computes queries/keys/values with a single fused
`attn.qkv: nn.Linear(hidden, 3 * hidden)` and projects back with `attn.out_proj`. `apply_lora`
wraps those linear layers in place, so the module tree (and therefore the checkpoint keys of the
original weights) stays unchanged: `...attn.qkv.base.weight` holds the frozen pretrained weight,
`...attn.qkv.lora_a` / `lora_b` hold the adapter.

Hyperparameters (`model.lora_rank`, `model.lora_alpha`, `model.lora_targets`) are set per task
in the YAML configs. See `downstream/README.md` for the values used in the paper.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """`y = base(x) + (alpha / r) * B(A(dropout(x)))`, with `base` frozen and `B` zero-initialized
    so the wrapped layer is exactly the pretrained layer at the start of training."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:  # some MONAI code paths read `.weight` directly
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora = (self.dropout(x) @ self.lora_a.t()) @ self.lora_b.t()
        return self.base(x) + self.scaling * lora

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """The equivalent dense weight, for exporting a LoRA model without the adapter overhead."""
        return self.base.weight + self.scaling * (self.lora_b @ self.lora_a)


def apply_lora(encoder: nn.Module, rank: int, alpha: float, dropout: float = 0.0,
               targets: tuple[str, ...] = ("qkv",)) -> int:
    """Freeze `encoder` and wrap every `nn.Linear` whose attribute name is in `targets` inside an
    attention module with a `LoRALinear`. Returns the number of wrapped layers.

    `targets` defaults to the fused query/key/value projection, as in the paper ("trainable
    low-rank matrices were introduced into the attention layers"); add "out_proj" to also adapt
    the attention output projection.
    """
    for p in encoder.parameters():
        p.requires_grad = False

    wrapped = 0
    for module in list(encoder.modules()):
        for name in targets:
            child = getattr(module, name, None)
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha, dropout))
                wrapped += 1
    if wrapped == 0:
        raise RuntimeError(f"apply_lora found no nn.Linear named any of {targets} in the encoder")
    return wrapped


def lora_state_dict(module: nn.Module) -> dict:
    """Only the LoRA parameters (small enough to share separately from the pretrained encoder)."""
    return {k: v for k, v in module.state_dict().items() if "lora_" in k}


def count_trainable(module: nn.Module) -> tuple[int, int]:
    """`(trainable, total)` parameter counts, for logging the size of the adapted model."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total
