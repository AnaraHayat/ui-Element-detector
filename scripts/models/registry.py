"""Model registry: the 5 CNNs of the classification stage.

DenseNet, Xception and SE-ResNet come from timm with ImageNet weights (real
transfer learning). PyramidNet and Residual Attention Network have no published
pretrained weights, so they are built from our own implementations and trained
from scratch - see docs/memory.md.

The YOLO detector is not in this registry; it is loaded through Ultralytics in
scripts/training.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import timm
import torch.nn as nn

from attention_net import residual_attention
from pyramidnet import pyramidnet110

# name shown in results -> (timm name, human label)
PRETRAINED: dict[str, str] = {
    "densenet121": "densenet121.ra_in1k",
    "xception41": "xception41.tf_in1k",
    "seresnet50": "legacy_seresnet50.in1k",
}

# name -> factory(num_classes) -> model, trained from scratch
FROM_SCRATCH = {
    "pyramidnet110": pyramidnet110,
    "residual_attention": residual_attention,
}

ALL_MODELS: list[str] = list(PRETRAINED) + list(FROM_SCRATCH)
PRETRAINED_AVAILABLE: dict[str, bool] = {**{k: True for k in PRETRAINED}, **{k: False for k in FROM_SCRATCH}}


def build_model(name: str, num_classes: int = 15, pretrained: bool = True) -> nn.Module:
    """Build one classifier. `pretrained` is ignored for models without weights."""
    if name in PRETRAINED:
        return timm.create_model(
            PRETRAINED[name], pretrained=pretrained, num_classes=num_classes, drop_rate=0.2
        )
    if name in FROM_SCRATCH:
        return FROM_SCRATCH[name](num_classes=num_classes)
    raise KeyError(f"unknown model {name!r}; known: {ALL_MODELS}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
