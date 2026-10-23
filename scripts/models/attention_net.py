"""Residual Attention Network (Wang et al., 2017) - no pretrained weights exist.

Reimplementation of the attention module from the paper:
  trunk branch  T(x)  : residual blocks
  mask branch   M(x)  : encoder (p residual blocks + max-pool, t times)
                        -> decoder (t times: upsample + conv + residual blocks)
                        -> 1x1 conv + sigmoid
  output              : (1 + M(x)) * T(x)

Stack used here: preprocessing conv + pool, then 3 stages of attention modules
(1/1/2 modules), then attention refinement (global average pool + 1x1 conv +
sigmoid), then the classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two 3x3 convs with a projection shortcut when the shape changes."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class MaskBranch(nn.Module):
    """Encoder-decoder that produces the soft attention mask."""

    def __init__(self, channels: int, p: int = 1, t: int = 2, r: int = 1) -> None:
        super().__init__()
        self.encoder = nn.ModuleList()
        for _ in range(t):
            self.encoder.append(ResidualBlock(channels, channels))
            self.encoder.append(nn.MaxPool2d(2, 2))

        middle: list[nn.Module] = [ResidualBlock(channels, channels) for _ in range(p)]
        self.middle = nn.Sequential(*middle)

        self.decoder = nn.ModuleList()
        for _ in range(t):
            self.decoder.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.decoder.append(ResidualBlock(channels, channels))
            for _ in range(r):
                self.decoder.append(ResidualBlock(channels, channels))

        self.out_conv = nn.Conv2d(channels, channels, 1, 1, 0)
        self.out_bn = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        x = self.middle(x)
        for layer in self.decoder:
            x = layer(x)
        return self.sigmoid(self.out_bn(self.out_conv(x)))


class AttentionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, p: int = 1, t: int = 2, r: int = 1) -> None:
        super().__init__()
        self.trunk = ResidualBlock(in_channels, out_channels)
        self.mask = MaskBranch(out_channels, p=p, t=t, r=r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk = self.trunk(x)
        mask = self.mask(trunk)
        # the encoder/decoder can drift by a pixel on odd feature sizes; force alignment
        if mask.shape[-2:] != trunk.shape[-2:]:
            mask = nn.functional.interpolate(mask, size=trunk.shape[-2:], mode="nearest")
        return (1.0 + mask) * trunk


class AttentionRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1, 1, 0)
        self.bn = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.sigmoid(self.bn(self.conv(self.pool(x))))
        return (1.0 + mask) * x


class ResidualAttentionNet(nn.Module):
    def __init__(self, num_classes: int = 15, dropout: float = 0.2, widths: tuple[int, int, int] = (64, 128, 256)) -> None:
        super().__init__()
        w1, w2, w3 = widths
        self.pre = nn.Sequential(
            nn.Conv2d(3, w1, 7, 2, 3, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.stage1 = nn.Sequential(AttentionModule(w1, w1), nn.MaxPool2d(2, 2))
        self.stage2 = nn.Sequential(AttentionModule(w1, w2), nn.MaxPool2d(2, 2))
        self.stage3 = nn.Sequential(
            AttentionModule(w2, w3),
            AttentionModule(w3, w3),
        )
        self.refine = AttentionRefinement(w3)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(w3, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.refine(x)
        x = torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1)
        return self.fc(self.dropout(x))


def residual_attention(num_classes: int = 15, dropout: float = 0.2) -> ResidualAttentionNet:
    return ResidualAttentionNet(num_classes=num_classes, dropout=dropout)
