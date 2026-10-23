"""PyramidNet-A (Han et al., 2017) - no pretrained weights exist for it.

Faithful reimplementation of the additive-shortcut variant:
  * stem conv3x3 -> BN -> ReLU, 16 channels, no max-pool
  * 3 stages of BasicBlocks; the width grows smoothly from 16 to 16+alpha over
    the whole network (paper: alpha=48 -> final width 64 -> 1.7M params, same
    budget as ResNet-110), i.e. delta = alpha / total_blocks per block
  * first block of stages 2 and 3 has stride 2
  * shortcut = average-pool (when downsampling) + zero-padding of new channels
  * final conv1x1 -> BN -> ReLU -> global average pool -> dropout -> linear
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)

    def shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            x = F.avg_pool2d(x, 2, self.stride)
        pad = self.out_channels - self.in_channels
        if pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return F.relu(out + self.shortcut(x))


class PyramidNet(nn.Module):
    def __init__(self, blocks: tuple[int, int, int] = (18, 18, 18), alpha: int = 48,
                 num_classes: int = 15, dropout: float = 0.2) -> None:
        super().__init__()
        total_blocks = sum(blocks)
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layers = nn.ModuleList()
        done = 0
        for stage, n_blocks in enumerate(blocks):
            self.layers.append(self._make_stage(n_blocks, alpha, total_blocks, done,
                                                first_stride=1 if stage == 0 else 2))
            done += n_blocks

        self.bn_out = nn.BatchNorm2d(self.in_channels)
        self.conv_out = nn.Conv2d(self.in_channels, self.in_channels, 1, 1, 0, bias=False)
        self.bn_final = nn.BatchNorm2d(self.in_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.in_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_stage(self, n_blocks: int, alpha: int, total_blocks: int, done: int,
                    first_stride: int) -> nn.Sequential:
        blocks: list[nn.Module] = []
        for i in range(n_blocks):
            done += 1
            out_channels = int(round(16 + alpha * done / total_blocks))
            stride = first_stride if i == 0 else 1
            blocks.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            x = layer(x)
        x = F.relu(self.bn_out(x))
        x = self.conv_out(x)
        x = F.relu(self.bn_final(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(self.dropout(x))


def pyramidnet110(num_classes: int = 15, dropout: float = 0.2) -> PyramidNet:
    return PyramidNet(blocks=(18, 18, 18), alpha=48, num_classes=num_classes, dropout=dropout)
