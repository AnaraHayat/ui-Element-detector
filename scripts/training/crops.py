"""Crop dataset: cut boxes out of the full screenshots on the fly.

Nothing is written to disk. Training crops come from the ground-truth YOLO
labels of the train split; at inference the boxes come from YOLO detections.

Class imbalance is handled with class-weighted loss in train_cnn.py, not by
resampling, so the natural class distribution is preserved.

Usage (from other scripts):
    ds = CropDataset(images_dir, labels_dir, img_size=128, train=True, max_crops=200_000)
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CropDataset(Dataset):
    """Each item is (image tensor, class id)."""

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        img_size: int = 128,
        train: bool = True,
        max_crops: int = 0,
        max_per_class: int = 0,
        seed: int = 42,
        augment: bool = True,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.train = train

        samples: list[tuple[str, int, float, float, float, float]] = []
        for label_file in sorted(self.labels_dir.glob("*.txt")):
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cid = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:])
                samples.append((label_file.stem, cid, cx, cy, w, h))

        if max_per_class:
            rng = random.Random(seed)
            by_class: dict[int, list] = {}
            for s in samples:
                by_class.setdefault(s[1], []).append(s)
            samples = [s for c in sorted(by_class) for s in rng.sample(by_class[c], min(max_per_class, len(by_class[c])))]

        if max_crops and len(samples) > max_crops:
            samples = random.Random(seed).sample(samples, max_crops)

        self.samples = samples

        if train and augment:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.8, 1.25)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            # validation/test: no augmentation, only the model's input adjustment
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for _, cid, *_ in self.samples:
            counts[cid] = counts.get(cid, 0) + 1
        return counts

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        stem, cid, cx, cy, w, h = self.samples[index]
        with Image.open(self.images_dir / f"{stem}.jpg") as img:
            img = img.convert("RGB")
            width, height = img.size
            x1 = max(0, int(round((cx - w / 2) * width)))
            y1 = max(0, int(round((cy - h / 2) * height)))
            x2 = min(width, int(round((cx + w / 2) * width)))
            y2 = min(height, int(round((cy + h / 2) * height)))
            if x2 <= x1:
                x2 = min(width, x1 + 2)
            if y2 <= y1:
                y2 = min(height, y1 + 2)
            crop = img.crop((x1, y1, x2, y2))
        return self.tf(crop), cid


def class_weights(counts: dict[int, int], num_classes: int = 15, power: float = 1.0) -> torch.Tensor:
    """Inverse-frequency weights normalized to mean 1 (imported by train_cnn.py)."""
    total = sum(counts.values())
    weights = []
    for cid in range(num_classes):
        n = counts.get(cid, 0)
        weights.append(0.0 if n == 0 else (total / (num_classes * n)) ** power)
    present = [w for w in weights if w > 0]
    mean = sum(present) / len(present) if present else 1.0
    return torch.tensor([w / mean for w in weights], dtype=torch.float32)
