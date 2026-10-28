"""Phase 6 step 3: run each trained classifier on the test split.

Two sources of crops (both saved, different questions they answer):

* --source gt   crops from the ground-truth test boxes -> classifier quality alone
                (primary cross-model comparison).
* --source det  crops from the boxes YOLO26n predicted on the test images ->
                end-to-end system quality. Each predicted box is matched to the
                best-IoU ground-truth box (IoU >= 0.5); that class is the "truth",
                unmatched boxes get truth = -1 (the detector missed/misplaced them).

Output per run:
    results/testing/<model>_<source>_test.csv    one row per crop
    results/testing/<model>_<source>_test.npz    float16 class probabilities (for ROC)

Usage:
    python scripts/testing/test_classifiers.py --model convnext_tiny --source gt
    python scripts/testing/test_classifiers.py --model fastvit_sa12 --source det --max-images 50
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from class_mapping.class_map import CLASSES  # noqa: E402
from registry import build_model  # noqa: E402
from crops import CropDataset, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from torchvision import transforms  # noqa: E402

NUM_CLASSES = len(CLASSES)


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float, float]]:
    """-> list of (class, cx, cy, w, h, conf); GT files have no conf -> 1.0."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            c, cx, cy, w, h = parts[:5]
            conf = float(parts[5]) if len(parts) >= 6 else 1.0
            out.append((int(c), float(cx), float(cy), float(w), float(h), conf))
    return out


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix, iy = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class DetectedCropDataset(Dataset):
    """Crops from the detector's predicted boxes, matched to ground truth by IoU."""

    def __init__(self, images_dir: Path, det_dir: Path, gt_dir: Path,
                 img_size: int, iou_thresh: float = 0.5):
        self.images_dir = images_dir
        self.img_size = img_size
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.meta: list[tuple[str, int, float, float]] = []  # stem, det_cls, conf, best_iou
        self.truth: list[int] = []
        self.boxes: list[tuple[float, float, float, float]] = []

        for det_file in sorted(det_dir.glob("*.txt")):
            stem = det_file.stem
            gt = read_yolo_labels(gt_dir / f"{stem}.txt")
            for det_cls, cx, cy, w, h, conf in read_yolo_labels(det_file):
                best_iou, best_cls = 0.0, -1
                for gcls, gcx, gcy, gw, gh, _gconf in gt:
                    v = iou((cx, cy, w, h), (gcx, gcy, gw, gh))
                    if v > best_iou:
                        best_iou, best_cls = v, gcls
                self.meta.append((stem, det_cls, conf, best_iou))
                self.truth.append(best_cls if best_iou >= iou_thresh else -1)
                self.boxes.append((cx, cy, w, h))

    def __len__(self) -> int:
        return len(self.boxes)

    def __getitem__(self, index: int):
        stem, det_cls, conf, best_iou = self.meta[index]
        cx, cy, w, h = self.boxes[index]
        with Image.open(self.images_dir / f"{stem}.jpg") as img:
            width, height = img.size
            x1 = max(0, int(round((cx - w / 2) * width)))
            y1 = max(0, int(round((cy - h / 2) * height)))
            x2 = min(width, int(round((cx + w / 2) * width)))
            y2 = min(height, int(round((cy + h / 2) * height)))
            if x2 <= x1:
                x2 = min(width, x1 + 2)
            if y2 <= y1:
                y2 = min(height, y1 + 2)
            scale = 1.0
            if min(x2 - x1, y2 - y1) >= 1.5 * self.img_size:
                img.draft("RGB", (width // 2, height // 2))
                scale = img.size[0] / width
            crop = img.convert("RGB").crop(
                (int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)))
        return self.tf(crop), index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", choices=["gt", "det"], required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--images", default="data/yolo/images/test")
    parser.add_argument("--gt-labels", default="data/yolo/labels/test")
    parser.add_argument("--det-labels", default="results/testing/yolo26n/labels")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0, help="0 = all")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    if args.threads:
        torch.set_num_threads(args.threads)

    ckpt_path = Path(args.weights or f"results/{args.model}/best.pt")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    cfg = blob.get("config", {}) if isinstance(blob, dict) else {}
    img_size = args.img_size or int(cfg.get("img_size", 128))

    model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.eval()

    images_dir = PROJECT_ROOT / args.images
    out_csv = PROJECT_ROOT / "results/testing" / f"{args.model}_{args.source}_test.csv"
    out_npz = PROJECT_ROOT / "results/testing" / f"{args.model}_{args.source}_test.npz"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "gt":
        labels_dir = PROJECT_ROOT / args.gt_labels
        files = sorted(labels_dir.glob("*.txt"))
        if args.max_images:
            allowed = {p for p in files[: args.max_images]}
            files = files[: args.max_images]
        ds = CropDataset(images_dir, labels_dir, img_size=img_size, train=False)
        if args.max_images:
            ds.samples = [s for s in ds.samples if (labels_dir / f"{s[0]}.txt") in allowed]
        meta_rows = [{"image": s[0], "cx": s[2], "cy": s[3], "w": s[4], "h": s[5],
                      "truth": s[1]} for s in ds.samples]
    else:
        det_dir = PROJECT_ROOT / args.det_labels
        if args.max_images:
            keep = {p.stem for p in sorted(det_dir.glob("*.txt"))[: args.max_images]}
        ds = DetectedCropDataset(images_dir, det_dir, PROJECT_ROOT / args.gt_labels, img_size)
        if args.max_images:
            keep_mask = [m[0] in keep for m in ds.meta]
            ds.meta = [m for m, k in zip(ds.meta, keep_mask) if k]
            ds.truth = [t for t, k in zip(ds.truth, keep_mask) if k]
            ds.boxes = [b for b, k in zip(ds.boxes, keep_mask) if k]
        meta_rows = [{"image": m[0], "cx": b[0], "cy": b[1], "w": b[2], "h": b[3],
                      "truth": t, "det_cls": m[1], "conf": m[2], "iou": m[3]}
                     for m, b, t in zip(ds.meta, ds.boxes, ds.truth)]

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False,
                        persistent_workers=args.num_workers > 0)

    print(f"{args.model} | source={args.source} | {len(ds):,} crops | img {img_size} | "
          f"batch {args.batch_size} | weights {ckpt_path}", flush=True)

    n_pred = np.zeros(len(ds), dtype=np.int64)
    n_conf = np.zeros(len(ds), dtype=np.float32)
    probs = np.zeros((len(ds), NUM_CLASSES), dtype=np.float16)
    truth_arr = np.array([r["truth"] for r in meta_rows], dtype=np.int64)

    t0 = time.perf_counter()
    done = 0
    for xb, idx in loader:
        out = model(xb)
        p = torch.softmax(out, dim=1).numpy().astype(np.float16)
        idx = idx.numpy()
        probs[idx] = p
        n_pred[idx] = p.argmax(axis=1)
        n_conf[idx] = p.max(axis=1).astype(np.float32)
        done += len(idx)
        if done % (args.batch_size * 20) < args.batch_size:
            rate = done / max(time.perf_counter() - t0, 1e-9)
            print(f"  {done:,}/{len(ds):,} crops | {rate:,.0f}/s", flush=True)

    dt = time.perf_counter() - t0
    correct = (n_pred == truth_arr) & (truth_arr >= 0)
    acc = correct.sum() / max((truth_arr >= 0).sum(), 1)

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "image", "cx", "cy", "w", "h", "det_cls", "det_conf", "iou", "truth", "pred",
            "pred_conf", "correct"])
        writer.writeheader()
        for i, row in enumerate(meta_rows):
            out_row = {**row, "det_conf": row.get("conf", ""),
                       "pred": int(n_pred[i]), "pred_conf": float(n_conf[i]),
                       "correct": bool(correct[i])}
            out_row.pop("conf", None)
            writer.writerow(out_row)

    np.savez_compressed(out_npz, probs=probs, pred=n_pred, conf=n_conf, truth=truth_arr)
    print(f"done in {dt / 60:.1f} min | test accuracy (truth >= 0): {acc:.4f} | "
          f"unmatched (truth == -1): {int((truth_arr < 0).sum()):,}", flush=True)
    print(f"saved: {out_csv.name} + {out_npz.name}", flush=True)


if __name__ == "__main__":
    main()