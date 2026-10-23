"""Phase 5 - train the YOLO detector (box stage).

Config comes from configs/<model>.yaml. Class imbalance: Ultralytics only exposes
a single scalar `cls_pw` (positive-class weight), not per-class weights; the
per-class weighting lives in the classifier stage. This limitation is recorded in
docs/memory.md.

Usage:
    python scripts/training/train_yolo.py --model yolo26n
    python scripts/training/train_yolo.py --model yolo26n --epochs 1 --imgsz 320 --limit-images 40 --out-suffix smoke
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "class_mapping"))

from class_map import CLASSES  # noqa: E402


def make_subset_yaml(data_dir: Path, limit: int, seed: int) -> Path:
    """Small train/val file lists so a smoke run finishes on CPU."""
    rng = random.Random(seed)
    smoke = data_dir / "_smoke"
    smoke.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        images = sorted((data_dir / "images" / split).glob("*.jpg"))
        if not images:
            raise SystemExit(f"no images in {data_dir / 'images' / split}")
        picked = rng.sample(images, min(limit, len(images)))
        (smoke / f"{split}.txt").write_text(
            "\n".join(p.resolve().as_posix() for p in picked) + "\n", encoding="utf-8"
        )
    yaml_path = smoke / "data_smoke.yaml"
    yaml_path.write_text(
        "\n".join([
            f"path: {data_dir.as_posix()}",
            f"train: {(smoke / 'train.txt').as_posix()}",
            f"val: {(smoke / 'val.txt').as_posix()}",
            f"nc: {len(CLASSES)}",
            "names:",
        ] + [f"  {i}: {name}" for i, name in enumerate(CLASSES)]) + "\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26n")
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "configs")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "yolo")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--out-suffix", default="")
    parser.add_argument("--limit-images", type=int, default=0, help="smoke run: N train and N val images")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg_path = args.config_dir / f"{args.model}.yaml"
    if not cfg_path.is_file():
        raise SystemExit(f"missing config {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    for key in ("epochs", "imgsz", "batch"):
        override = getattr(args, key)
        if override is not None:
            cfg[key] = override

    data_yaml = args.data_dir / "data.yaml"
    if args.limit_images:
        data_yaml = make_subset_yaml(args.data_dir, args.limit_images, cfg.get("seed", 42))
        print(f"smoke run on {args.limit_images} images -> {data_yaml}")

    from ultralytics import YOLO

    model = YOLO(cfg["model"])
    run_dir = args.results_dir / f"{args.model}{args.out_suffix}"
    print(f"training {cfg['model']} | epochs {cfg['epochs']} | imgsz {cfg['imgsz']} | batch {cfg['batch']}")
    print(f"output {run_dir}")

    model.train(
        data=str(data_yaml),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        optimizer=cfg.get("optimizer", "AdamW"),
        lr0=cfg.get("lr0", 0.001),
        weight_decay=cfg.get("weight_decay", 0.0005),
        patience=cfg.get("patience", 15),
        cls_pw=cfg.get("cls_pw", 0.0),
        cos_lr=cfg.get("cos_lr", True),
        workers=cfg.get("workers", 4),
        seed=cfg.get("seed", 42),
        deterministic=True,
        device=args.device or "",
        project=str(args.results_dir),
        name=f"{args.model}{args.out_suffix}",
        exist_ok=True,
        plots=True,
        val=True,
    )
    metrics = model.val(data=str(data_yaml), project=str(run_dir), name="val", exist_ok=True)
    print(f"\nmAP50 {metrics.box.map50:.4f} | mAP50-95 {metrics.box.map:.4f}")
    print(f"weights: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
