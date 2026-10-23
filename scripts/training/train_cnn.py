"""Phase 5 - train one classifier on RICO crops (on the fly, nothing stored).

Protocol (same for every model):
  * crops from the train split, ground-truth boxes, augmented
  * validation on the val split, resize only, no augmentation
  * class-weighted cross-entropy (inverse frequency, mean weight = 1)
  * AdamW + cosine schedule, weight decay, label smoothing, early stopping
  * model selection on validation macro-F1, best checkpoint kept

Usage:
    python scripts/training/train_cnn.py --model densenet121
    python scripts/training/train_cnn.py --model pyramidnet110 --epochs 2 --max-crops 400 --out-suffix smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "class_mapping"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from class_map import CLASSES  # noqa: E402
from crops import CropDataset, class_weights  # noqa: E402
from registry import build_model, count_params  # noqa: E402

NUM_CLASSES = len(CLASSES)


def load_config(model: str, config_dir: Path) -> dict:
    path = config_dir / f"{model}.yaml"
    if not path.is_file():
        raise SystemExit(f"missing config {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_train_classes(labels_dir: Path) -> dict[int, int]:
    """Class counts over the whole train split (not the capped crop sample)."""
    counts: dict[int, int] = {}
    for label_file in labels_dir.glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                cid = int(parts[0])
                counts[cid] = counts.get(cid, 0) + 1
    return counts


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss, n = 0.0, 0
    preds: list[int] = []
    targets: list[int] = []
    use_amp = device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        targets.extend(y.cpu().tolist())

    labels = list(range(NUM_CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=labels, zero_division=0
    )
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, labels=labels, average="macro", zero_division=0),
        "per_class": {
            CLASSES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in labels
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "configs")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "yolo")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--out-suffix", default="", help="suffix for the result folder (smoke runs)")
    # overrides used by smoke runs
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--img-size", type=int)
    parser.add_argument("--max-crops", type=int)
    parser.add_argument("--val-per-class", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--no-pretrained", action="store_true", help="random init (offline smoke runs)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.model, args.config_dir)
    for key in ("epochs", "batch_size", "img_size", "max_crops", "val_per_class", "lr", "num_workers"):
        override = getattr(args, key)
        if override is not None:
            cfg[key] = override
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.no_pretrained:
        cfg["pretrained"] = False
    seed_everything(cfg.get("seed", 42))

    images = args.data_dir / "images"
    labels = args.data_dir / "labels"
    train_ds = CropDataset(
        images / "train", labels / "train", img_size=cfg["img_size"], train=True,
        max_crops=cfg.get("max_crops", 0), seed=cfg.get("seed", 42),
    )
    val_ds = CropDataset(
        images / "val", labels / "val", img_size=cfg["img_size"], train=False,
        max_per_class=cfg.get("val_per_class", 0), seed=cfg.get("seed", 42),
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("empty train or val crop set - is the split built?")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg.get("num_workers", 4), pin_memory=device.type == "cuda",
                              drop_last=len(train_ds) > cfg["batch_size"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg.get("num_workers", 4), pin_memory=device.type == "cuda")

    counts = count_train_classes(labels / "train")
    weights = class_weights(counts, NUM_CLASSES, power=cfg.get("weight_power", 1.0)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.get("label_smoothing", 0.0))

    model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=cfg.get("pretrained", True)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg["epochs"], 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    run_name = f"{args.model}{args.out_suffix}"
    out_dir = args.results_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    print(f"model {args.model} | {count_params(model) / 1e6:.2f} M params | device {device}")
    print(f"train crops {len(train_ds):,} | val crops {len(val_ds):,} | batch {cfg['batch_size']} | img {cfg['img_size']}")
    print(f"class weights min {weights.min():.3f} max {weights.max():.3f}")
    print(f"output {out_dir}\n")

    history: list[dict] = []
    best_f1, best_epoch, bad_epochs = -1.0, 0, 0
    patience = cfg.get("patience", 10)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        total_loss, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * y.size(0)
            seen += y.size(0)
        train_loss = total_loss / max(seen, 1)

        val = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": round(train_loss, 5),
            "val_loss": round(val["loss"], 5),
            "val_accuracy": round(val["accuracy"], 5),
            "val_macro_f1": round(val["macro_f1"], 5),
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(f"epoch {epoch:>3}/{cfg['epochs']} | train {train_loss:.4f} | val {val['loss']:.4f} | "
              f"acc {val['accuracy']:.4f} | macroF1 {val['macro_f1']:.4f} | {row['seconds']}s", flush=True)

        torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg,
                    "classes": CLASSES}, out_dir / "last.pt")
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = val["macro_f1"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg,
                        "classes": CLASSES, "val_macro_f1": best_f1}, out_dir / "best.pt")
            (out_dir / "val_per_class.json").write_text(json.dumps(val["per_class"], indent=2), encoding="utf-8")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch} (best {best_f1:.4f} @ epoch {best_epoch})")
                break

    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nbest val macro-F1 {best_f1:.4f} @ epoch {best_epoch}")
    print(f"saved: {out_dir / 'best.pt'} + metrics.csv + history.json")


if __name__ == "__main__":
    main()
