"""Phase 3 - split the YOLO dataset 70:20:10 (train:test:validate) by image.

Input : data/yolo/labels/all/*.txt + the screenshots in data/raw/rico/unique_uis/combined
Output: data/yolo/images/{train,test,val}/<id>.jpg   (hard link, copy as fallback)
        data/yolo/labels/{train,test,val}/<id>.txt
        data/yolo/class_index/{split}/<ClassName>/images.txt
        data/yolo/data.yaml
        results/split/class_counts_per_split.csv, split_summary.json

The split is by image, never per box: every box of an image goes to the same split.
An image that contains 6 classes appears in 6 class_index folders (as a line in
images.txt), but its file is stored exactly once.

Usage:
    python scripts/split/make_split.py
    python scripts/split/make_split.py --seed 42 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "class_mapping"))

from class_map import CLASSES  # noqa: E402

RAW_IMAGES = PROJECT_ROOT / "data" / "raw" / "rico" / "unique_uis" / "combined"
YOLO = PROJECT_ROOT / "data" / "yolo"
LABELS_ALL = YOLO / "labels" / "all"
RESULTS = PROJECT_ROOT / "results" / "split"

# order matters only for reporting; the user's order is train:test:validate
SPLIT_RATIOS = [("train", 0.70), ("test", 0.20), ("val", 0.10)]


def split_names(stems: list[str], seed: int) -> dict[str, list[str]]:
    """Deterministic 70/20/10 split of image stems."""
    ordered = sorted(stems, key=lambda s: int(s) if s.isdigit() else s)
    random.Random(seed).shuffle(ordered)

    n = len(ordered)
    n_train = int(n * 0.70)
    n_test = int(n * 0.20)
    return {
        "train": ordered[:n_train],
        "test": ordered[n_train : n_train + n_test],
        "val": ordered[n_train + n_test :],
    }


def link_or_copy(src: Path, dst: Path) -> str:
    """Hard-link when the filesystem allows it (no extra disk), else copy."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def read_classes(label_path: Path) -> list[int]:
    ids: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.append(int(line.split()[0]))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing split")
    args = parser.parse_args()

    if not LABELS_ALL.is_dir():
        raise SystemExit(f"missing {LABELS_ALL} - run preprocessing first")

    label_files = sorted(LABELS_ALL.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not label_files:
        raise SystemExit(f"no label files in {LABELS_ALL}")

    out_dirs = [YOLO / "images" / s for s, _ in SPLIT_RATIOS] + [YOLO / "labels" / s for s, _ in SPLIT_RATIOS]
    existing = [d for d in out_dirs if d.is_dir() and any(d.iterdir())]
    if existing and not args.overwrite:
        raise SystemExit(f"split already exists ({existing[0]}) - pass --overwrite to rebuild")

    stems = [p.stem for p in label_files]
    labels_by_stem = {p.stem: p for p in label_files}
    parts = split_names(stems, args.seed)

    # sanity: no image in two splits, nothing lost
    seen = [s for part in parts.values() for s in part]
    assert len(seen) == len(set(seen)) == len(stems), "split is not a partition"

    per_split_images: Counter = Counter()
    per_split_boxes: Counter = Counter()
    class_images: dict[str, Counter] = {s: Counter() for s, _ in SPLIT_RATIOS}
    class_boxes: dict[str, Counter] = {s: Counter() for s, _ in SPLIT_RATIOS}
    class_stems: dict[str, dict[int, list[str]]] = {s: defaultdict(list) for s, _ in SPLIT_RATIOS}
    link_mode: Counter = Counter()

    for split, _ in SPLIT_RATIOS:
        (YOLO / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, _ in SPLIT_RATIOS:
        img_dir = YOLO / "images" / split
        lbl_dir = YOLO / "labels" / split
        for stem in parts[split]:
            src_img = RAW_IMAGES / f"{stem}.jpg"
            if not src_img.is_file():
                raise SystemExit(f"missing screenshot for label {stem}: {src_img}")
            link_mode[link_or_copy(src_img, img_dir / f"{stem}.jpg")] += 1
            shutil.copy2(labels_by_stem[stem], lbl_dir / f"{stem}.txt")

            ids = read_classes(labels_by_stem[stem])
            per_split_images[split] += 1
            per_split_boxes[split] += len(ids)
            for cid in set(ids):
                class_images[split][cid] += 1
                class_stems[split][cid].append(stem)
            for cid in ids:
                class_boxes[split][cid] += 1

    # one folder per class per split: which images contain that class
    for split, _ in SPLIT_RATIOS:
        for cid, name in enumerate(CLASSES):
            folder = YOLO / "class_index" / split / name
            folder.mkdir(parents=True, exist_ok=True)
            hits = sorted(class_stems[split][cid], key=lambda s: int(s) if s.isdigit() else s)
            (folder / "images.txt").write_text("\n".join(hits) + ("\n" if hits else ""), encoding="utf-8")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "class_counts_per_split.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "class_id", "class", "images_containing", "boxes"])
        for split, _ in SPLIT_RATIOS:
            for cid, name in enumerate(CLASSES):
                writer.writerow([split, cid, name, class_images[split][cid], class_boxes[split][cid]])

    summary = {
        "seed": args.seed,
        "ratios": {s: r for s, r in SPLIT_RATIOS},
        "source_images": len(stems),
        "images": {s: per_split_images[s] for s, _ in SPLIT_RATIOS},
        "boxes": {s: per_split_boxes[s] for s, _ in SPLIT_RATIOS},
        "link_modes": dict(link_mode),
    }
    (RESULTS / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # final data.yaml with the split paths. No `path:` key on purpose: Ultralytics then
    # resolves train/val/test relative to the yaml file itself, so the same file works
    # locally and in Colab/Kaggle (an absolute path breaks everywhere else).
    yaml_lines = [
        "# YOLO dataset - 15 RICO UI classes. Generated by scripts/split/make_split.py",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "nc: 15",
        "names:",
    ] + [f"  {i}: {name}" for i, name in enumerate(CLASSES)]
    (YOLO / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    total = len(stems)
    print(f"seed {args.seed} | images {total:,} | links: {dict(link_mode)}")
    print(f"\n{'split':<8}{'images':>10}{'%':>8}{'boxes':>12}{'boxes/img':>11}")
    for split, ratio in SPLIT_RATIOS:
        n = per_split_images[split]
        print(f"{split:<8}{n:>10,}{100 * n / total:>7.2f}%{per_split_boxes[split]:>12,}{per_split_boxes[split] / n:>11.1f}")
    print(f"\n{'class':<14}" + "".join(f"{s:>12}" for s, _ in SPLIT_RATIOS) + f"{'total':>12}")
    for cid, name in enumerate(CLASSES):
        row = [class_boxes[s][cid] for s, _ in SPLIT_RATIOS]
        print(f"{name:<14}" + "".join(f"{v:>12,}" for v in row) + f"{sum(row):>12,}")
    print(f"\nclass_index: {YOLO / 'class_index'} ({len(CLASSES)} folders x 3 splits)")
    print(f"summary    : {RESULTS}")


if __name__ == "__main__":
    main()
