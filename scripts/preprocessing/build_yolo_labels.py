"""Phase 2 - RICO JSON view hierarchies -> YOLO label files.

For every screenshot:
  * walk the view tree,
  * map each raw class with class_map.py (unmapped = dropped),
  * clip the box to the screenshot (user decision: clip, do not drop),
  * drop boxes that are not a rectangle with positive area after clipping,
  * write one `<id>.txt` line per surviving element: `cls cx cy w h`, normalized,
  * skip the screenshot completely if no line survives.

No images are copied, resized or modified. Only label files are written.

Usage:
    python scripts/preprocessing/build_yolo_labels.py
    python scripts/preprocessing/build_yolo_labels.py --limit 200 --out-dir data/yolo/labels/debug
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "class_mapping"))

from class_map import CLASSES, CLASS_TO_ID, map_label  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "rico" / "unique_uis" / "combined"
LABELS_DIR = PROJECT_ROOT / "data" / "yolo" / "labels" / "all"
DATA_YAML = PROJECT_ROOT / "data" / "yolo" / "data.yaml"
REPORT = PROJECT_ROOT / "results" / "preprocessing" / "preprocess_report.json"


def to_yolo(bounds, width: int, height: int) -> tuple[tuple[float, float, float, float] | None, str | None]:
    """Pixel bounds [x1, y1, x2, y2] -> normalized (cx, cy, w, h), or (None, reason)."""
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None, "bad_bounds"
    try:
        x1, y1, x2, y2 = (float(v) for v in bounds)
    except (TypeError, ValueError):
        return None, "bad_bounds"
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None, "bad_bounds"
    # reversed coordinates = corrupted entry, not a rectangle (docs: drop negative size)
    if x2 < x1 or y2 < y1:
        return None, "negative_size"

    # clip to the screenshot (decision: clip the part outside, keep the element)
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))

    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None, "zero_size"
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, box_w / width, box_h / height), None


def iter_nodes(root: dict):
    """Depth-first walk, iterative so deep trees cannot hit the recursion limit."""
    stack = [root]
    while stack:
        node = stack.pop()
        if not node:
            continue
        yield node
        for child in node.get("children") or ():
            stack.append(child)


def process_one(json_path: Path, jpg_path: Path | None) -> tuple[list[str], Counter, Counter, str | None]:
    """Return (label lines, class counter, drop counter, skip reason)."""
    lines: list[str] = []
    per_class: Counter = Counter()
    dropped: Counter = Counter()

    try:
        with json_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return [], per_class, dropped, "bad_json"

    root = data.get("activity", {}).get("root")
    if not root:
        return [], per_class, dropped, "no_root"

    if jpg_path is None or not jpg_path.is_file():
        return [], per_class, dropped, "no_image"
    try:
        with Image.open(jpg_path) as img:
            width, height = img.size
    except OSError:
        return [], per_class, dropped, "bad_image"
    if width <= 0 or height <= 0:
        return [], per_class, dropped, "bad_image"

    for node in iter_nodes(root):
        name, reason = map_label(node.get("class", ""))
        if name is None:
            dropped[reason] += 1
            continue
        box, bad = to_yolo(node.get("bounds"), width, height)
        if box is None:
            dropped[bad] += 1
            continue
        cx, cy, bw, bh = box
        lines.append(f"{CLASS_TO_ID[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        per_class[name] += 1

    if not lines:
        return [], per_class, dropped, "no_boxes"
    return lines, per_class, dropped, None


def write_data_yaml(path: Path) -> None:
    lines = [
        "# YOLO dataset config - 15 RICO UI classes.",
        "# train/val/test paths are added in Phase 3 together with the split.",
        f"path: {(path.parent).as_posix()}",
        "nc: 15",
        "names:",
    ] + [f"  {i}: {name}" for i, name in enumerate(CLASSES)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--limit", type=int, default=0, help="only the first N JSONs (debug)")
    parser.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty out-dir")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if not args.raw_dir.is_dir():
        raise SystemExit(f"raw folder not found: {args.raw_dir}")
    if args.out_dir.is_dir() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out_dir} is not empty - pass --overwrite to replace the label files")

    json_files = sorted(args.raw_dir.glob("*.json"))
    if args.limit:
        json_files = json_files[: args.limit]
    if not json_files:
        raise SystemExit(f"no *.json in {args.raw_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    class_counts: Counter = Counter()
    drop_counts: Counter = Counter()
    skip_counts: Counter = Counter()
    images_kept = 0
    boxes_written = 0

    for json_path in tqdm(json_files, desc="building labels", unit="img", disable=not args.progress):
        lines, per_class, dropped, skip = process_one(json_path, json_path.with_suffix(".jpg"))
        class_counts.update(per_class)
        drop_counts.update(dropped)
        if skip:
            skip_counts[skip] += 1
            continue
        (args.out_dir / f"{json_path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        images_kept += 1
        boxes_written += len(lines)

    report = {
        "images_scanned": len(json_files),
        "images_kept": images_kept,
        "images_skipped": dict(skip_counts),
        "boxes_written": boxes_written,
        "boxes_dropped": dict(drop_counts),
        "class_counts": {name: class_counts.get(name, 0) for name in CLASSES},
    }

    print(f"\nimages scanned   : {len(json_files):,}")
    print(f"images kept      : {images_kept:,} ({100 * images_kept / len(json_files):.2f}%)")
    for reason, n in skip_counts.most_common():
        print(f"   skipped        : {reason:<10}{n:>8,}")
    print(f"boxes written    : {boxes_written:,}  ({boxes_written / max(images_kept, 1):.1f} per image)")
    for reason, n in drop_counts.most_common():
        print(f"   box dropped    : {reason:<10}{n:>10,}")
    print(f"\n{'class':<14}{'id':>3}{'boxes':>12}{'% boxes':>9}")
    for i, name in enumerate(CLASSES):
        n = class_counts.get(name, 0)
        print(f"{name:<14}{i:>3}{n:>12,}{100 * n / max(boxes_written, 1):>8.2f}%")
    print(f"\nlabels: {args.out_dir}")

    if not args.limit:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_data_yaml(DATA_YAML)
        print(f"report: {REPORT}")
        print(f"yaml  : {DATA_YAML}")


if __name__ == "__main__":
    main()
