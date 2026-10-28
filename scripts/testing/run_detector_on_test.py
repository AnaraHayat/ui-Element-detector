"""Phase 6 step 2: run the trained YOLO26n detector on the test split.

Writes, for each of the 13,208 test images, one prediction file in YOLO format
    <class> <cx> <cy> <w> <h> <confidence>
plus a per-image summary. Phase 7 turns these into detection metrics.

Usage:
    python scripts/testing/run_detector_on_test.py                  # full run
    python scripts/testing/run_detector_on_test.py --max-images 20  # smoke
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from class_mapping.class_map import CLASSES  # noqa: E402  (canonical 15-class order)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="results/yolo26n/weights/best.pt")
    parser.add_argument("--images", default="data/yolo/images/test")
    parser.add_argument("--out", default="results/testing/yolo26n")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=0, help="0 = all")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from ultralytics import YOLO

    images = sorted((PROJECT_ROOT / args.images).glob("*.jpg"))
    if args.max_images:
        images = images[: args.max_images]
    assert images, f"no test images under {args.images}"

    out = PROJECT_ROOT / args.out
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(PROJECT_ROOT / args.weights))
    print(f"detector {args.weights} | {len(images):,} test images | "
          f"imgsz {args.imgsz} conf {args.conf} device {args.device}", flush=True)

    summary_rows = []
    total_boxes = 0
    for i, r in enumerate(model.predict(
            source=[str(p) for p in images], imgsz=args.imgsz, conf=args.conf,
            device=args.device, stream=True, verbose=False)):
        stem = images[i].stem
        boxes = r.boxes
        lines = []
        if boxes is not None:
            xywhn = boxes.xywhn.tolist()          # normalized cx, cy, w, h
            cls = boxes.cls.tolist()
            conf = boxes.conf.tolist()
            for (cx, cy, w, h), c, p in zip(xywhn, cls, conf):
                lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {p:.4f}")
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                encoding="utf-8")
        total_boxes += len(lines)
        summary_rows.append({"image": stem, "boxes": len(lines)})

        if (i + 1) % 500 == 0:
            print(f"  {i + 1:,}/{len(images):,} images, {total_boxes:,} boxes", flush=True)

    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "boxes"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"done: {len(images):,} images, {total_boxes:,} predicted boxes "
          f"({total_boxes / max(len(images), 1):.1f} per image) -> {out}", flush=True)


if __name__ == "__main__":
    main()