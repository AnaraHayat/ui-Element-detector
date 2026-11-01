"""Phase 6 step 2: run the trained YOLO26n detector on the test split.

Writes, for each test image, one prediction file in YOLO format
    <class> <cx> <cy> <w> <h> <confidence>
plus a per-image summary. Phase 7 turns these into detection metrics.

Resume-safe: images that already have a prediction file are skipped, so a run that
dies (GPU OOM, quota, kernel restart) can simply be started again. Run it in chunks
(--max-images 1000) from a loop if you want progress to survive every few minutes.

Usage:
    python scripts/testing/run_detector_on_test.py                  # full run, resumes
    python scripts/testing/run_detector_on_test.py --max-images 1000 # one chunk
    python scripts/testing/run_detector_on_test.py --max-images 8    # smoke
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def device_is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="results/yolo26n/weights/best.pt")
    parser.add_argument("--images", default="data/yolo/images/test")
    parser.add_argument("--out", default="results/testing/yolo26n")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=8,
                        help="images per GPU batch (auto-halved on CUDA OOM)")
    parser.add_argument("--max-images", type=int, default=0,
                        help="process at most N not-yet-done images (0 = all)")
    parser.add_argument("--no-resume", action="store_true", help="redo every image")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from ultralytics import YOLO

    all_images = sorted((PROJECT_ROOT / args.images).glob("*.jpg"))
    assert all_images, f"no test images under {args.images}"

    out = PROJECT_ROOT / args.out
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    done = set() if args.no_resume else {p.stem for p in labels_dir.glob("*.txt")}
    todo = [p for p in all_images if p.stem not in done]
    if args.max_images:
        todo = todo[: args.max_images]

    print(f"detector {args.weights} | {len(all_images):,} test images | "
          f"already done {len(done):,} | this run {len(todo):,} | "
          f"imgsz {args.imgsz} conf {args.conf} batch {args.batch} device {args.device}",
          flush=True)

    total_boxes, failed = 0, 0
    written = 0
    if todo:
        import gc

        import torch

        batch = args.batch
        model = YOLO(str(PROJECT_ROOT / args.weights))
        while todo and batch >= 1:
            try:
                if device_is_cuda(args.device):
                    torch.cuda.empty_cache()
                for r in model.predict(
                        source=[str(p) for p in todo], imgsz=args.imgsz, conf=args.conf,
                        batch=batch, device=args.device, stream=True, verbose=False):
                    stem = Path(r.path).stem
                    boxes = r.boxes
                    lines = []
                    if boxes is not None:
                        for (cx, cy, w, h), c, p in zip(boxes.xywhn.tolist(),
                                                        boxes.cls.tolist(),
                                                        boxes.conf.tolist()):
                            lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {p:.4f}")
                    (labels_dir / f"{stem}.txt").write_text(
                        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                    total_boxes += len(lines)
                    written += 1

                    if written % 200 == 0:
                        print(f"  {len(done) + written:,}/{len(all_images):,} images done",
                              flush=True)
                break  # finished this chunk
            except Exception as exc:  # noqa: BLE001
                oom = "out of memory" in str(exc).lower() or \
                    type(exc).__name__ == "OutOfMemoryError"
                kept = len(list(labels_dir.glob("*.txt")))
                if oom and batch > 1:
                    batch = max(1, batch // 2)
                    print(f"\nCUDA OOM - retrying with batch {batch} "
                          f"({kept:,} images already saved)", flush=True)
                    gc.collect()
                    if device_is_cuda(args.device):
                        torch.cuda.empty_cache()
                    todo = [p for p in todo if not (labels_dir / f"{p.stem}.txt").exists()]
                    continue
                failed += 1
                print("\nDETECTOR RUN FAILED - traceback:", flush=True)
                traceback.print_exc()
                print(f"\nprogress kept: {kept:,} images. "
                      f"Run the same command again to continue.", flush=True)
                break

    # summary always rebuilt from what is on disk (cheap, and correct after resumes)
    rows = []
    for p in sorted(labels_dir.glob("*.txt")):
        rows.append({"image": p.stem,
                     "boxes": sum(1 for line in p.read_text().splitlines() if line.strip())})
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "boxes"])
        writer.writeheader()
        writer.writerows(rows)

    total = sum(r["boxes"] for r in rows)
    print(f"labels on disk: {len(rows):,}/{len(all_images):,} images, "
          f"{total:,} predicted boxes ({total / max(len(rows), 1):.1f} per image)",
          flush=True)
    print(f"this run: {len(todo):,} images, {total_boxes:,} boxes -> {out}", flush=True)

    if len(rows) < len(all_images):
        print(f"INCOMPLETE: {len(all_images) - len(rows):,} images left - run again",
              flush=True)
        raise SystemExit(2)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
