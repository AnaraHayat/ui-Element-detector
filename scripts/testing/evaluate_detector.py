"""Phase 7: detection metrics for the YOLO26n detector from its saved test predictions.

Standard detection evaluation at IoU 0.5, one-to-one greedy matching (predictions sorted
by confidence): per-class precision / recall / F1 at the operating point, AP@0.5 per class
(all-point interpolation) with mAP@0.5, and the detection confusion matrix (matched pairs,
plus "missed" ground truth and false alarms). Cohen's kappa is computed over matched pairs
(unmatched predictions are false alarms and excluded, as in the literature).

Usage:
    python scripts/testing/evaluate_detector.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from class_mapping.class_map import CLASSES  # noqa: E402

K = len(CLASSES)


def read_labels(path: Path, with_conf: bool = False):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
            conf = float(parts[5]) if with_conf and len(parts) >= 6 else 1.0
            out.append((c, cx, cy, w, h, conf))
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det-dir", default="results/testing/yolo26n/labels")
    ap.add_argument("--gt-labels", default="data/yolo/labels/test")
    ap.add_argument("--out", default="results/evaluation/detector")
    ap.add_argument("--conf", type=float, default=0.25, help="operating point for P/R/F1")
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    args = ap.parse_args()

    det_dir = PROJECT_ROOT / args.det_dir
    gt_dir = PROJECT_ROOT / args.gt_labels
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    # pass 1: one-to-one matches at every confidence (for AP), plus op-point counts
    records = []          # (conf, is_tp, pred_cls, matched_gt_cls) for every prediction
    tp_op = np.zeros(K)   # matched at the operating point
    fp_op = np.zeros(K)
    fn_op = np.zeros(K)   # ground-truth boxes nothing matched

    det_files = sorted(det_dir.glob("*.txt"))
    print(f"matching {len(det_files):,} images (IoU {args.iou_thresh}) ...", flush=True)

    for df in det_files:
        gt = read_labels(gt_dir / df.name)
        preds = sorted(read_labels(df, with_conf=True), key=lambda p: -p[5])
        matched = [False] * len(gt)
        for dcls, dcx, dcy, dw, dh, dconf in preds:
            best_iou, best_j = 0.0, -1
            for j, (gcls, gcx, gcy, gw, gh, _c) in enumerate(gt):
                if matched[j] or gcls != dcls:
                    continue
                v = iou((dcx, dcy, dw, dh), (gcx, gcy, gw, gh))
                if v > best_iou:
                    best_iou, best_j = v, j
            is_tp = best_j >= 0 and best_iou >= args.iou_thresh
            records.append((dconf, is_tp, dcls, gt[best_j][0] if is_tp else -1))
            if is_tp:
                matched[best_j] = True
                if dconf >= args.conf:
                    tp_op[dcls] += 1
            elif dconf >= args.conf:
                fp_op[dcls] += 1
        for j, (gcls, *_rest) in enumerate(gt):
            if not matched[j]:
                fn_op[gcls] += 1

    # AP@0.5 per class (all-point interpolation over the full confidence range)
    recs = pd.DataFrame(records, columns=["conf", "is_tp", "det_cls", "gt_cls"])
    ap = np.full(K, np.nan)
    for k in range(K):
        sub = recs[recs["det_cls"] == k].sort_values("conf", ascending=False)
        n_gt = tp_op[k] + fn_op[k]
        if n_gt == 0 or sub.empty:
            ap[k] = np.nan if n_gt == 0 else 0.0
            continue
        tp = sub["is_tp"].to_numpy().cumsum()
        fp = (~sub["is_tp"].to_numpy()).cumsum()
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / n_gt
        # precision envelope, then integrate over recall steps
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        r_prev, p_sum = 0.0, 0.0
        for p_i, r_i in zip(precision, recall):
            if r_i > r_prev:
                p_sum += p_i * (r_i - r_prev)
                r_prev = r_i
        ap[k] = p_sum

    per_class = pd.DataFrame({
        "class": CLASSES,
        "gt_boxes": (tp_op + fn_op).astype(int),
        "pred_boxes@conf": (tp_op + fp_op).astype(int),
        "precision@conf": tp_op / np.maximum(tp_op + fp_op, 1),
        "recall@conf": tp_op / np.maximum(tp_op + fn_op, 1),
        "f1@conf": 2 * tp_op / np.maximum(2 * tp_op + fp_op + fn_op, 1),
        "ap50": ap,
    })
    per_class.to_csv(out / "detector_per_class.csv", index=False)

    macro = {
        "mAP50": float(np.nanmean(ap)),
        "precision_macro@conf": float(np.mean(per_class["precision@conf"])),
        "recall_macro@conf": float(np.mean(per_class["recall@conf"])),
        "f1_macro@conf": float(np.mean(per_class["f1@conf"])),
        "conf_threshold": args.conf, "iou_threshold": args.iou_thresh,
        "images": len(det_files),
    }
    pd.DataFrame([macro]).to_csv(out / "detector_summary.csv", index=False)

    # pass 1b: class-AGNOSTIC matching (same confidence order) for the confusion matrix,
    # so cross-class mistakes (Box detected as Text, ...) are visible. Per-class matching
    # above stays the standard for AP/P/R.
    cm = np.zeros((K, K), dtype=int)
    missed = np.zeros(K, dtype=int)
    y_true, y_pred = [], []
    for df in det_files:
        gt = read_labels(gt_dir / df.name)
        preds = sorted(read_labels(df, with_conf=True), key=lambda p: -p[5])
        matched = [False] * len(gt)
        for dcls, dcx, dcy, dw, dh, dconf in preds:
            best_iou, best_j = 0.0, -1
            for j, (gcls, gcx, gcy, gw, gh, _c) in enumerate(gt):
                if matched[j]:
                    continue
                v = iou((dcx, dcy, dw, dh), (gcx, gcy, gw, gh))
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and best_iou >= args.iou_thresh:
                matched[best_j] = True
                cm[gt[best_j][0], dcls] += 1
                y_true.append(gt[best_j][0])
                y_pred.append(dcls)
        for j, (gcls, *_rest) in enumerate(gt):
            if not matched[j]:
                missed[gcls] += 1
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    cm_df["MISSED"] = missed
    cm_df.to_csv(out / "detector_confusion.csv")

    # Cohen's kappa over the class-agnostic matched pairs
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(y_true, y_pred)

    matched_total = int(cm.sum())
    gt_total = int(cm.sum() + missed.sum())
    print(pd.DataFrame([macro]).round(4).to_string(index=False))
    print(f"kappa over matched pairs: {kappa:.4f}")
    print(f"matched GT boxes: {matched_total:,} / {gt_total:,} ({matched_total / gt_total:.1%})")
    print(f"per-class:\n{per_class.round(3).to_string(index=False)}")
    print(f"\nall files written to {out}/")


if __name__ == "__main__":
    main()
