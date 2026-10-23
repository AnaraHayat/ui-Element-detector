"""Aggregate the raw class counts through class_map.py -> counts per project class.

Reads results/preprocessing/raw_class_counts.csv (output of scan_raw_classes.py)
so the 6.6k JSON files do not need to be parsed again.

Usage:
    python scripts/dataset/count_mapped_classes.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from class_map import CLASSES, map_label  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = PROJECT_ROOT / "results" / "preprocessing" / "raw_class_counts.csv"
DEFAULT_OUT = PROJECT_ROOT / "results" / "preprocessing" / "mapped_class_counts.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-csv", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.in_csv.is_file():
        raise SystemExit(f"missing {args.in_csv} - run scan_raw_classes.py first")

    mapped: Counter = Counter()
    dropped: Counter = Counter()
    labels_per_class: Counter = Counter()
    total = 0

    with args.in_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw, n = row["raw_class"], int(row["node_count"])
            total += n
            name, reason = map_label(raw)
            if name is None:
                dropped[reason] += n
            else:
                mapped[name] += n
                labels_per_class[name] += 1

    kept = sum(mapped.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "class_id", "element_count", "pct_of_kept", "raw_labels"])
        for i, name in enumerate(CLASSES):
            n = mapped.get(name, 0)
            writer.writerow([name, i, n, round(100 * n / kept, 3), labels_per_class[name]])

    print(f"total nodes      : {total:,}")
    print(f"kept             : {kept:,} ({100 * kept / total:.2f}%)")
    print(f"dropped          : {total - kept:,} ({100 * (total - kept) / total:.2f}%)")
    for reason, n in dropped.most_common():
        print(f"   {reason:<12}{n:>10,} ({100 * n / total:5.2f}%)")
    print(f"\n{'class':<14}{'id':>3}{'elements':>12}{'% kept':>9}   ratio vs smallest")
    smallest = min(mapped.values())
    for i, name in enumerate(CLASSES):
        n = mapped.get(name, 0)
        print(f"{name:<14}{i:>3}{n:>12,}{100 * n / kept:>8.2f}%{n / smallest:>10.1f}x")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
