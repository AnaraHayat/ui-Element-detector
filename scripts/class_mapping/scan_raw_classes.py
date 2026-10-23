"""Phase 1, step 2: scan every RICO JSON and count the raw `class` labels.

Reads all view hierarchies, walks the node tree and writes a CSV with, per raw
class: how many nodes use it and in how many screenshots it appears.

Usage:
    python scripts/class_mapping/scan_raw_classes.py
    python scripts/class_mapping/scan_raw_classes.py --json-dir data/raw/rico/unique_uis/combined
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_DIR = PROJECT_ROOT / "data" / "raw" / "rico" / "unique_uis" / "combined"
DEFAULT_OUT = PROJECT_ROOT / "results" / "class_mapping" / "raw_class_counts.csv"


def walk(node: dict | None, node_counts: Counter, seen: set[str]) -> None:
    """Count this node and recurse into its children. Some RICO children are null."""
    if not node:
        return
    cls = node.get("class")
    if cls:
        node_counts[cls] += 1
        seen.add(cls)
    for child in node.get("children") or ():
        walk(child, node_counts, seen)


def scan(json_dir: Path, show_progress: bool = True) -> tuple[Counter, Counter, int, int]:
    node_counts: Counter = Counter()
    file_counts: Counter = Counter()
    files_ok = 0
    files_bad = 0

    json_files = sorted(json_dir.glob("*.json"))
    for path in tqdm(json_files, desc="scanning JSON", unit="file", disable=not show_progress):
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            root = data.get("activity", {}).get("root")
        except (json.JSONDecodeError, OSError):
            files_bad += 1
            continue
        if not root:
            files_bad += 1
            continue
        seen: set[str] = set()
        walk(root, node_counts, seen)
        file_counts.update(seen)
        files_ok += 1

    return node_counts, file_counts, files_ok, files_bad


def write_csv(node_counts: Counter, file_counts: Counter, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["raw_class", "node_count", "screenshot_count"])
        for cls, n in node_counts.most_common():
            writer.writerow([cls, n, file_counts[cls]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top", type=int, default=40, help="rows to print")
    parser.add_argument("--progress", action="store_true", help="show a progress bar")
    args = parser.parse_args()

    if not args.json_dir.is_dir():
        raise SystemExit(f"JSON folder not found: {args.json_dir}")

    node_counts, file_counts, files_ok, files_bad = scan(args.json_dir, args.progress)
    write_csv(node_counts, file_counts, args.out)

    print(f"\nfiles read: {files_ok}  |  unreadable: {files_bad}")
    print(f"distinct raw classes: {len(node_counts)}  |  total nodes: {sum(node_counts.values())}")
    print(f"CSV written to: {args.out}\n")
    print(f"{'raw class':<62}{'nodes':>10}{'screens':>10}")
    print("-" * 82)
    for cls, n in node_counts.most_common(args.top):
        print(f"{cls:<62}{n:>10,}{file_counts[cls]:>10,}")
    if len(node_counts) > args.top:
        print(f"... {len(node_counts) - args.top} more rows in the CSV")


if __name__ == "__main__":
    main()
