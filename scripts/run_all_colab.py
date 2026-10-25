"""Run the whole training pipeline from one command (for Colab, or any GPU box).

Equivalent to the notebook, but as a plain script so it can be launched with
    !python scripts/run_all_colab.py --data-zip /content/drive/MyDrive/ui-data.zip

Steps: extract data (optional) -> verify dataset -> smoke test -> train CNNs -> train YOLO -> zip results.

Usage:
    python scripts/run_all_colab.py --data-zip /content/drive/MyDrive/ui-data.zip
    python scripts/run_all_colab.py --models densenet121 xception41 --no-yolo
    python scripts/run_all_colab.py --skip-data --skip-smoke --models pyramidnet110
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALL_CNNS = ["densenet121", "xception41", "seresnet50", "pyramidnet110", "residual_attention"]


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"FAILED ({result.returncode}): {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-zip", type=Path, default=None, help="ui-data.zip to extract into data/")
    parser.add_argument("--skip-data", action="store_true", help="assume data/yolo already exists")
    parser.add_argument("--models", nargs="+", default=ALL_CNNS, help="classifiers to train")
    parser.add_argument("--yolo", default="yolo26n", help="YOLO config name, empty string to skip")
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs for every model")
    parser.add_argument("--batch-size", type=int, default=None, help="override batch size for every CNN")
    parser.add_argument("--zip-out", type=Path, default=None, help="write results zip here when finished")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    data_yaml = PROJECT_ROOT / "data" / "yolo" / "data.yaml"

    if args.data_zip and not args.skip_data:
        if data_yaml.exists():
            print(f"dataset already present at {data_yaml} - not extracting again")
        else:
            print(f"extracting {args.data_zip} (~7.8 GB, several minutes)...", flush=True)
            with zipfile.ZipFile(args.data_zip) as zf:
                zf.extractall(PROJECT_ROOT / "data")
            print("extracted")

    if not data_yaml.exists():
        raise SystemExit("data/yolo/data.yaml missing - pass --data-zip or build the dataset first")

    # dataset sanity: images match labels, classes defined
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    counts = {split: len(list((PROJECT_ROOT / "data" / "yolo" / "images" / split).glob("*.jpg")))
              for split in ("train", "val", "test")}
    print(f"\nclasses: {cfg['nc']} | images: {counts}")
    if not all(counts.values()):
        raise SystemExit("a split is empty - the extraction failed")

    if not args.skip_smoke:
        run([args.python, "scripts/models/check_models.py", "--size", "128"], "smoke: model definitions")
        run([args.python, "scripts/training/train_cnn.py", "--model", args.models[0],
             "--epochs", "1", "--max-crops", "500", "--val-per-class", "20",
             "--batch-size", "32", "--out-suffix", "_smoke"], f"smoke: train {args.models[0]} 1 epoch")
        if args.smoke_only:
            print("\nsmoke run finished (--smoke-only)")
            return

    for model in args.models:
        cmd = [args.python, "scripts/training/train_cnn.py", "--model", model]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        if args.batch_size:
            cmd += ["--batch-size", str(args.batch_size)]
        run(cmd, f"train classifier: {model}")

    if not args.no_yolo and args.yolo:
        cmd = [args.python, "scripts/training/train_yolo.py", "--model", args.yolo]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        run(cmd, f"train detector: {args.yolo}")

    if args.zip_out:
        args.zip_out.parent.mkdir(parents=True, exist_ok=True)
        print(f"\npacking results -> {args.zip_out}")
        with zipfile.ZipFile(args.zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in (PROJECT_ROOT / "results").rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(PROJECT_ROOT / "results"))
        print("done")

    print("\nall requested runs finished")


if __name__ == "__main__":
    main()
