"""Smoke test for Phase 4 models: build, forward, report size.

Runs on CPU on purpose, so the local machine can verify every model without a GPU.

Usage:
    python scripts/models/check_models.py
    python scripts/models/check_models.py --no-pretrained   # skip weight downloads
    python scripts/models/check_models.py --size 128 --batch 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from registry import ALL_MODELS, PRETRAINED, PRETRAINED_AVAILABLE, build_model, count_params  # noqa: E402


def check_yolo() -> None:
    """The detector comes from Ultralytics; check the configs we intend to use."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed - skipping detector check")
        return
    for cfg in ("yolo26n.yaml", "yolo11n.yaml"):
        try:
            model = YOLO(cfg)
            n = sum(p.numel() for p in model.model.parameters())
            print(f"{'YOLO ' + cfg:<22}{n / 1e6:>8.2f} M{'':>14}built from yaml")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{'YOLO ' + cfg:<22}{'':>10}FAILED: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128, help="square input size")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    print(f"input {args.batch}x3x{args.size}x{args.size}\n")
    print(f"{'model':<22}{'params':>10}{'pretrained':>13}{'forward':>10}")

    x = torch.randn(args.batch, 3, args.size, args.size)
    failures = 0
    for name in ALL_MODELS:
        try:
            model = build_model(name, num_classes=15, pretrained=not args.no_pretrained)
            model.eval()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(x)
            ms = (time.perf_counter() - t0) * 1000
            assert out.shape == (args.batch, 15), f"bad output shape {tuple(out.shape)}"
            src = PRETRAINED.get(name, "timm") if PRETRAINED_AVAILABLE[name] else "from scratch"
            if args.no_pretrained and PRETRAINED_AVAILABLE[name]:
                src = "random init"
            print(f"{name:<22}{count_params(model) / 1e6:>8.2f} M{src:>13}{ms:>8.0f} ms")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"{name:<22}{'':>10}{'':>13}FAILED: {type(exc).__name__}: {exc}")

    print()
    check_yolo()
    print(f"\nfailures: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
