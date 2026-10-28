"""Phase 6 driver: detector on the test split, then every classifier, GT + detected crops.

Resume-safe: a step is skipped when its output file already exists.

    python scripts/testing/run_all_tests.py            # everything
    python scripts/testing/run_all_tests.py --sources gt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
DETECTOR_OUT = ROOT / "results/testing/yolo26n/summary.csv"
MODELS = ["convnext_tiny", "convnextv2_tiny", "fastvit_sa12", "efficientvit_b1", "repvit_m1_5"]


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'=' * 76}\n{label}\n{'=' * 76}", flush=True)
    t0 = time.time()
    r = subprocess.run([PY, *cmd], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {label}")
    print(f"[{label}] finished in {(time.time() - t0) / 60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=["det", "gt"],
                        choices=["gt", "det"])
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if not DETECTOR_OUT.exists():
        run(["scripts/testing/run_detector_on_test.py"], "detector on test split")
    else:
        print("detector predictions already exist, skipping")

    for model in MODELS:
        for source in args.sources:
            out_csv = ROOT / "results/testing" / f"{model}_{source}_test.csv"
            if out_csv.exists():
                print(f"{model} [{source}] already tested, skipping")
                continue
            run(["scripts/testing/test_classifiers.py", "--model", model,
                 "--source", source, "--num-workers", str(args.num_workers)],
                f"classifier {model} ({source} crops)")

    print("\nPhase 6 complete: all prediction files are in results/testing/")


if __name__ == "__main__":
    main()