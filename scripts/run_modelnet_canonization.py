#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "ModelNet" / "training"

MODELS = [
    "PurePCA",
    "FrameAveraging",
    "Skewness",
    "RandomFrame",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one ModelNet rotation/canonicalization experiment."
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        required=True,
        help="Model to run.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=200,
        help="Stop after this many epochs without train-loss improvement. Use 0 to disable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cmd = [
        sys.executable,
        "train_rot.py",
        "--model",
        args.model,
        "--dataset",
        "modelnet40",
        "--run_5_seeds",
        "true",
        "--early_stop_patience",
        str(args.early_stop_patience),
    ]

    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=TRAINING_DIR, check=True)


if __name__ == "__main__":
    main()
