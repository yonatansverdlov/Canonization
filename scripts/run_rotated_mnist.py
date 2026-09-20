#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MNIST_DIR = REPO_ROOT / "mnist"


def main():
    parser = argparse.ArgumentParser(
        description="Run one Rotated MNIST training experiment."
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "average", "learned_can", "frozen_can"],
        required=True,
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "train.py",
        "--model_type",
        args.model,
        "--num_seeds",
        "5",
    ]

    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=MNIST_DIR, check=True)


if __name__ == "__main__":
    main()
