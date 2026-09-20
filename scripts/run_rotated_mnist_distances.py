#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MNIST_DIR = REPO_ROOT / "mnist"


def main():
    parser = argparse.ArgumentParser(
        description="Run the Rotated MNIST distance experiment."
    )
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="test",
    )
    parser.add_argument(
        "--reduce",
        choices=["average", "max"],
        default="average",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "compute_distances.py",
        "--split",
        args.split,
        "--reduce_mode",
        args.reduce,
        "--seed",
        str(args.seed),
    ]

    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=MNIST_DIR, check=True)


if __name__ == "__main__":
    main()
