#!/usr/bin/env python3

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MNIST_DIR = REPO_ROOT / "mnist"
DATA_DIR = MNIST_DIR / "data" / "rotated_mnist"

sys.path.insert(0, str(MNIST_DIR))

from utils.data_funcs import obtain


def main():
    print(f"Rotated MNIST data directory: {DATA_DIR}")
    obtain(str(DATA_DIR))


if __name__ == "__main__":
    main()
