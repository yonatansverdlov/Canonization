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
        description="Run all four ModelNet rotation/canonicalization models."
    )
    parser.add_argument(
        "--model",
        choices=["all", *MODELS],
        default="all",
        help="Run all four models (default), or select one.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models = MODELS if args.model == "all" else [args.model]

    for model in models:
        cmd = [
            sys.executable,
            "train_rot.py",
            "--model",
            model,
            "--dataset",
            "modelnet40",
            "--run_5_seeds",
            "true",
        ]

        print(f"\n=== ModelNet40: {model} (5 seeds) ===", flush=True)
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=TRAINING_DIR, check=True)

    print("\nAll requested ModelNet40 models completed.", flush=True)


if __name__ == "__main__":
    main()
