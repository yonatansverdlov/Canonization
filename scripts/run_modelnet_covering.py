#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELNET_ROOT = REPO_ROOT / "ModelNet"
DATA_CREATION_DIR = MODELNET_ROOT / "data_creation"
DISTANCE_DIR = MODELNET_ROOT / "compute_distances"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the ModelNet covering-number experiment."
    )
    parser.add_argument(
        "--dataset",
        choices=["10", "40"],
        required=True,
        help="ModelNet10 or ModelNet40.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=256,
        help="Number of sampled points per shape.",
    )
    parser.add_argument(
        "--reduce",
        choices=["average", "max"],
        default="average",
        help="Dataset-level reduction used by the distance computation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the cached point clouds even if they already exist.",
    )
    return parser.parse_args()


def cache_paths(dataset: str, points: int):
    stem = f"modelnet{dataset}"
    data_dir = DATA_CREATION_DIR / "data"
    train = data_dir / (
        f"{stem}_train_P{points}_hilbm12_norm_with_perms.pt"
    )
    test = data_dir / (
        f"{stem}_test_P{points}_hilbm12_norm_with_perms.pt"
    )
    return train, test


def run(cmd, cwd):
    print()
    print("$", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    args = parse_args()
    train_cache, test_cache = cache_paths(args.dataset, args.points)

    if args.force or not (train_cache.exists() and test_cache.exists()):
        print("Creating point-cloud caches...")
        run(
            [
                sys.executable,
                "create_data.py",
                "--dataset_name",
                args.dataset,
                "--P",
                str(args.points),
            ],
            DATA_CREATION_DIR,
        )
    else:
        print("Point-cloud caches already exist; skipping data creation.")

    print("Computing distances...")
    run(
        [
            sys.executable,
            "compute_distance.py",
            "--dataset_name",
            args.dataset,
            "--P",
            str(args.points),
            "--dataset_reduce",
            args.reduce,
        ],
        DISTANCE_DIR,
    )


if __name__ == "__main__":
    main()
