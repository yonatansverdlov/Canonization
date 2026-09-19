#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "ModelNet" / "training"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ModelNet classification experiments."
    )
    parser.add_argument(
        "--dataset",
        choices=["10", "40"],
        required=True,
        help="ModelNet10 or ModelNet40.",
    )
    parser.add_argument(
        "--ordering",
        choices=["all", "ply", "lex", "hilbert"],
        default="all",
        help="Ordering method to run. Default: all.",
    )
    return parser.parse_args()


def run(cmd):
    print()
    print("$", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=TRAINING_DIR, check=True)


def main():
    args = parse_args()

    dataset = f"modelnet{args.dataset}"
    orderings = (
        ["ply", "lex", "hilbert"]
        if args.ordering == "all"
        else [args.ordering]
    )

    for ordering in orderings:
        print()
        print(
            f"=== {dataset}: {ordering} "
            "(5 seeds) ==="
        )

        run(
            [
                sys.executable,
                "train.py",
                "--dataset",
                dataset,
                "--ordering",
                ordering,
                "--run_5_seeds",
                "true",
                "--exp_name",
                f"{dataset}_{ordering}",
            ]
        )


if __name__ == "__main__":
    main()
