#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "ModelNet" / "training"
RESULTS_ROOT = REPO_ROOT / "results" / "modelnet" / "canonization"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from experiment_tables import format_mean_std, print_table

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
    rows = []

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

        summary_path = RESULTS_ROOT / f"{model}_summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"Missing result summary: {summary_path}")
        with summary_path.open() as stream:
            summary = json.load(stream)
        if (
            summary.get("dataset") != "modelnet40"
            or summary.get("model") != model
            or summary.get("seeds") != [0, 1, 2, 3, 4]
        ):
            raise RuntimeError(f"Unexpected result summary: {summary_path}")
        rows.append((
            model,
            format_mean_std(summary["test_acc_mean"], summary["test_acc_std"]),
            format_mean_std(summary["gen_gap_mean"], summary["gen_gap_std"]),
        ))

    print_table(
        "MODELNET40 CANONIZATION RESULTS (5 SEEDS)",
        ("Model", "Test accuracy (%)", "Generalization gap (pp)"),
        rows,
        right_align=(1, 2),
    )


if __name__ == "__main__":
    main()
