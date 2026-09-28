#!/usr/bin/env python3
"""Run the DeepSets extension of Table 9 (ModelNet40 data scarcity)."""

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "ModelNet" / "training"
RESULTS_DIR = REPO_ROOT / "results" / "modelnet"

SEEDS = [0, 1, 2, 3, 4]
SETTINGS = (
    (1, 9840),
    (2, 4920),
    (4, 2460),
    (8, 1230),
)


def run_setting(stride, n_train):
    exp_name = f"modelnet40_table9_deepsets_n{n_train}"
    cmd = [
        sys.executable,
        "train.py",
        "--dataset", "modelnet40",
        "--ordering", "ply",
        "--model", "deepsets",
        "--dataset_stride", str(stride),
        "--run_5_seeds", "true",
        "--seeds", *(str(seed) for seed in SEEDS),
        "--exp_name", exp_name,

        # Fixed DeepSets recipe chosen on ModelNet10 validation.
        "--epochs", "150",
        "--batch_size", "32",
        "--apply_scale", "true",
        "--lr", "0.001",
        "--weight_decay", "1e-7",
        "--optimizer", "adam",
        "--adam_eps", "0.001",
        "--dropout", "0",
        "--point_dropout", "0",
        "--input_dropout", "0",
        "--fourier_scale", "1",
        "--label_smoothing", "0",
        "--scheduler", "multistep",
        "--lr_milestones", "60", "120",
        "--lr_gamma", "0.1",
        "--gradient_clip_val", "5",
    ]

    print(f"\n=== DeepSets / ModelNet40 / n={n_train} / stride={stride} ===", flush=True)
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=TRAINING_DIR, check=True)

    summary_path = TRAINING_DIR / "checkpoints" / exp_name / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Missing summary: {summary_path}")

    with summary_path.open() as f:
        summary = json.load(f)

    if summary.get("seeds") != SEEDS:
        raise RuntimeError(f"Unexpected seeds in {summary_path}")

    return {
        "training_samples": n_train,
        "dataset_stride": stride,
        "n_seeds": len(SEEDS),
        "test_acc_mean": summary["test_acc_mean"],
        "test_acc_std": summary["test_acc_std"],
        "gen_gap_mean": summary["gen_gap_mean"],
        "gen_gap_std": summary["gen_gap_std"],
    }


def print_table(rows):
    headers = ("Training samples", "DeepSets test acc (%)", "Gen. gap (pp)")
    values = [
        (
            str(row["training_samples"]),
            f"{100 * row['test_acc_mean']:.2f} ± {100 * row['test_acc_std']:.2f}",
            f"{100 * row['gen_gap_mean']:.2f} ± {100 * row['gen_gap_std']:.2f}",
        )
        for row in rows
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in values))
        for i in range(len(headers))
    ]

    def border(left, mid, right):
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def line(row):
        return "│ " + " │ ".join(
            f"{row[i]:>{widths[i]}}" for i in range(len(row))
        ) + " │"

    print()
    print(border("┌", "┬", "┐"))
    print(line(headers))
    print(border("├", "┼", "┤"))
    for row in values:
        print(line(row))
    print(border("└", "┴", "┘"))


def main():
    rows = [run_setting(stride, n_train) for stride, n_train in SETTINGS]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "table9_deepsets.json"
    csv_path = RESULTS_DIR / "table9_deepsets.csv"

    with json_path.open("w") as f:
        json.dump(
            {
                "dataset": "modelnet40",
                "model": "deepsets",
                "epochs": 150,
                "seeds": SEEDS,
                "rows": rows,
            },
            f,
            indent=2,
        )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print_table(rows)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
