#!/usr/bin/env python3
"""Table 2: run Hilbert, Lex-Sort, and the unsorted MLP on one ModelNet dataset."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "ModelNet" / "training"
RESULTS_ROOT = REPO_ROOT / "results" / "modelnet"

# All three rows use the same GlobalMLPClassifier. The 'ply' ordering is the
# unsorted MLP baseline; 'lex' and 'hilbert' sort the input point cloud.
TABLE2_MODELS = (
    ("Hilbert", "hilbert"),
    ("Lex-Sort", "lex"),
    ("MLP", "ply"),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run ModelNet Table 2: Hilbert, Lex-Sort and unsorted MLP "
            "on one dataset, five seeds per model."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["10", "40"],
        required=True,
        help="ModelNet10 or ModelNet40.",
    )
    parser.add_argument(
        "--ordering",
        choices=["all", "hilbert", "lex", "ply"],
        default="all",
        help="Run all three Table 2 models (default), or just one.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Training seeds (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the trainer's default epoch count, if needed.",
    )
    return parser.parse_args(argv)


def write_table(dataset, rows):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"table2_{dataset}"
    json_path = RESULTS_ROOT / f"{stem}.json"
    csv_path = RESULTS_ROOT / f"{stem}.csv"
    payload = {"dataset": dataset, "metric_unit": "fraction", "rows": rows}
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    columns = (
        "dataset", "model", "ordering", "n_seeds",
        "test_acc_mean", "test_acc_std", "gen_gap_mean", "gen_gap_std",
    )
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({name: row[name] for name in columns} for row in rows)
    return json_path, csv_path


def main(argv=None):
    args = parse_args(argv)
    dataset = f"modelnet{args.dataset}"
    models = [
        (label, ordering)
        for label, ordering in TABLE2_MODELS
        if args.ordering in ("all", ordering)
    ]
    rows = []

    for label, ordering in models:
        exp_name = f"{dataset}_{ordering}"
        cmd = [
            sys.executable, "train.py",
            "--dataset", dataset,
            "--ordering", ordering,
            "--model", "global_mlp",
            "--run_5_seeds", "true",
            "--seeds", *(str(seed) for seed in args.seeds),
            "--exp_name", exp_name,
        ]
        if args.epochs is not None:
            cmd.extend(("--epochs", str(args.epochs)))

        print(f"\n=== Table 2: {dataset} / {label} ({len(args.seeds)} seeds) ===", flush=True)
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=TRAINING_DIR, check=True)

        summary_path = TRAINING_DIR / "checkpoints" / exp_name / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(
                f"Training finished, but its summary is missing: {summary_path}"
            )
        with summary_path.open() as f:
            summary = json.load(f)
        if summary.get("dataset") != dataset or summary.get("ordering") != ordering:
            raise RuntimeError(f"Unexpected or stale summary: {summary_path}")
        if summary.get("seeds") != args.seeds:
            raise RuntimeError(f"Summary seeds do not match this run: {summary_path}")

        rows.append({
            "dataset": dataset,
            "model": label,
            "ordering": ordering,
            "n_seeds": len(args.seeds),
            "test_acc_mean": summary["test_acc_mean"],
            "test_acc_std": summary["test_acc_std"],
            "gen_gap_mean": summary["gen_gap_mean"],
            "gen_gap_std": summary["gen_gap_std"],
        })
        # Preserve completed rows if a later model fails or is interrupted.
        write_table(dataset, rows)

    json_path, csv_path = write_table(dataset, rows)
    print(f"\n=== Table 2: {dataset} ===")
    for row in rows:
        print(
            f"{row['model']}: test acc "
            f"{100 * row['test_acc_mean']:.2f} ± "
            f"{100 * row['test_acc_std']:.2f}%"
        )
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
