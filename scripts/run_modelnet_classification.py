#!/usr/bin/env python3
"""Run all four ModelNet classification models by default."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "ModelNet" / "training"
RESULTS_ROOT = REPO_ROOT / "results" / "modelnet"

# Hilbert, Lex-Sort and unsorted MLP use GlobalMLPClassifier; DeepSets
# applies the same MLP per point, sums its logits and reuses the 'ply' config.
MODELS = (
    ("Hilbert", "hilbert"),
    ("Lex-Sort", "lex"),
    ("MLP", "ply"),
    ("DeepSets", "deepsets"),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run Hilbert, Lex-Sort, unsorted MLP and DeepSets "
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
        choices=["all", "hilbert", "lex", "ply", "deepsets"],
        default="all",
        help="Run all four models by default, or just one.",
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
    stem = f"{dataset}_results"
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



def print_results_table(dataset, rows):
    """Print a compact, aligned Unicode table with both evaluation metrics."""
    headers = ("Model", "Test accuracy (%)", "Generalization gap (pp)")
    values = [
        (
            row["model"],
            f"{100 * row['test_acc_mean']:.2f} ± {100 * row['test_acc_std']:.2f}",
            f"{100 * row['gen_gap_mean']:.2f} ± {100 * row['gen_gap_std']:.2f}",
        )
        for row in rows
    ]
    widths = [
        max(len(header), *(len(value[i]) for value in values))
        for i, header in enumerate(headers)
    ]
    span = sum(widths) + 2 * len(widths) + len(widths) - 1
    title = f"{dataset.upper()} RESULTS"
    if len({row["n_seeds"] for row in rows}) == 1:
        title += f" ({rows[0]['n_seeds']} SEEDS)"

    def rule(left, middle, right):
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def cells(items):
        return (
            "│ "
            + f"{items[0]:<{widths[0]}}"
            + " │ "
            + f"{items[1]:>{widths[1]}}"
            + " │ "
            + f"{items[2]:>{widths[2]}}"
            + " │"
        )

    print()
    print("┌" + "─" * span + "┐")
    print("│" + title.center(span) + "│")
    print(rule("├", "┬", "┤"))
    print(cells(headers))
    print(rule("├", "┼", "┤"))
    for value in values:
        print(cells(value))
    print(rule("└", "┴", "┘"))



def main(argv=None):
    args = parse_args(argv)
    dataset = f"modelnet{args.dataset}"
    models = [
        (label, selection)
        for label, selection in MODELS
        if args.ordering in ("all", selection)
    ]
    rows = []

    for label, selection in models:
        training_model = "deepsets" if selection == "deepsets" else "global_mlp"
        ordering = "ply" if selection == "deepsets" else selection
        # Keep the earlier DeepSets checkpoints intact for comparison.
        exp_name = (
            f"{dataset}_deepsets_reference_hps"
            if selection == "deepsets" else f"{dataset}_{selection}"
        )
        cmd = [
            sys.executable, "train.py",
            "--dataset", dataset,
            "--ordering", ordering,
            "--model", training_model,
            "--run_5_seeds", "true",
            "--seeds", *(str(seed) for seed in args.seeds),
            "--exp_name", exp_name,
        ]
        if args.epochs is not None:
            cmd.extend(("--epochs", str(args.epochs)))

        print(f"\n=== {dataset} / {label} ({len(args.seeds)} seeds) ===", flush=True)
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=TRAINING_DIR, check=True)

        summary_path = TRAINING_DIR / "checkpoints" / exp_name / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(
                f"Training finished, but its summary is missing: {summary_path}"
            )
        with summary_path.open() as f:
            summary = json.load(f)
        if (
            summary.get("dataset") != dataset
            or summary.get("ordering") != ordering
            or summary.get("model", training_model) != training_model
        ):
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

    write_table(dataset, rows)
    print_results_table(dataset, rows)


if __name__ == "__main__":
    main()
