#!/usr/bin/env python3
"""
Local DeepSets hyperparameter sweep for ModelNet.

Place this file at:
    Canonization/scripts/run_deepsets_sweep.py

The script is tracked in scripts/ so git pull provides it automatically.
Its checkpoints and local_sweeps/ results are ignored by Git.

Protocol:
- fixed stratified validation split from the training partition
- test split is NEVER evaluated during the sweep
- every configuration is run on seeds 0,1,2,3,4
- configurations are ranked by mean best validation accuracy
- sample standard deviation across the five seeds is reported

Run:
    python scripts/run_deepsets_sweep.py --dataset 10
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

SEEDS = [0, 1, 2, 3, 4]

CONFIGS = [
    {"name": "base"},
    {"name": "jitter", "apply_jitter": True},
    {"name": "scale", "apply_scale": True},
    {"name": "jitter_scale", "apply_jitter": True, "apply_scale": True},
    {"name": "scale_lr5e-4", "apply_scale": True, "lr": 5e-4},
    {"name": "scale_lr2e-3", "apply_scale": True, "lr": 2e-3},
    {"name": "jitter_scale_lr5e-4", "apply_jitter": True, "apply_scale": True, "lr": 5e-4},
    {"name": "jitter_scale_lr2e-3", "apply_jitter": True, "apply_scale": True, "lr": 2e-3},
    {"name": "scale_wd1e-6", "apply_scale": True, "weight_decay": 1e-6},
    {"name": "scale_wd1e-5", "apply_scale": True, "weight_decay": 1e-5},
    {"name": "scale_bs32", "apply_scale": True, "batch_size": 32},
    {"name": "scale_bs128", "apply_scale": True, "batch_size": 128},
    {"name": "scale_dropout005", "apply_scale": True, "dropout": 0.05},
    {
        "name": "scale_point_input_dropout005",
        "apply_scale": True,
        "point_dropout": 0.05,
        "input_dropout": 0.05,
    },
    {"name": "scale_fourier05", "apply_scale": True, "fourier_scale": 0.5},
    {"name": "scale_fourier20", "apply_scale": True, "fourier_scale": 2.0},
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["10", "40"], default="10")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--val-split-seed", type=int, default=2026)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rerun", action="store_true")
    return p.parse_args()


def add_override(cmd, key, value):
    cli_key = "--" + key
    if isinstance(value, bool):
        cmd.extend([cli_key, "true" if value else "false"])
    else:
        cmd.extend([cli_key, str(value)])


def summary_matches(summary, dataset):
    return (
        summary.get("dataset") == f"modelnet{dataset}"
        and summary.get("model") == "deepsets"
        and summary.get("ordering") == "ply"
        and summary.get("seeds") == SEEDS
        and isinstance(summary.get("val_acc_mean"), (int, float))
        and math.isfinite(summary["val_acc_mean"])
    )


def pct(x):
    if isinstance(x, (int, float)) and math.isfinite(x):
        return f"{100*x:.2f}"
    return "nan"


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    training_dir = repo_root / "ModelNet" / "training"
    if not (training_dir / "train.py").is_file():
        raise SystemExit(
            "Run this script from the Canonization checkout (scripts/run_deepsets_sweep.py)"
        )

    output_dir = repo_root / "local_sweeps" / f"deepsets_modelnet{args.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = CONFIGS[args.start:args.stop]
    if not selected:
        raise SystemExit("No configurations selected.")

    rows = []

    print("=" * 88)
    print(f"DeepSets sweep | ModelNet{args.dataset} | {len(selected)} configs | 5 seeds each")
    print(f"Validation fraction: {args.val_fraction:.2f} | split seed: {args.val_split_seed}")
    print("TEST SPLIT IS DISABLED DURING THIS SWEEP")
    print("=" * 88)

    for idx, cfg in enumerate(selected, start=args.start):
        name = cfg["name"]
        exp_name = f"local_sweep_mn{args.dataset}_{idx:02d}_{name}"
        summary_path = training_dir / "checkpoints" / exp_name / "summary.json"

        print(f"\n[{idx:02d}] {name}")
        print("     ", {k: v for k, v in cfg.items() if k != "name"} or {"recipe": "base"})

        summary = None
        if summary_path.is_file() and not args.rerun:
            try:
                candidate = json.loads(summary_path.read_text())
                if summary_matches(candidate, args.dataset):
                    summary = candidate
                    print("      using existing completed summary")
            except Exception:
                pass

        if summary is None:
            cmd = [
                sys.executable,
                "train.py",
                "--dataset", f"modelnet{args.dataset}",
                "--ordering", "ply",
                "--model", "deepsets",
                "--run_5_seeds", "true",
                "--seeds", *(str(s) for s in SEEDS),
                "--exp_name", exp_name,
                "--epochs", str(args.epochs),
                "--val_fraction", str(args.val_fraction),
                "--val_split_seed", str(args.val_split_seed),
                "--skip_test", "true",
            ]
            for key, value in cfg.items():
                if key != "name":
                    add_override(cmd, key, value)

            print("      $", " ".join(cmd))
            if args.dry_run:
                continue

            subprocess.run(cmd, cwd=training_dir, check=True)

            if not summary_path.is_file():
                raise RuntimeError(f"Missing summary: {summary_path}")
            summary = json.loads(summary_path.read_text())
            if not summary_matches(summary, args.dataset):
                raise RuntimeError(f"Unexpected summary: {summary_path}")

        runs = summary["runs"]
        best_epochs = [r["best_val_epoch"] for r in runs]
        train_accs = [r["train_acc"] for r in runs]

        row = {
            "index": idx,
            "name": name,
            "val_mean": summary["val_acc_mean"],
            "val_std": summary["val_acc_std"],
            "train_mean": sum(train_accs) / len(train_accs),
            "best_epoch_mean": sum(best_epochs) / len(best_epochs),
            "config": {k: v for k, v in cfg.items() if k != "name"},
        }
        rows.append(row)

        print(
            f"      VALIDATION: {pct(row['val_mean'])} ± {pct(row['val_std'])}%"
            f" | train={pct(row['train_mean'])}%"
            f" | mean best epoch={row['best_epoch_mean']:.1f}"
        )

    if args.dry_run:
        return

    rows.sort(key=lambda r: r["val_mean"], reverse=True)

    print("\n" + "=" * 96)
    print("FINAL RANKING — FIVE-SEED VALIDATION RESULTS")
    print("=" * 96)
    print(f"{'Rank':>4}  {'Config':<33} {'Val mean ± std':>20} {'Train':>9} {'Best ep.':>9}")
    print("-" * 96)
    for rank, row in enumerate(rows, 1):
        val = f"{pct(row['val_mean'])} ± {pct(row['val_std'])}"
        print(
            f"{rank:>4}  {row['name']:<33} {val:>20}"
            f" {pct(row['train_mean']):>8}% {row['best_epoch_mean']:>9.1f}"
        )

    json_path = output_dir / "ranking.json"
    csv_path = output_dir / "ranking.csv"
    json_path.write_text(json.dumps(rows, indent=2))

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank", "index", "name", "val_mean", "val_std",
                "train_mean", "best_epoch_mean", "config",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({
                "rank": rank,
                "index": row["index"],
                "name": row["name"],
                "val_mean": row["val_mean"],
                "val_std": row["val_std"],
                "train_mean": row["train_mean"],
                "best_epoch_mean": row["best_epoch_mean"],
                "config": json.dumps(row["config"], sort_keys=True),
            })

    print("\nBest configuration:")
    print(json.dumps(rows[0], indent=2))
    print(f"\nSaved local-only results:\n  {json_path}\n  {csv_path}")
    print("\nAfter choosing the winner, run the final five-seed TEST exactly once.")


if __name__ == "__main__":
    main()