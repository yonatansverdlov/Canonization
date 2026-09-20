from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from dws.data import (
    DWSProcessedDataset,
    collate_weight_space,
    get_statistics,
    seed_worker,
)
from dws.models import (
    DWSModelForClassification,
    MLPModelForClassification,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "dws"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    torch.use_deterministic_algorithms(True)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available.")

    return torch.device(device_arg)


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
):
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        collate_fn=collate_weight_space,
    )


def build_model(
    model_name: str,
    example,
    device: torch.device,
):
    weight_shapes = tuple(
        tuple(weight.shape[:2])
        for weight in example.weights
    )
    bias_shapes = tuple(
        tuple(bias.shape[:1])
        for bias in example.biases
    )

    if model_name in {"mlp", "can_mlp"}:
        in_dim = sum(
            tensor.numel()
            for tensor in example.weights + example.biases
        )

        model = MLPModelForClassification(
            in_dim=in_dim,
            hidden_dim=32,
            n_hidden=4,
            n_classes=10,
            bn=True,
        )

    elif model_name == "dwsnet":
        model = DWSModelForClassification(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            input_features=1,
            hidden_dim=32,
            n_hidden=4,
            n_classes=10,
            reduction="max",
            n_fc_layers=1,
            num_heads=8,
            set_layer="sab",
            n_out_fc=1,
            dropout_rate=0.0,
            bn=True,
            diagonal=False,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model((batch.weights, batch.biases))

        total_loss += F.cross_entropy(
            logits,
            batch.label,
            reduction="sum",
        ).item()
        total_correct += (
            logits.argmax(dim=1)
            .eq(batch.label)
            .sum()
            .item()
        )
        total += batch.label.numel()

    return {
        "loss": total_loss / total,
        "acc": total_correct / total,
    }


def train_one_seed(
    dataset_name: str,
    model_name: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    statistics: dict,
    device: torch.device,
) -> dict:
    set_seed(seed)

    representation = (
        "canon"
        if model_name == "can_mlp"
        else "raw"
    )

    # The official DWSNet training command enables its generic INR
    # augmentation by default. The MLP and CanMLP experiments in our
    # paper explicitly disable generic augmentation.
    train_augmentation = model_name == "dwsnet"

    train_set = DWSProcessedDataset(
        dataset=dataset_name,
        split="train",
        representation=representation,
        statistics=statistics,
        augmentation=train_augmentation,
    )
    val_set = DWSProcessedDataset(
        dataset=dataset_name,
        split="val",
        representation=representation,
        statistics=statistics,
        augmentation=False,
    )
    test_set = DWSProcessedDataset(
        dataset=dataset_name,
        split="test",
        representation=representation,
        statistics=statistics,
        augmentation=False,
    )

    train_loader = make_loader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
        device=device,
    )
    val_loader = make_loader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed + 100000,
        device=device,
    )
    test_loader = make_loader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed + 200000,
        device=device,
    )

    example = train_set[0]
    model = build_model(
        model_name=model_name,
        example=example,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        amsgrad=True,
    )

    output_dir = (
        RESULTS_ROOT
        / dataset_name
        / model_name
        / f"seed_{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1

    epoch_iter = trange(
        1,
        epochs + 1,
        desc=f"{dataset_name}:{model_name}:seed{seed}",
    )

    for epoch in epoch_iter:
        model.train()

        train_loss_sum = 0.0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model((batch.weights, batch.biases))
            loss = F.cross_entropy(
                logits,
                batch.label,
            )

            loss.backward()
            optimizer.step()

            batch_size_actual = batch.label.numel()
            train_loss_sum += loss.item() * batch_size_actual
            train_total += batch_size_actual

        train_loss = train_loss_sum / train_total
        val = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )

        if val["acc"] >= best_val_acc:
            best_val_acc = val["acc"]
            best_val_loss = val["loss"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                    "val_loss": best_val_loss,
                },
                checkpoint_path,
            )

        epoch_iter.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_acc=f"{val['acc']:.4f}",
            best_val=f"{best_val_acc:.4f}",
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    test = evaluate(
        model=model,
        loader=test_loader,
        device=device,
    )

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_acc": test["acc"],
        "test_loss": test["loss"],
        "checkpoint": str(checkpoint_path),
    }

    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"Seed {seed}: best_epoch={best_epoch}, "
        f"val_acc={best_val_acc:.6f}, "
        f"test_acc={test['acc']:.6f}"
    )

    return result


def run(args) -> list[dict]:
    device = get_device(args.device)

    representation = (
        "canon"
        if args.model == "can_mlp"
        else "raw"
    )

    statistics = get_statistics(
        dataset=args.dataset,
        representation=representation,
        sample_size=args.statistics_sample_size,
        seed=args.statistics_seed,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Representation: {representation}")
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")

    results = []

    for seed in args.seeds:
        result = train_one_seed(
            dataset_name=args.dataset,
            model_name=args.model,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            statistics=statistics,
            device=device,
        )
        results.append(result)

    test_accs = [
        result["test_acc"]
        for result in results
    ]

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "seeds": args.seeds,
        "mean_test_acc": mean(test_accs),
        "std_test_acc": (
            stdev(test_accs)
            if len(test_accs) > 1
            else 0.0
        ),
        "results": results,
    }

    summary_dir = RESULTS_ROOT / args.dataset / args.model
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    display_dataset = (
        "MNIST-INR"
        if args.dataset == "mnist"
        else "Fashion-MNIST-INR"
    )
    display_model = {
        "mlp": "MLP",
        "can_mlp": "CanMLP",
        "dwsnet": "DWSNet",
    }[args.model]

    print()
    print(f"Dataset: {display_dataset}")
    print(f"Model: {display_model}")
    print(
        f"Mean test accuracy: "
        f"{summary['mean_test_acc']:.6f}"
    )
    print(
        f"Std test accuracy: "
        f"{summary['std_test_acc']:.6f}"
    )

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DWS INR-classification baselines."
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fmnist"],
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "can_mlp", "dwsnet"],
        required=True,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--statistics-sample-size",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--statistics-seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
