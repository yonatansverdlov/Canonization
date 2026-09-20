from __future__ import annotations

import json
import random
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dws.canonicalize import get_layers


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = REPO_ROOT / "data" / "dws" / "processed"
SOURCE_ROOT = REPO_ROOT / "data" / "dws" / "source"


class WeightSpaceExample(NamedTuple):
    weights: tuple[torch.Tensor, ...]
    biases: tuple[torch.Tensor, ...]
    label: int


class WeightSpaceBatch(NamedTuple):
    weights: tuple[torch.Tensor, ...]
    biases: tuple[torch.Tensor, ...]
    label: torch.Tensor

    def to(self, device: torch.device):
        return WeightSpaceBatch(
            weights=tuple(w.to(device, non_blocking=True) for w in self.weights),
            biases=tuple(b.to(device, non_blocking=True) for b in self.biases),
            label=self.label.to(device, non_blocking=True),
        )


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def state_to_tensors(state: dict) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """
    Match the tensor convention used by the original DWS implementation.

    PyTorch stores Linear weights as [out_dim, in_dim].
    DWS uses [in_dim, out_dim, feature_dim], with feature_dim=1.
    """
    layers = get_layers(state)

    weights = tuple(
        state[weight_key]
        .detach()
        .to(dtype=torch.float32, device="cpu")
        .T
        .contiguous()
        .unsqueeze(-1)
        for weight_key, _ in layers
    )
    biases = tuple(
        state[bias_key]
        .detach()
        .to(dtype=torch.float32, device="cpu")
        .contiguous()
        .unsqueeze(-1)
        for _, bias_key in layers
    )

    return weights, biases


def normalize_tensors(
    weights: tuple[torch.Tensor, ...],
    biases: tuple[torch.Tensor, ...],
    statistics: dict,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    weight_means = statistics["weights"]["mean"]
    weight_stds = statistics["weights"]["std"]
    bias_means = statistics["biases"]["mean"]
    bias_stds = statistics["biases"]["std"]

    weights = tuple(
        (w - mean) / std.clamp_min(1e-8)
        for w, mean, std in zip(weights, weight_means, weight_stds)
    )
    biases = tuple(
        (b - mean) / std.clamp_min(1e-8)
        for b, mean, std in zip(biases, bias_means, bias_stds)
    )

    return weights, biases


class DWSProcessedDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        split: str,
        representation: str,
        statistics: dict | None,
    ):
        if dataset not in {"mnist", "fmnist"}:
            raise ValueError(f"Unknown dataset: {dataset}")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if representation not in {"raw", "canon"}:
            raise ValueError(f"Unknown representation: {representation}")

        self.dataset = dataset
        self.split = split
        self.representation = representation
        self.statistics = statistics

        self.root = PROCESSED_ROOT / dataset
        manifest_path = self.root / "splits.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Processed DWS dataset not found: {manifest_path}. "
                "Run scripts/setup_dws_data.py first."
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.paths = [
            self.root / rel_path
            for rel_path in manifest["splits"][split]
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int) -> WeightSpaceExample:
        data = _torch_load(self.paths[index])
        state = getattr(data, self.representation)

        weights, biases = state_to_tensors(state)

        if self.statistics is not None:
            weights, biases = normalize_tensors(
                weights,
                biases,
                self.statistics,
            )

        label = int(data.y.item())

        return WeightSpaceExample(
            weights=weights,
            biases=biases,
            label=label,
        )


def collate_weight_space(batch: list[WeightSpaceExample]) -> WeightSpaceBatch:
    n_weight_layers = len(batch[0].weights)
    n_bias_layers = len(batch[0].biases)

    weights = tuple(
        torch.stack([example.weights[layer] for example in batch], dim=0)
        for layer in range(n_weight_layers)
    )
    biases = tuple(
        torch.stack([example.biases[layer] for example in batch], dim=0)
        for layer in range(n_bias_layers)
    )
    labels = torch.tensor(
        [example.label for example in batch],
        dtype=torch.long,
    )

    return WeightSpaceBatch(
        weights=weights,
        biases=biases,
        label=labels,
    )


def _statistics_path(dataset: str, representation: str) -> Path:
    return PROCESSED_ROOT / dataset / f"statistics_{representation}.pt"


def _load_bundled_raw_statistics(dataset: str) -> dict | None:
    """
    Fashion-MNIST-INR is distributed with statistics.pth.
    If compatible bundled statistics exist, use them for the raw representation.
    """
    candidates = list((SOURCE_ROOT / dataset).rglob("statistics.pth"))

    for path in candidates:
        try:
            stats = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            stats = torch.load(path, map_location="cpu")

        if (
            isinstance(stats, dict)
            and "weights" in stats
            and "biases" in stats
        ):
            return stats

    return None


def compute_statistics(
    dataset: str,
    representation: str,
    sample_size: int = 10000,
    seed: int = 0,
) -> dict:
    """
    Reproduce the original DWS statistics protocol deterministically:
    compute per-parameter-coordinate mean/std from one 10k subset of train.
    """
    raw_dataset = DWSProcessedDataset(
        dataset=dataset,
        split="train",
        representation=representation,
        statistics=None,
    )

    n = min(sample_size, len(raw_dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(raw_dataset),
        generator=generator,
    )[:n].tolist()

    first = raw_dataset[indices[0]]
    weight_values = [[] for _ in first.weights]
    bias_values = [[] for _ in first.biases]

    for index in tqdm(
        indices,
        desc=f"Statistics {dataset}:{representation}",
        unit="INR",
    ):
        example = raw_dataset[index]

        for layer, weight in enumerate(example.weights):
            weight_values[layer].append(weight)

        for layer, bias in enumerate(example.biases):
            bias_values[layer].append(bias)

    weights = [torch.stack(values, dim=0) for values in weight_values]
    biases = [torch.stack(values, dim=0) for values in bias_values]

    statistics = {
        "weights": {
            "mean": [w.mean(dim=0) for w in weights],
            "std": [w.std(dim=0) for w in weights],
        },
        "biases": {
            "mean": [b.mean(dim=0) for b in biases],
            "std": [b.std(dim=0) for b in biases],
        },
        "sample_size": n,
        "seed": seed,
        "representation": representation,
    }

    return statistics


def get_statistics(
    dataset: str,
    representation: str,
    sample_size: int = 10000,
    seed: int = 0,
) -> dict:
    path = _statistics_path(dataset, representation)

    if path.exists():
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    if representation == "raw":
        bundled = _load_bundled_raw_statistics(dataset)
        if bundled is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(bundled, path)
            print(f"Using bundled raw statistics: {path}")
            return bundled

    statistics = compute_statistics(
        dataset=dataset,
        representation=representation,
        sample_size=sample_size,
        seed=seed,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(statistics, path)
    print(f"Saved statistics: {path}")

    return statistics


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
