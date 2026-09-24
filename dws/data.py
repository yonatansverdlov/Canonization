from __future__ import annotations

import json
import random
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from dws.canonicalize import get_layers


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = REPO_ROOT / "data" / "dws" / "processed"


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


def augment_tensors(
    weights: tuple[torch.Tensor, ...],
    biases: tuple[torch.Tensor, ...],
    translation_scale: float = 0.25,
    rotation_degree: float = 45.0,
    noise_scale: float = 1e-1,
    drop_rate: float = 1e-2,
    resize_scale: float = 0.2,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """
    Original DWS INR augmentation used by the official MNIST trainer.
    Hidden-neuron permutation augmentation is intentionally NOT included.
    """
    new_weights = [w.squeeze(-1).clone() for w in weights]
    new_biases = [b.squeeze(-1).clone() for b in biases]

    translation = torch.empty(
        new_weights[0].shape[0]
    ).uniform_(
        -translation_scale,
        translation_scale,
    )

    order = random.sample(
        range(1, len(new_weights)),
        1,
    )[0]

    bias_res = translation
    layer_index = 0

    for layer_index in range(order):
        bias_res = bias_res @ new_weights[layer_index]

    new_biases[layer_index] = (
        new_biases[layer_index] + bias_res
    )

    if new_weights[0].shape[0] == 2:
        angle = torch.empty(1).uniform_(
            -rotation_degree,
            rotation_degree,
        )
        angle_rad = angle * (torch.pi / 180.0)
        c = torch.cos(angle_rad).squeeze(0)
        s = torch.sin(angle_rad).squeeze(0)
        rotation = torch.stack(
            [
                torch.stack([c, -s]),
                torch.stack([s, c]),
            ]
        )
        new_weights[0] = rotation @ new_weights[0]

    # Keep the original DWS implementation exactly: this is a
    # deterministic std-scaled offset, not sampled Gaussian noise.
    new_weights = [
        w + w.std() * noise_scale
        for w in new_weights
    ]
    new_biases = [
        b + b.std() * noise_scale if b.shape[0] > 1 else b
        for b in new_biases
    ]

    new_weights = [
        F.dropout(w, p=drop_rate, training=True)
        for w in new_weights
    ]
    new_biases = [
        F.dropout(b, p=drop_rate, training=True)
        for b in new_biases
    ]

    random_scale = (
        1.0
        + (torch.rand(1).item() - 0.5)
        * 2.0
        * resize_scale
    )
    new_weights[0] = new_weights[0] * random_scale

    return (
        tuple(w.unsqueeze(-1) for w in new_weights),
        tuple(b.unsqueeze(-1) for b in new_biases),
    )


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
        augmentation: bool = False,
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
        self.augmentation = augmentation

        if self.augmentation and split != "train":
            raise ValueError(
                "DWS augmentation may only be enabled for the train split"
            )

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

        if self.augmentation:
            weights, biases = augment_tensors(
                weights,
                biases,
            )

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

    if path.is_file():
        print(f"Loading statistics: {path}")
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    print(f"Statistics cache not found: {path}. Computing statistics...")

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
