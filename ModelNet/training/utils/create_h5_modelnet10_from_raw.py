import os
import random
import h5py
import numpy as np
import torch
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.io import read_off


SEED = 0
NUM_POINTS = 2048

RAW_ROOT = "/home/yonatans/canonization/ModelNet/data_creation/data/datasets/ModelNet40/raw"
OUT_ROOT = "/home/yonatans/canonization/ModelNet/training/data/modelnet10_ply_hdf5_2048"


MODELNET10_CLASSES = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]

CLASS_TO_LABEL = {name: i for i, name in enumerate(MODELNET10_CLASSES)}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pc_normalize(points: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    PointNet-style normalization:
    zero mean + unit sphere.
    """
    points = points.astype("float32")

    centroid = np.mean(points, axis=0, keepdims=True)
    points = points - centroid

    radius = np.max(np.linalg.norm(points, axis=1))
    points = points / max(radius, eps)

    return points.astype("float32")


def find_dataset_root(raw_root: str) -> str:
    """
    Handles both possible structures:

    raw/
        bathtub/
            train/
            test/

    or:

    raw/
        ModelNet40/
            bathtub/
                train/
                test/
    """
    if os.path.isdir(os.path.join(raw_root, "bathtub")):
        return raw_root

    nested = os.path.join(raw_root, "ModelNet40")
    if os.path.isdir(os.path.join(nested, "bathtub")):
        return nested

    raise FileNotFoundError(
        f"Could not find class folders under {raw_root}. "
        f"Expected folders like bathtub/train/*.off"
    )


def collect_off_files(dataset_root: str, split: str):
    examples = []

    for class_name in MODELNET10_CLASSES:
        class_dir = os.path.join(dataset_root, class_name, split)

        if not os.path.isdir(class_dir):
            print(f"Warning: missing folder {class_dir}")
            continue

        files = sorted(
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith(".off")
        )

        label = CLASS_TO_LABEL[class_name]

        for path in files:
            examples.append((path, label))

        print(f"{split} / {class_name}: {len(files)} files")

    return examples


def build_split(split: str):
    assert split in ["train", "test"]

    dataset_root = find_dataset_root(RAW_ROOT)
    examples = collect_off_files(dataset_root, split)

    print()
    print(f"Total {split} examples: {len(examples)}")

    sampler = T.SamplePoints(NUM_POINTS)

    all_points = []
    all_labels = []

    for off_path, label in tqdm(examples, desc=f"Creating ModelNet10 {split} H5"):
        data = read_off(off_path)

        data = sampler(data)

        points = data.pos.detach().cpu().numpy().astype("float32")

        if points.shape != (NUM_POINTS, 3):
            raise ValueError(
                f"Bad point shape for {off_path}: "
                f"expected {(NUM_POINTS, 3)}, got {points.shape}"
            )

        points = pc_normalize(points)

        all_points.append(points)
        all_labels.append([label])

    all_points = np.stack(all_points, axis=0).astype("float32")
    all_labels = np.array(all_labels, dtype="int64")

    os.makedirs(OUT_ROOT, exist_ok=True)

    out_path = os.path.join(OUT_ROOT, f"ply_data_{split}0.h5")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=all_points)
        f.create_dataset("label", data=all_labels)

    print()
    print(f"Saved: {out_path}")
    print("data:", all_points.shape, all_points.dtype)
    print("label:", all_labels.shape, all_labels.dtype)
    print("label min/max:", all_labels.min(), all_labels.max())


def save_shape_names():
    os.makedirs(OUT_ROOT, exist_ok=True)

    out_path = os.path.join(OUT_ROOT, "shape_names.txt")

    with open(out_path, "w") as f:
        for name in MODELNET10_CLASSES:
            f.write(name + "\n")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    seed_everything(SEED)

    build_split("train")
    build_split("test")
    save_shape_names()