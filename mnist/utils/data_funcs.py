from torch.utils.data import TensorDataset
import os
import zipfile
import urllib.request as url_req
import numpy as np
import torch
from tqdm import tqdm

def obtain(dir_path):
    os.makedirs(dir_path, exist_ok=True)

    zip_path = os.path.join(dir_path, "mnist_rotated.zip")
    train_path = os.path.join(dir_path, "mnist_rotated_train.amat")
    valid_path = os.path.join(dir_path, "mnist_rotated_valid.amat")
    test_path = os.path.join(dir_path, "mnist_rotated_test.amat")

    if all(os.path.exists(p) for p in [train_path, valid_path, test_path]):
        print("Rotated MNIST already exists; skipping download.")
        return

    url = (
        "http://www.iro.umontreal.ca/~lisa/icml2007data/"
        "mnist_rotation_new.zip"
    )

    if not os.path.exists(zip_path):
        print("Downloading Rotated MNIST...")

        progress = tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="mnist_rotation_new.zip",
        )

        def reporthook(block_num, block_size, total_size):
            if total_size > 0 and progress.total is None:
                progress.total = total_size
            downloaded = block_num * block_size
            progress.update(max(0, downloaded - progress.n))

        try:
            url_req.urlretrieve(url, zip_path, reporthook=reporthook)
        finally:
            progress.close()
    else:
        print("Archive already exists; skipping download.")

    print("Extracting Rotated MNIST...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dir_path)

    source_train_valid = os.path.join(
        dir_path,
        "mnist_all_rotation_normalized_float_train_valid.amat",
    )
    source_test = os.path.join(
        dir_path,
        "mnist_all_rotation_normalized_float_test.amat",
    )

    if os.path.exists(source_test):
        os.replace(source_test, test_path)

    if os.path.exists(source_train_valid):
        with open(source_train_valid, "r") as f:
            lines = f.readlines()

        with open(train_path, "w") as f_train:
            f_train.writelines(lines[:10000])

        with open(valid_path, "w") as f_valid:
            f_valid.writelines(lines[10000:])

        os.remove(source_train_valid)

    if not all(os.path.exists(p) for p in [train_path, valid_path, test_path]):
        raise RuntimeError(
            "Rotated MNIST setup did not produce all expected split files."
        )

    if os.path.exists(zip_path):
        os.remove(zip_path)

    print("Rotated MNIST setup complete.")


def custom_load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    data = np.loadtxt(file_path, dtype=np.float32)
    images = torch.from_numpy(data[:, :-1])
    labels = torch.from_numpy(data[:, -1].astype(np.int64))
    return images, labels


def get_dataset(dir_path, split="train"):
    file_map = {
        "train": "mnist_rotated_train.amat",
        "valid": "mnist_rotated_valid.amat",
        "test": "mnist_rotated_test.amat",
    }

    if split not in file_map:
        raise ValueError(f"Unknown split: {split}")

    file_path = os.path.join(dir_path, file_map[split])
    images, labels = custom_load_data(file_path)
    return TensorDataset(images, labels)