from torch.utils.data import TensorDataset
import os
import zipfile
import urllib.request as url_req
import numpy as np
import torch

def obtain(dir_path):
    os.makedirs(dir_path, exist_ok=True)

    zip_path = os.path.join(dir_path, "mnist_rotated.zip")
    train_path = os.path.join(dir_path, "mnist_rotated_train.amat")
    valid_path = os.path.join(dir_path, "mnist_rotated_valid.amat")
    test_path = os.path.join(dir_path, "mnist_rotated_test.amat")

    if all(os.path.exists(p) for p in [train_path, valid_path, test_path]):
        print("Dataset already exists")
        return

    print("Downloading the dataset")
    url_req.urlretrieve(
        "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip",
        zip_path,
    )

    print("Extracting the dataset")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dir_path)

    os.rename(
        os.path.join(dir_path, "mnist_all_rotation_normalized_float_train_valid.amat"),
        train_path,
    )
    os.rename(
        os.path.join(dir_path, "mnist_all_rotation_normalized_float_test.amat"),
        test_path,
    )

    with open(train_path, "r") as f:
        lines = f.readlines()

    with open(train_path, "w") as f_train, open(valid_path, "w") as f_valid:
        f_train.writelines(lines[:10000])
        f_valid.writelines(lines[10000:])

    os.remove(zip_path)
    print("Done")


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