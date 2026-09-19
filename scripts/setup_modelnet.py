#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys
import zipfile
from urllib.request import Request, urlopen

from tqdm import tqdm
from torch_geometric.datasets import ModelNet


REPO_ROOT = Path(__file__).resolve().parents[1]

MODELNET_DIR = REPO_ROOT / "ModelNet"
PYG_DATA_ROOT = MODELNET_DIR / "data_creation" / "data" / "datasets"
TRAINING_DATA_ROOT = MODELNET_DIR / "training" / "data"

MODELNET40_H5_NAME = "modelnet40_ply_hdf5_2048"
MODELNET40_H5_URL = (
    "https://shapenet.cs.stanford.edu/media/"
    "modelnet40_ply_hdf5_2048.zip"
)
MODELNET40_H5_DIR = TRAINING_DATA_ROOT / MODELNET40_H5_NAME
MODELNET40_H5_ARCHIVE = TRAINING_DATA_ROOT / f"{MODELNET40_H5_NAME}.zip"

MODELNET10_H5_DIR = TRAINING_DATA_ROOT / "modelnet10_ply_hdf5_2048"


def h5_dataset_complete(dataset_dir: Path) -> bool:
    return (
        dataset_dir.is_dir()
        and any(dataset_dir.glob("ply_data_train*.h5"))
        and any(dataset_dir.glob("ply_data_test*.h5"))
    )


def download_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    request = Request(url, headers=headers)
    response = urlopen(request)

    status = getattr(response, "status", 200)
    resume = existing > 0 and status == 206

    if not resume:
        existing = 0

    content_length = response.headers.get("Content-Length")
    total = int(content_length) + existing if content_length else None

    mode = "ab" if resume else "wb"

    print(f"Downloading: {url}")
    with open(partial, mode) as f, tqdm(
        total=total,
        initial=existing,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=destination.name,
    ) as progress:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            progress.update(len(chunk))

    partial.replace(destination)


def ensure_pyg_modelnet(name: str) -> None:
    root = PYG_DATA_ROOT / f"ModelNet{name}"

    print()
    print(f"[ModelNet{name}] Checking PyG dataset...")

    # Instantiating both splits triggers PyG's own download/processing only
    # when required. Existing data is reused automatically.
    ModelNet(root=str(root), name=name, train=True)
    ModelNet(root=str(root), name=name, train=False)

    print(f"[ModelNet{name}] Ready: {root}")


def ensure_modelnet40_h5() -> None:
    print()
    print("[ModelNet40 HDF5] Checking dataset...")

    if h5_dataset_complete(MODELNET40_H5_DIR):
        print(f"[ModelNet40 HDF5] Already complete: {MODELNET40_H5_DIR}")
        return

    TRAINING_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if MODELNET40_H5_ARCHIVE.exists():
        if zipfile.is_zipfile(MODELNET40_H5_ARCHIVE):
            print(
                "[ModelNet40 HDF5] Archive already exists; "
                "skipping download."
            )
        else:
            print(
                "[ModelNet40 HDF5] Existing archive is corrupted; "
                "deleting it."
            )
            MODELNET40_H5_ARCHIVE.unlink()

    if not MODELNET40_H5_ARCHIVE.exists():
        download_with_progress(
            MODELNET40_H5_URL,
            MODELNET40_H5_ARCHIVE,
        )

    if not zipfile.is_zipfile(MODELNET40_H5_ARCHIVE):
        MODELNET40_H5_ARCHIVE.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded ModelNet40 archive is not a valid ZIP file."
        )

    print("[ModelNet40 HDF5] Extracting archive...")
    with zipfile.ZipFile(MODELNET40_H5_ARCHIVE, "r") as zf:
        zf.extractall(TRAINING_DATA_ROOT)

    if not h5_dataset_complete(MODELNET40_H5_DIR):
        raise RuntimeError(
            "ModelNet40 extraction finished, but the expected HDF5 "
            f"files were not found in {MODELNET40_H5_DIR}"
        )

    print(f"[ModelNet40 HDF5] Ready: {MODELNET40_H5_DIR}")


def ensure_modelnet10_h5() -> None:
    print()
    print("[ModelNet10 HDF5] Checking dataset...")

    if h5_dataset_complete(MODELNET10_H5_DIR):
        print(f"[ModelNet10 HDF5] Already complete: {MODELNET10_H5_DIR}")
        return

    converter = (
        MODELNET_DIR
        / "training"
        / "utils"
        / "create_h5_modelnet10_from_raw.py"
    )

    print("[ModelNet10 HDF5] Creating from raw ModelNet40...")
    subprocess.run(
        [sys.executable, str(converter)],
        cwd=REPO_ROOT,
        check=True,
    )

    if not h5_dataset_complete(MODELNET10_H5_DIR):
        raise RuntimeError(
            "ModelNet10 conversion finished, but the expected HDF5 "
            f"files were not found in {MODELNET10_H5_DIR}"
        )

    print(f"[ModelNet10 HDF5] Ready: {MODELNET10_H5_DIR}")


def main() -> None:
    print("=== ModelNet data setup ===")

    # Raw ModelNet10/40 used by the covering-number experiments.
    ensure_pyg_modelnet("10")
    ensure_pyg_modelnet("40")

    # HDF5 datasets used by the classification experiments.
    ensure_modelnet40_h5()
    ensure_modelnet10_h5()

    print()
    print("=== ModelNet setup complete ===")


if __name__ == "__main__":
    main()
