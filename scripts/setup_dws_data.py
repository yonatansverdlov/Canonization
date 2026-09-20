#!/usr/bin/env python3

import argparse
import copy
import json
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dws.canonicalize import (
    canonicalize_state_dict,
    random_hidden_permutation,
    states_equal,
)


DATA_ROOT = REPO_ROOT / "data" / "dws"
DOWNLOAD_ROOT = DATA_ROOT / "downloads"
SOURCE_ROOT = DATA_ROOT / "source"
PROCESSED_ROOT = DATA_ROOT / "processed"

DATASETS = {
    "mnist": {
        "archive": "mnist-inrs.zip",
        "url": (
            "https://www.dropbox.com/sh/56pakaxe58z29mq/"
            "AABrctdu2U65jGYr2WQRzmMna/mnist-inrs.zip?dl=1"
        ),
    },
    "fmnist": {
        "archive": "fmnist_inrs.zip",
        "url": (
            "https://www.dropbox.com/sh/56pakaxe58z29mq/"
            "AAAssoHq719OmSHSKKTiKKHGa/fmnist_inrs.zip?dl=1"
        ),
    },
}


def download_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}

    if existing:
        headers["Range"] = f"bytes={existing}-"

    response = urlopen(
        Request(url, headers=headers),
        timeout=60,
    )

    status = getattr(response, "status", 200)
    resume = existing > 0 and status == 206

    if not resume:
        existing = 0

    content_length = response.headers.get("Content-Length")
    total = (
        int(content_length) + existing
        if content_length is not None
        else None
    )

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


def extract_zip_with_progress(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive, "r") as zf:
        members = zf.infolist()

        for member in tqdm(
            members,
            desc=f"Extracting {archive.name}",
            unit="file",
        ):
            zf.extract(member, destination)


def source_ready(source_root: Path) -> bool:
    if not source_root.is_dir():
        return False

    count = sum(1 for _ in source_root.rglob("*.pth"))
    return count == 70000


def ensure_source_dataset(dataset: str) -> Path:
    cfg = DATASETS[dataset]
    source_root = SOURCE_ROOT / dataset
    archive = DOWNLOAD_ROOT / cfg["archive"]

    print()
    print(f"[{dataset}] Checking source INRs...")

    if source_ready(source_root):
        print(f"[{dataset}] Source already available: {source_root}")
        return source_root

    if archive.exists() and not zipfile.is_zipfile(archive):
        print(f"[{dataset}] Corrupted archive found; deleting it.")
        archive.unlink()

    if not archive.exists():
        download_with_progress(cfg["url"], archive)
    else:
        print(f"[{dataset}] Archive already exists: {archive}")

    if not zipfile.is_zipfile(archive):
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded archive is not a valid ZIP: {archive}"
        )

    if source_root.exists():
        shutil.rmtree(source_root)

    extract_zip_with_progress(
        archive=archive,
        destination=source_root,
    )

    if not source_ready(source_root):
        raise RuntimeError(
            f"No .pth INR files found after extraction in {source_root}"
        )

    archive.unlink(missing_ok=True)

    print(f"[{dataset}] Source ready: {source_root}")
    return source_root


def infer_label(path: Path) -> int:
    """
    Match the label convention used by the original DWS MNIST split script:
        p.parent.parent.stem.split("_")[-2]
    """
    parts = path.parent.parent.stem.split("_")

    if len(parts) < 2:
        raise RuntimeError(
            f"Could not infer label from path: {path}"
        )

    try:
        return int(parts[-2])
    except ValueError as exc:
        raise RuntimeError(
            f"Could not infer integer label from path: {path}"
        ) from exc


def discover_source_split(source_root: Path):
    train = []
    test = []

    for path in sorted(source_root.rglob("*.pth")):
        item = {
            "path": path,
            "label": infer_label(path),
        }

        if "train" in path.as_posix().lower():
            train.append(item)
        else:
            test.append(item)

    if len(train) != 60000 or len(test) != 10000:
        raise RuntimeError(
            "Expected the original DWS MNIST/FMNIST INR split to contain "
            f"60000 train and 10000 test INRs, found "
            f"{len(train)} train and {len(test)} test."
        )

    return train, test


def build_split(source_root: Path, split_seed: int):
    train_all, test = discover_source_split(source_root)

    indices = list(range(len(train_all)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=5000,
        random_state=split_seed,
        shuffle=True,
    )

    train = [train_all[i] for i in train_idx]
    val = [train_all[i] for i in val_idx]

    if not (
        len(train) == 55000
        and len(val) == 5000
        and len(test) == 10000
    ):
        raise RuntimeError(
            "Unexpected split sizes: "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )

    return {
        "train": train,
        "val": val,
        "test": test,
    }


def load_state_dict(path: Path):
    try:
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            path,
            map_location="cpu",
        )

    if not hasattr(state, "items"):
        raise RuntimeError(
            f"Expected a state dict at {path}, got {type(state)}"
        )

    return state


def clone_state_to_cpu(state):
    out = {}

    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().clone()
        else:
            out[key] = copy.deepcopy(value)

    return out


def processed_dataset_complete(
    processed_root: Path,
    manifest_path: Path,
) -> bool:
    if not manifest_path.exists():
        return False

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception:
        return False

    expected = {
        "train": 55000,
        "val": 5000,
        "test": 10000,
    }

    for split_name, count in expected.items():
        paths = manifest.get("splits", {}).get(split_name, [])

        if len(paths) != count:
            return False

        if paths:
            first = processed_root / paths[0]
            last = processed_root / paths[-1]

            if not first.exists() or not last.exists():
                return False

    return True


def build_geometric_dataset(
    dataset: str,
    source_root: Path,
    split_seed: int,
    verify: int,
    overwrite: bool,
) -> None:
    processed_root = PROCESSED_ROOT / dataset
    manifest_path = processed_root / "splits.json"

    if (
        not overwrite
        and processed_dataset_complete(
            processed_root,
            manifest_path,
        )
    ):
        print(
            f"[{dataset}] Processed PyG dataset already complete: "
            f"{processed_root}"
        )
        return

    if overwrite and processed_root.exists():
        shutil.rmtree(processed_root)

    processed_root.mkdir(parents=True, exist_ok=True)

    split = build_split(
        source_root=source_root,
        split_seed=split_seed,
    )

    print()
    print(f"[{dataset}] Building PyG data")
    print("Fields: raw, canon")
    print(
        "Canonization: sequential lexicographic hidden-neuron sorting"
    )
    print(
        "Key: incoming weights + bias + sorted outgoing weights"
    )
    print(
        f"Split: 55000 train / 5000 val / 10000 test "
        f"(split seed {split_seed})"
    )

    generator = torch.Generator()
    generator.manual_seed(12345)

    verified = 0
    manifest = {
        "dataset": dataset,
        "split_seed": split_seed,
        "canonization": {
            "order": "input_to_output",
            "key": (
                "incoming weights + bias + "
                "sorted outgoing weights"
            ),
            "propagation": (
                "P_l sorts rows of W_l and b_l, then the same "
                "permutation sorts columns of W_{l+1}"
            ),
        },
        "splits": {
            "train": [],
            "val": [],
            "test": [],
        },
    }

    for split_name in ("train", "val", "test"):
        out_dir = processed_root / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        examples = split[split_name]

        for index, item in enumerate(
            tqdm(
                examples,
                desc=f"{dataset}:{split_name}",
                unit="INR",
            )
        ):
            destination = out_dir / f"{index:05d}.pt"

            if destination.exists() and not overwrite and verified >= verify:
                manifest["splits"][split_name].append(
                    str(destination.relative_to(processed_root))
                )
                continue

            state = load_state_dict(item["path"])
            canonical = canonicalize_state_dict(state)

            if verified < verify:
                permuted = random_hidden_permutation(
                    state,
                    generator=generator,
                )
                canonical_from_permuted = (
                    canonicalize_state_dict(permuted)
                )

                if not states_equal(
                    canonical,
                    canonical_from_permuted,
                ):
                    raise RuntimeError(
                        "Permutation-invariance verification failed for "
                        f"{item['path']}"
                    )

                verified += 1

            if destination.exists() and not overwrite:
                manifest["splits"][split_name].append(
                    str(destination.relative_to(processed_root))
                )
                continue

            data = Data(
                raw=clone_state_to_cpu(state),
                canon=clone_state_to_cpu(canonical),
                y=torch.tensor(
                    item["label"],
                    dtype=torch.long,
                ),
                split=split_name,
                source=str(
                    item["path"].relative_to(source_root)
                ),
            )

            tmp = destination.with_suffix(".pt.tmp")
            torch.save(data, tmp)
            tmp.replace(destination)

            manifest["splits"][split_name].append(
                str(destination.relative_to(processed_root))
            )

    with open(manifest_path, "w") as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print(f"[{dataset}] DONE")
    print(f"[{dataset}] Verified: {verified} permutation tests")
    print(f"[{dataset}] Manifest: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download original DWS MNIST/FMNIST INRs and convert them "
            "to PyG Data objects containing raw and canonized weights."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fmnist", "all"],
        default="all",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--verify",
        type=int,
        default=20,
        help=(
            "Number of examples on which to verify exact invariance "
            "to random hidden-neuron permutations."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    datasets = (
        ["mnist", "fmnist"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for dataset in datasets:
        source_root = ensure_source_dataset(dataset)

        build_geometric_dataset(
            dataset=dataset,
            source_root=source_root,
            split_seed=args.split_seed,
            verify=args.verify,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
