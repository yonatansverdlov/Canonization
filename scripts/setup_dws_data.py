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
    """Resume extraction without rewriting files that are already complete."""
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive, "r") as zf:
        members = zf.infolist()

        for member in tqdm(
            members,
            desc=f"Extracting {archive.name}",
            unit="file",
        ):
            target = destination / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif (
                target.is_file()
                and target.stat().st_size == member.file_size
            ):
                continue
            else:
                zf.extract(member, destination)


def archive_extraction_complete(archive: Path, destination: Path) -> bool:
    """Check ZIP contents against extracted file sizes before resuming."""
    if not destination.is_dir():
        return False

    with zipfile.ZipFile(archive, "r") as zf:
        return all(
            (destination / member.filename).is_file()
            and (destination / member.filename).stat().st_size
            == member.file_size
            for member in zf.infolist()
            if not member.is_dir()
        )


def _mnist_checkpoint_groups(source_root: Path):
    """
    Find DWS MNIST checkpoints using the authors' mnist_png_*/**/*.pth
    layout. An archive may also contain unrelated .pth files, so checking
    the total number of .pth files in the extraction root is incorrect.

    The key of each group is the directory containing mnist_png_* folders.
    """
    groups = {}
    total_pth = 0
    skipped_pth = 0

    for path in source_root.rglob("*.pth"):
        if not path.is_file():
            continue
        total_pth += 1

        relative_parts = path.relative_to(source_root).parts
        if path.name.startswith("._") or "__MACOSX" in relative_parts:
            skipped_pth += 1
            continue

        group_index = next(
            (
                i for i, part in enumerate(relative_parts[:-1])
                if part.startswith("mnist_png_")
            ),
            None,
        )
        if group_index is None:
            skipped_pth += 1
            continue

        try:
            label = infer_label(path)
        except RuntimeError:
            skipped_pth += 1
            continue

        if not 0 <= label <= 9:
            skipped_pth += 1
            continue

        collection_root = source_root.joinpath(
            *relative_parts[:group_index]
        )
        split_name = (
            "train" if "train" in path.as_posix().lower() else "test"
        )
        group = groups.setdefault(
            collection_root, {"train": [], "test": []}
        )
        group[split_name].append(path)

    return groups, total_pth, skipped_pth


def _select_mnist_checkpoints(source_root: Path, verbose=False):
    if not source_root.is_dir():
        return None

    groups, total_pth, skipped_pth = _mnist_checkpoint_groups(
        source_root
    )

    complete = [
        (root, group)
        for root, group in groups.items()
        if len(group["train"]) == 60000
        and len(group["test"]) == 10000
    ]
    if complete:
        # A genuine extraction is preferred over nested copies, if present.
        complete.sort(key=lambda pair: (len(pair[0].parts), str(pair[0])))
        root, group = complete[0]
        return (
            root,
            sorted(group["train"]),
            sorted(group["test"]),
        )

    if verbose:
        print(
            f"[mnist] Discovered {total_pth:,} .pth files; "
            f"{skipped_pth:,} are outside the original MNIST INR layout "
            "or have invalid labels."
        )
        for root, group in sorted(
            groups.items(),
            key=lambda pair: -(
                len(pair[1]["train"]) + len(pair[1]["test"])
            ),
        )[:5]:
            print(
                f"[mnist] Candidate {root}: "
                f"train={len(group['train']):,}, "
                f"test={len(group['test']):,}"
            )
            sample = (group["train"] or group["test"])
            if sample:
                print(f"[mnist] Example: {sample[0]}")
        if not groups and total_pth:
            examples = list(source_root.rglob("*.pth"))[:3]
            for example in examples:
                print(f"[mnist] Unrecognized .pth: {example}")

    return None


def find_source_dataset_root(source_root: Path, dataset: str):
    """
    Return the actual dataset directory when all source INRs are present.

    MNIST is identified using the original DWS mnist_png_*/**/*.pth
    pattern, rather than counting every .pth file in the archive.
    FMNIST ships with its own splits.json inside fmnist_inrs/.
    """
    if not source_root.is_dir():
        return None

    if dataset == "fmnist":
        candidates = [source_root] + [
            path.parent for path in source_root.rglob("splits.json")
        ]

        for candidate in candidates:
            train_dir = candidate / "train"
            test_dir = candidate / "test"
            split_path = candidate / "splits.json"

            if not (
                train_dir.is_dir()
                and test_dir.is_dir()
                and split_path.exists()
            ):
                continue

            model_count = (
                sum(1 for _ in train_dir.glob("model_*.pth"))
                + sum(1 for _ in test_dir.glob("model_*.pth"))
            )

            if model_count == 70000:
                return candidate

        return None

    selection = _select_mnist_checkpoints(source_root)
    return selection[0] if selection is not None else None


def source_ready(source_root: Path, dataset: str) -> bool:
    return find_source_dataset_root(source_root, dataset) is not None


def ensure_source_dataset(dataset: str) -> Path:
    cfg = DATASETS[dataset]
    source_root = SOURCE_ROOT / dataset
    archive = DOWNLOAD_ROOT / cfg["archive"]

    print()
    print(f"[{dataset}] Checking source INRs...")

    dataset_root = find_source_dataset_root(source_root, dataset)
    if dataset_root is not None:
        print(f"[{dataset}] Source already available: {dataset_root}")
        # A previous interrupted run may have left the ZIP behind.
        archive.unlink(missing_ok=True)
        return dataset_root

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

    if archive_extraction_complete(archive, source_root):
        # Do not delete or re-extract a complete 10-minute extraction if
        # the archive has an unexpected layout: report the structure.
        if dataset == "mnist":
            _select_mnist_checkpoints(source_root, verbose=True)
        raise RuntimeError(
            f"The {dataset} archive is already fully extracted, but "
            "its INR layout is unrecognized. Existing files were left "
            f"untouched at {source_root}. See the report above."
        )

    if source_root.exists():
        print(f"[{dataset}] Resuming extraction into {source_root}")

    extract_zip_with_progress(
        archive=archive,
        destination=source_root,
    )

    dataset_root = find_source_dataset_root(source_root, dataset)
    if dataset_root is None:
        if dataset == "mnist":
            _select_mnist_checkpoints(source_root, verbose=True)
        raise RuntimeError(
            f"Could not locate the {dataset} INR dataset after extraction "
            f"in {source_root}. Existing files were left untouched."
        )

    archive.unlink(missing_ok=True)

    print(f"[{dataset}] Source ready: {dataset_root}")
    return dataset_root


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


def build_fmnist_split(source_root: Path):
    """
    Use the authors' split bundled with FMNIST-INRs.

    splits.json contains absolute paths from the original machine, so each
    entry is resolved by its final two path components:
        train/model_k.pth
        test/model_k.pth

    Labels are stored inside the checkpoint itself.
    """
    split_path = source_root / "splits.json"

    with open(split_path) as f:
        source_split = json.load(f)

    split = {}

    for split_name in ("train", "val", "test"):
        examples = []

        for path_str in source_split[split_name]:
            original_path = Path(path_str)
            rel = Path(
                original_path.parent.name,
                original_path.name,
            )
            path = source_root / rel

            if not path.exists():
                raise RuntimeError(
                    f"FMNIST split entry does not exist locally: {path}"
                )

            examples.append(
                {
                    "path": path,
                    "label": None,
                }
            )

        split[split_name] = examples

    expected = {
        "train": 55000,
        "val": 5000,
        "test": 10000,
    }

    actual = {
        name: len(split[name])
        for name in ("train", "val", "test")
    }

    if actual != expected:
        raise RuntimeError(
            f"Unexpected FMNIST authors' split sizes: {actual}; "
            f"expected {expected}"
        )

    return split


def discover_source_split(source_root: Path):
    selection = _select_mnist_checkpoints(source_root, verbose=True)
    if selection is None:
        raise RuntimeError(
            "Cannot identify the original 60,000 train / 10,000 test "
            "MNIST-INR checkpoints. See the discovery report above."
        )

    _, train_paths, test_paths = selection
    train = [
        {"path": path, "label": infer_label(path)}
        for path in train_paths
    ]
    test = [
        {"path": path, "label": infer_label(path)}
        for path in test_paths
    ]
    return train, test


def build_split(dataset: str, source_root: Path, split_seed: int):
    if dataset == "fmnist":
        return build_fmnist_split(source_root)

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


def load_checkpoint(path: Path):
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

    state = dict(state)
    embedded_label = state.pop("label", None)

    if isinstance(embedded_label, torch.Tensor):
        embedded_label = int(embedded_label.item())
    elif embedded_label is not None:
        embedded_label = int(embedded_label)

    return state, embedded_label


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

        split_dir = processed_root / split_name
        if not split_dir.is_dir():
            return False

        file_count = sum(1 for _ in split_dir.glob("*.pt"))
        if file_count != count:
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
        dataset=dataset,
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
    if dataset == "fmnist":
        print(
            "Split: 55000 train / 5000 val / 10000 test "
            "(authors' bundled split)"
        )
    else:
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

            state, embedded_label = load_checkpoint(item["path"])

            label = item["label"]
            if label is None:
                if embedded_label is None:
                    raise RuntimeError(
                        f"No label found for {item['path']}"
                    )
                label = embedded_label

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
                    label,
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
