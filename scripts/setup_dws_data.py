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


def _zip_member_is_junk(name: str) -> bool:
    """Ignore macOS resource forks and unsafe/irrelevant ZIP members."""
    parts = Path(name).parts
    return (
        name.startswith("/")
        or ".." in parts
        or "__MACOSX" in parts
        or any(part.startswith("._") for part in parts)
    )


def extract_zip_with_progress(archive: Path, destination: Path) -> None:
    """Resume a partial extraction; never delete already extracted files."""
    destination.mkdir(parents=True, exist_ok=True)
    extracted = skipped = 0

    with zipfile.ZipFile(archive, "r") as zf:
        for member in tqdm(
            zf.infolist(), desc=f"Extracting {archive.name}", unit="file"
        ):
            if member.is_dir() or _zip_member_is_junk(member.filename):
                continue

            target = destination / member.filename
            if target.is_file() and target.stat().st_size == member.file_size:
                skipped += 1
                continue

            zf.extract(member, destination)
            extracted += 1

    print(f"Extraction finished: {extracted} files extracted, {skipped} reused.")


def mnist_model_paths(dataset_root: Path) -> list[Path]:
    """Only enumerate original DWS mnist_png_* checkpoints, not ZIP sidecars."""
    groups = sorted(
        path for path in dataset_root.glob("mnist_png_*") if path.is_dir()
    )
    return sorted(
        path
        for group in groups
        for path in group.rglob("*.pth")
        if not _zip_member_is_junk(path.relative_to(dataset_root).as_posix())
        and path.name != "statistics.pth"
    )


def _mnist_split_counts(paths: list[Path], root: Path) -> tuple[int, int]:
    train_count = sum(
        "train" in path.relative_to(root).as_posix().lower()
        for path in paths
    )
    return train_count, len(paths) - train_count


def find_source_dataset_root(source_root: Path, dataset: str):
    """Locate a complete dataset, including ZIPs with an extra wrapper dir."""
    if not source_root.is_dir():
        return None

    if dataset == "fmnist":
        candidates = [source_root] + sorted({
            p.parent
            for p in source_root.rglob("splits.json")
            if not _zip_member_is_junk(p.relative_to(source_root).as_posix())
        })
        for candidate in candidates:
            train_dir = candidate / "train"
            test_dir = candidate / "test"
            if not (
                (candidate / "splits.json").is_file()
                and train_dir.is_dir()
                and test_dir.is_dir()
            ):
                continue
            if (
                sum(1 for _ in train_dir.glob("model_*.pth")) == 60000
                and sum(1 for _ in test_dir.glob("model_*.pth")) == 10000
            ):
                return candidate
        return None

    candidates = sorted({
        group.parent
        for group in source_root.rglob("mnist_png_*")
        if group.is_dir()
        and not _zip_member_is_junk(
            group.relative_to(source_root).as_posix()
        )
    })
    for candidate in candidates:
        files = mnist_model_paths(candidate)
        if _mnist_split_counts(files, candidate) == (60000, 10000):
            return candidate
    return None


def source_diagnostic(source_root: Path, dataset: str) -> str:
    """Describe an incomplete extraction instead of issuing a bare error."""
    if dataset == "mnist":
        all_pth = [
            p for p in source_root.rglob("*.pth")
            if not _zip_member_is_junk(
                p.relative_to(source_root).as_posix()
            )
        ]
        groups = [
            p for p in source_root.rglob("mnist_png_*") if p.is_dir()
        ]
        group_counts = [
            (str(g.parent.relative_to(source_root)), len(mnist_model_paths(g.parent)))
            for g in groups[:1]
        ]
        return (
            f"{len(all_pth)} non-metadata .pth files found; "
            f"example: {[str(p.relative_to(source_root)) for p in all_pth[:3]]}; "
            f"mnist_png_* groups: {len(groups)}; "
            f"group counts: {group_counts}. "
            "Expected 60000 train and 10000 test checkpoints."
        )

    splits = [
        p.relative_to(source_root).as_posix()
        for p in source_root.rglob("splits.json")
        if not _zip_member_is_junk(p.relative_to(source_root).as_posix())
    ]
    model_count = sum(1 for _ in source_root.rglob("model_*.pth"))
    return (
        f"Found {model_count} model_*.pth files and splits at {splits[:3]}. "
        "Expected 60000 train, 10000 test and splits.json."
    )


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
        archive.unlink(missing_ok=True)
        print(f"[{dataset}] Source already available: {dataset_root}")
        return dataset_root

    if archive.exists() and not zipfile.is_zipfile(archive):
        print(f"[{dataset}] Incomplete/corrupt archive; keeping extracted files.")
        archive.unlink()

    if not archive.exists():
        download_with_progress(cfg["url"], archive)
    else:
        print(f"[{dataset}] Reusing archive: {archive}")

    if not zipfile.is_zipfile(archive):
        raise RuntimeError(
            f"Downloaded archive is not a valid ZIP: {archive}. "
            "Any previously extracted files were preserved."
        )

    # Resume an incomplete extraction rather than deleting its entire tree.
    extract_zip_with_progress(archive, source_root)

    dataset_root = find_source_dataset_root(source_root, dataset)
    if dataset_root is None:
        raise RuntimeError(
            f"Could not find complete {dataset} INRs after extraction: "
            f"{source_diagnostic(source_root, dataset)} "
            f"Files and archive were preserved in {source_root} and {archive}."
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
    train = []
    test = []

    for path in mnist_model_paths(source_root):
        item = {"path": path, "label": infer_label(path)}
        if "train" in path.relative_to(source_root).as_posix().lower():
            train.append(item)
        else:
            test.append(item)

    if len(train) != 60000 or len(test) != 10000:
        raise RuntimeError(
            "Expected 60000 train and 10000 test MNIST INRs, found "
            f"{len(train)} train and {len(test)} test."
        )

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
        ["mnist", "fmnist"] if args.dataset == "all"
        else [args.dataset]
    )
    failures = []

    for dataset in datasets:
        try:
            processed_root = PROCESSED_ROOT / dataset
            if not args.overwrite and processed_dataset_complete(
                processed_root, processed_root / "splits.json"
            ):
                print(f"[{dataset}] Processed dataset already complete; skipping.")
                continue

            source_root = ensure_source_dataset(dataset)
            build_geometric_dataset(
                dataset=dataset,
                source_root=source_root,
                split_seed=args.split_seed,
                verify=args.verify,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            # A problem with MNIST must not prevent FMNIST from downloading.
            failures.append((dataset, exc))
            print(
                f"\n[{dataset}] FAILED ({type(exc).__name__}): {exc}",
                file=sys.stderr,
            )
            print(
                f"[{dataset}] Existing downloads and extracted files retained.",
                file=sys.stderr,
            )

    print("\n========== DWS DATA SETUP SUMMARY ==========")
    for dataset in datasets:
        failure = next((e for name, e in failures if name == dataset), None)
        print(f"{dataset}: {'FAILED: ' + str(failure) if failure else 'OK'}")
    print("============================================")

    if failures:
        raise SystemExit(1)




if __name__ == "__main__":
    main()
