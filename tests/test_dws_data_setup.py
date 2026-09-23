"""Fast, dataset-free regression tests for DWS source setup.

Load only source-management functions from AST so these tests do not need
PyTorch, PyTorch Geometric, or the 70k-checkpoint dataset.
"""
import ast
import os
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

SOURCE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "setup_dws_data.py"
FUNCTIONS = {
    "_zip_member_is_junk",
    "extract_zip_with_progress",
    "archive_extraction_complete",
    "infer_label",
    "_mnist_checkpoint_groups",
    "_select_mnist_checkpoints",
    "find_source_dataset_root",
    "ensure_source_dataset",
    "main",
}


@pytest.fixture
def setup_functions():
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    ]
    assert {node.name for node in selected} == FUNCTIONS
    namespace = {
        "Path": Path,
        "os": os,
        "zipfile": zipfile,
        "tqdm": lambda items, **kwargs: items,
        "sys": sys,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE_PATH), "exec"),
        namespace,
    )
    return namespace


def test_resume_extraction_preserves_completed_files(setup_functions, tmp_path):
    archive = tmp_path / "mnist.zip"
    destination = tmp_path / "source"
    complete = destination / "mnist-inrs" / "one.pth"
    complete.parent.mkdir(parents=True)
    complete.write_bytes(b"already complete")
    os.utime(complete, (100, 100))
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("mnist-inrs/one.pth", b"already complete")
        zf.writestr("mnist-inrs/two.pth", b"new")
        zf.writestr("__MACOSX/._one.pth", b"junk")

    extract = setup_functions["extract_zip_with_progress"]
    extract(archive, destination)
    extract(archive, destination)
    assert complete.stat().st_mtime == 100
    assert (destination / "mnist-inrs" / "two.pth").read_bytes() == b"new"
    assert not (destination / "__MACOSX").exists()
    assert setup_functions["archive_extraction_complete"](archive, destination)


def test_nested_mnist_recognizes_groups_not_metadata(setup_functions, tmp_path):
    # Match the exact structure of the downloaded mixed MNIST/CIFAR archive.
    root = tmp_path / "mnist"
    nested = root / "mnist-inrs"
    for split, suffix in [
        ("training", "0001"), ("training", "0002"), ("testing", "0003")
    ]:
        checkpoint = (
            nested / f"mnist_png_{split}_7_{suffix}"
            / "checkpoints" / "model_final.pth"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")

    (nested / "statistics.pth").write_bytes(b"stats")
    cifar = (
        nested / "cifar10_png_train_airplane_0001"
        / "checkpoints" / "model_final.pth"
    )
    cifar.parent.mkdir(parents=True)
    cifar.write_bytes(b"cifar - not mnist")
    junk = (
        root / "__MACOSX" / "mnist_png_training_7_9999"
        / "checkpoints" / "model_final.pth"
    )
    junk.parent.mkdir(parents=True)
    junk.write_bytes(b"metadata")

    grouped, candidate_dirs, rejected = (
        setup_functions["_mnist_checkpoint_groups"](root)
    )
    assert candidate_dirs == 3  # Direct children only; macOS metadata is ignored.
    assert rejected == 0
    assert set(grouped) == {nested}
    assert len(grouped[nested]["train"]) == 2
    assert len(grouped[nested]["test"]) == 1
    assert all(
        path.name == "model_final.pth"
        for paths in grouped[nested].values() for path in paths
    )

    # Simulate the complete authors' split without creating 70k checkpoints.
    grouped[nested]["train"] *= 30000
    grouped[nested]["test"] *= 10000
    setup_functions["_mnist_checkpoint_groups"] = lambda _: (grouped, 70000, 0)
    assert setup_functions["find_source_dataset_root"](root, "mnist") == nested


def test_incomplete_source_and_zip_are_retained(setup_functions, tmp_path):
    source_root = tmp_path / "source"
    extracted = source_root / "mnist"
    extracted.mkdir(parents=True)
    marker = extracted / "keep.txt"
    marker.write_text("keep")
    download_root = tmp_path / "downloads"
    download_root.mkdir()
    archive = download_root / "mnist-inrs.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("not-a-dataset/model.pth", b"partial")

    setup_functions.update({
        "SOURCE_ROOT": source_root,
        "DOWNLOAD_ROOT": download_root,
        "DATASETS": {"mnist": {"archive": archive.name, "url": "unused"}},
        "print_source_diagnostics": lambda *_: None,
    })
    with pytest.raises(RuntimeError, match="Cannot recognize"):
        setup_functions["ensure_source_dataset"]("mnist")
    assert marker.read_text() == "keep"
    assert archive.is_file()


def test_mnist_failure_does_not_prevent_fmnist(setup_functions, tmp_path):
    attempted, processed = [], []

    def prepare(dataset):
        attempted.append(dataset)
        if dataset == "mnist":
            raise RuntimeError("bad source")
        return tmp_path / "fmnist"

    setup_functions.update({
        "parse_args": lambda: Namespace(
            dataset="all", overwrite=False, split_seed=0,
            verify=2, inspect_source=False,
        ),
        "PROCESSED_ROOT": tmp_path / "processed",
        "processed_dataset_complete": lambda *_: False,
        "ensure_source_dataset": prepare,
        "build_geometric_dataset": lambda **kwargs: processed.append(
            kwargs["dataset"]
        ),
    })
    with pytest.raises(SystemExit) as exit_status:
        setup_functions["main"]()
    assert exit_status.value.code == 1
    assert attempted == ["mnist", "fmnist"]
    assert processed == ["fmnist"]


def test_complete_processed_data_never_downloads(setup_functions, tmp_path):
    setup_functions.update({
        "parse_args": lambda: Namespace(
            dataset="all", overwrite=False, split_seed=0,
            verify=2, inspect_source=False,
        ),
        "PROCESSED_ROOT": tmp_path,
        "processed_dataset_complete": lambda *_: True,
        "ensure_source_dataset": lambda _: pytest.fail("unnecessary download"),
    })
    setup_functions["main"]()


def test_inspection_is_read_only(setup_functions, tmp_path):
    inspected = []
    setup_functions.update({
        "parse_args": lambda: Namespace(
            dataset="all", overwrite=False, split_seed=0,
            verify=2, inspect_source=True,
        ),
        "SOURCE_ROOT": tmp_path / "source",
        "DOWNLOAD_ROOT": tmp_path / "downloads",
        "DATASETS": {
            "mnist": {"archive": "mnist.zip"},
            "fmnist": {"archive": "fmnist.zip"},
        },
        "print_source_diagnostics": lambda *args: inspected.append(args[1]),
        "ensure_source_dataset": lambda _: pytest.fail("inspection downloaded"),
    })
    setup_functions["main"]()
    assert inspected == ["mnist", "fmnist"]


def test_complete_mnist_selection_is_cached(setup_functions, tmp_path):
    """After initial discovery, build_split must not scan 70k files again."""
    root = tmp_path / "mnist"
    nested = root / "mnist-inrs"
    nested.mkdir(parents=True)
    train = nested / "mnist_png_training_0_1" / "checkpoints" / "model_final.pth"
    test = nested / "mnist_png_testing_0_1" / "checkpoints" / "model_final.pth"
    for checkpoint in (train, test):
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"example")

    calls = []

    def discover(path):
        calls.append(path)
        return {nested: {"train": [train] * 60000, "test": [test] * 10000}}, 70000, 0

    setup_functions["_mnist_checkpoint_groups"] = discover
    select = setup_functions["_select_mnist_checkpoints"]
    first = select(root)
    second = select(nested)  # build_split passes this nested path
    assert first is second
    assert calls == [root]
