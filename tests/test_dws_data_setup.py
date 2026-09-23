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
    root = tmp_path / "mnist"
    nested = root / "mnist-inrs"
    group = nested / "mnist_png_7_0"
    for split, name in [
        ("train", "0.pth"), ("train", "1.pth"), ("test", "2.pth")
    ]:
        file = group / split / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"checkpoint")

    unrelated = nested / "statistics.pth"
    unrelated.write_bytes(b"stats")
    junk = root / "__MACOSX" / "mnist_png_7_0" / "train"
    junk.mkdir(parents=True)
    (junk / "._0.pth").write_bytes(b"metadata")

    grouped, count, ignored = setup_functions["_mnist_checkpoint_groups"](root)
    assert count == 5  # 3 model checkpoints + statistics + macOS metadata
    assert ignored == 2
    assert set(grouped) == {nested}
    assert len(grouped[nested]["train"]) == 2
    assert len(grouped[nested]["test"]) == 1

    # Model a full dataset without writing 70,000 files to disk.
    grouped[nested]["train"] *= 30000
    grouped[nested]["test"] *= 10000
    setup_functions["_mnist_checkpoint_groups"] = lambda _: (grouped, 70001, 1)
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
