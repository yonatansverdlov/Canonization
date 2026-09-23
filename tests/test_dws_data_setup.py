"""Fast, dependency-light tests for DWS source recovery and discovery.

Load only the source-management functions via AST so these tests run even
on machines without CUDA, PyTorch Geometric or the 70k-example datasets.
"""
import ast
import os
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest


SOURCE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "setup_dws_data.py"
TEST_FUNCTIONS = {
    "_zip_member_is_junk",
    "extract_zip_with_progress",
    "mnist_model_paths",
    "_mnist_split_counts",
    "find_source_dataset_root",
    "source_diagnostic",
    "ensure_source_dataset",
    "discover_source_split",
    "main",
}


@pytest.fixture
def setup_functions():
    source = SOURCE_PATH.read_text()
    tree = ast.parse(source, filename=str(SOURCE_PATH))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in TEST_FUNCTIONS
    ]
    assert {f.name for f in functions} == TEST_FUNCTIONS
    namespace = {
        "Path": Path,
        "zipfile": zipfile,
        "sys": sys,
        "tqdm": lambda values, **kwargs: values,
    }
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace


def test_resume_extraction_skips_existing_and_metadata(setup_functions, tmp_path):
    archive = tmp_path / "data.zip"
    dest = tmp_path / "unpacked"
    existing = dest / "mnist-inrs" / "file1.pth"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"already here")
    os.utime(existing, (100, 100))

    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("mnist-inrs/file1.pth", b"already here")
        zf.writestr("mnist-inrs/file2.pth", b"new")
        zf.writestr("__MACOSX/._file2.pth", b"resource fork")

    setup_functions["extract_zip_with_progress"](archive, dest)
    assert existing.read_bytes() == b"already here"
    assert existing.stat().st_mtime == 100
    assert (dest / "mnist-inrs" / "file2.pth").read_bytes() == b"new"
    assert not (dest / "__MACOSX").exists()

    # Re-extraction should also be safe and should not overwrite either file.
    setup_functions["extract_zip_with_progress"](archive, dest)
    assert existing.stat().st_mtime == 100


def test_nested_mnist_ignores_unrelated_pth(setup_functions, tmp_path):
    root = tmp_path / "source" / "mnist"
    nested = root / "mnist-inrs"
    group = nested / "mnist_png_7_0"
    for split, name in [("train", "model0.pth"), ("train", "model1.pth"),
                        ("test", "model2.pth")]:
        target = group / split / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"checkpoint")
    (nested / "statistics.pth").write_bytes(b"stats")
    junk = root / "__MACOSX" / "mnist_png_7_0" / "train"
    junk.mkdir(parents=True)
    (junk / "._model0.pth").write_bytes(b"metadata")

    assert len(setup_functions["mnist_model_paths"](nested)) == 3

    # Use a three-file fixture, preserving the production 60k/10k validation.
    setup_functions["_mnist_split_counts"] = (
        lambda files, candidate: (60000, 10000) if len(files) == 3 else (0, 0)
    )
    assert setup_functions["find_source_dataset_root"](root, "mnist") == nested


def test_incomplete_source_and_zip_are_preserved(setup_functions, tmp_path):
    source_root = tmp_path / "source"
    download_root = tmp_path / "downloads"
    extracted = source_root / "mnist"
    extracted.mkdir(parents=True)
    marker = extracted / "existing-file.txt"
    marker.write_text("keep this file")
    download_root.mkdir()
    archive = download_root / "mnist-inrs.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("incomplete/not-an-original-model.pth", b"partial")

    setup_functions.update({
        "SOURCE_ROOT": source_root,
        "DOWNLOAD_ROOT": download_root,
        "DATASETS": {"mnist": {"archive": "mnist-inrs.zip", "url": "unused"}},
    })
    with pytest.raises(RuntimeError, match="Could not find complete mnist"):
        setup_functions["ensure_source_dataset"]("mnist")

    assert marker.read_text() == "keep this file"
    assert archive.is_file()


def test_mnist_failure_does_not_block_fmnist(setup_functions, tmp_path):
    attempted, processed = [], []
    def source(dataset):
        attempted.append(dataset)
        if dataset == "mnist":
            raise RuntimeError("incomplete MNIST archive")
        return tmp_path / "fmnist"

    setup_functions.update({
        "parse_args": lambda: Namespace(
            dataset="all", overwrite=False, split_seed=0, verify=2,
        ),
        "PROCESSED_ROOT": tmp_path / "processed",
        "processed_dataset_complete": lambda *_: False,
        "ensure_source_dataset": source,
        "build_geometric_dataset": lambda **kwargs: processed.append(
            kwargs["dataset"]
        ),
    })
    with pytest.raises(SystemExit) as caught:
        setup_functions["main"]()

    assert caught.value.code == 1
    assert attempted == ["mnist", "fmnist"]
    assert processed == ["fmnist"]


def test_completed_processed_datasets_skip_download(setup_functions, tmp_path):
    setup_functions.update({
        "parse_args": lambda: Namespace(
            dataset="all", overwrite=False, split_seed=0, verify=2,
        ),
        "PROCESSED_ROOT": tmp_path,
        "processed_dataset_complete": lambda *_: True,
        "ensure_source_dataset": lambda dataset: pytest.fail(
            f"{dataset} should not download"
        ),
    })
    setup_functions["main"]()
