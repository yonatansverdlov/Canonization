"""ModelNet DeepSets architecture and CLI tests without dataset downloads or PyG."""

import ast
import importlib.util
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_modelnet_classification.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("modelnet_deepsets_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_classifiers():
    # The same models.py also has unrelated torch-geometric rotation models.
    # Compile the actual classifier definitions in isolation for CPU-only tests.
    torch = pytest.importorskip("torch")
    source = ROOT / "ModelNet" / "training" / "utils" / "models.py"
    tree = ast.parse(source.read_text())
    names = {
        "FourierFeatureMap", "MLPResidualBlock", "DynamicOrdering",
        "GlobalMLPClassifier", "DeepSetsMLPClassifier",
    }
    selected = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    assert {node.name for node in selected} == names
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    scope = {"torch": torch, "nn": torch.nn, "math": math}
    exec(compile(module, str(source), "exec"), scope)
    return torch, scope["GlobalMLPClassifier"], scope["DeepSetsMLPClassifier"]


def test_pointwise_mlp_is_exact_sum_of_baseline_mlp_logits():
    torch, Baseline, DeepSets = load_classifiers()
    torch.manual_seed(7)
    baseline = Baseline(num_classes=10, num_points=8, num_bands=3)
    model = DeepSets(num_classes=10, num_points=8, num_bands=3)
    assert baseline.input_dim == 8 * 18
    assert model.input_dim == 18
    assert [b.linear1.out_features for b in baseline.blocks] == [256, 128, 64]
    assert [b.linear1.out_features for b in model.blocks] == [256, 128, 64]
    model.eval()
    x = torch.randn(2, 8, 3)
    actual = model(x)
    per_point = model.fourier_map(model.dynamic_order(x))
    per_point = model.point_dropout(per_point)
    per_point = model.input_drop(per_point)
    expected = model.head(model.final_norm(model.blocks(per_point))).sum(dim=1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        actual, model(x[:, torch.randperm(8)]), atol=1e-5, rtol=1e-5
    )
    assert actual.shape == (2, 10)


def test_deepsets_backward():
    torch, _, DeepSets = load_classifiers()
    model = DeepSets(num_classes=40, num_points=8, num_bands=3)
    model.train()
    model(torch.randn(2, 8, 3)).square().mean().backward()
    assert model.blocks[0].linear1.weight.grad is not None


@pytest.mark.parametrize("dataset", ["10", "40"])
def test_runner_selects_deepsets_with_ply_hyperparameters(monkeypatch, tmp_path, dataset):
    runner = load_runner()
    monkeypatch.setattr(runner, "TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path / "results")
    commands = []

    def fake_run(cmd, cwd, check):
        assert check is True
        assert cmd[cmd.index("--model") + 1] == "deepsets"
        assert cmd[cmd.index("--ordering") + 1] == "ply"
        assert cmd[cmd.index("--dataset") + 1] == f"modelnet{dataset}"
        assert cmd[cmd.index("--run_5_seeds") + 1] == "true"
        name = cmd[cmd.index("--exp_name") + 1]
        assert name == f"modelnet{dataset}_deepsets_reference_hps"
        commands.append(cmd)
        output = cwd / "checkpoints" / name / "summary.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "dataset": f"modelnet{dataset}", "model": "deepsets",
            "ordering": "ply", "seeds": [0, 1, 2, 3, 4],
            "test_acc_mean": 0.7, "test_acc_std": 0.02,
            "gen_gap_mean": 0.1, "gen_gap_std": 0.01,
        }))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.main(["--dataset", dataset, "--ordering", "deepsets"])
    assert len(commands) == 1
    result = json.loads(
        (runner.RESULTS_ROOT / f"modelnet{dataset}_results.json").read_text()
    )
    assert [(row["model"], row["ordering"]) for row in result["rows"]] == [
        ("DeepSets", "ply")
    ]


def test_runner_runs_all_four_by_default(monkeypatch, tmp_path):
    runner = load_runner()
    monkeypatch.setattr(runner, "TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path / "results")
    calls = []

    def fake_run(cmd, cwd, check):
        assert check is True
        model = cmd[cmd.index("--model") + 1]
        ordering = cmd[cmd.index("--ordering") + 1]
        name = cmd[cmd.index("--exp_name") + 1]
        calls.append((model, ordering))
        path = cwd / "checkpoints" / name / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "dataset": "modelnet40", "model": model,
            "ordering": ordering, "seeds": [0],
            "test_acc_mean": 0.7, "test_acc_std": 0.0,
            "gen_gap_mean": 0.1, "gen_gap_std": 0.0,
        }))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.main(["--dataset", "40", "--seeds", "0"])
    assert calls == [
        ("global_mlp", "hilbert"), ("global_mlp", "lex"),
        ("global_mlp", "ply"), ("deepsets", "ply"),
    ]
    rows = json.loads((runner.RESULTS_ROOT / "modelnet40_results.json").read_text())["rows"]
    assert [row["model"] for row in rows] == ["Hilbert", "Lex-Sort", "MLP", "DeepSets"]
