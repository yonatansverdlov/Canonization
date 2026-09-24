"""Dataset-free tests of the two-command ModelNet experiment runner."""

import importlib.util
import json
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_modelnet_classification.py"
spec = importlib.util.spec_from_file_location("modelnet_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("dataset", ["10", "40"])
def test_one_command_runs_three_models(monkeypatch, tmp_path, dataset, capsys):
    monkeypatch.setattr(runner, "TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path / "results")
    calls = []

    def fake_subprocess_run(cmd, cwd, check):
        assert check is True
        assert cwd == runner.TRAINING_DIR
        assert cmd[cmd.index("--model") + 1] == "global_mlp"
        assert cmd[cmd.index("--run_5_seeds") + 1] == "true"
        ordering = cmd[cmd.index("--ordering") + 1]
        assert cmd[cmd.index("--dataset") + 1] == f"modelnet{dataset}"
        assert cmd[cmd.index("--seeds") + 1:cmd.index("--exp_name")] == ["0", "1", "2", "3", "4"]
        exp_name = cmd[cmd.index("--exp_name") + 1]
        assert exp_name == f"modelnet{dataset}_{ordering}"
        output = cwd / "checkpoints" / exp_name / "summary.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "dataset": f"modelnet{dataset}",
            "ordering": ordering,
            "seeds": [0, 1, 2, 3, 4],
            "test_acc_mean": 0.8,
            "test_acc_std": 0.02,
            "gen_gap_mean": 0.1,
            "gen_gap_std": 0.01,
        }))
        calls.append(ordering)

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)
    runner.main(["--dataset", dataset])
    assert calls == ["hilbert", "lex", "ply"]
    payload = json.loads((runner.RESULTS_ROOT / f"modelnet{dataset}_results.json").read_text())
    assert [row["model"] for row in payload["rows"]] == ["Hilbert", "Lex-Sort", "MLP"]
    csv_content = (runner.RESULTS_ROOT / f"modelnet{dataset}_results.csv").read_text()
    assert len(csv_content.strip().splitlines()) == 4
    output = capsys.readouterr().out
    table = output[output.rfind("┌"):]
    assert f"MODELNET{dataset} RESULTS (5 SEEDS)" in table
    assert "Test accuracy (%)" in table
    assert "Generalization gap (pp)" in table
    assert table.count("│") == 18  # title 2, header 4, three data rows 12
    for label in ("Hilbert", "Lex-Sort", "MLP"):
        assert f"│ {label:<8} │" in table
    assert table.count("80.00 ± 2.00") == 3
    assert table.count("10.00 ± 1.00") == 3
    assert "JSON:" not in table and "CSV:" not in table


def test_explicit_ordering_is_backwards_compatible(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path / "results")
    calls = []

    def fake_run(cmd, cwd, check):
        ordering = cmd[cmd.index("--ordering") + 1]
        calls.append(ordering)
        exp_name = cmd[cmd.index("--exp_name") + 1]
        path = cwd / "checkpoints" / exp_name / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "dataset": "modelnet40", "ordering": ordering, "seeds": [9],
            "test_acc_mean": 0.75, "test_acc_std": 0.0,
            "gen_gap_mean": 0.04, "gen_gap_std": 0.0,
        }))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.main(["--dataset", "40", "--ordering", "lex", "--seeds", "9"])
    assert calls == ["lex"]
