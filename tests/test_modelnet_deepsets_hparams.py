"""Check the reference-inspired preset without downloading ModelNet."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_deepsets_reference_recipe_and_existing_baselines():
    config = json.loads((ROOT / "ModelNet/training/configs/modelnet.json").read_text())
    hp = config["deepsets"]
    assert hp["optimizer"] == "adam"
    assert hp["adam_eps"] == 1e-3
    assert hp["lr"] == 1e-3
    assert hp["weight_decay"] == 1e-7
    assert hp["batch_size"] == 64
    assert hp["epochs"] == 200
    assert hp["scheduler"] == "multistep"
    assert hp["lr_milestones"] == [80, 160]
    assert hp["lr_gamma"] == 0.1
    assert hp["gradient_clip_val"] == 5
    assert hp["dropout"] == 0
    assert hp["point_dropout"] == 0
    assert hp["input_dropout"] == 0
    assert hp["label_smoothing"] == 0
    # The three canonicalization study baselines keep their existing settings.
    assert config["ply"]["lr"] == 0.0014988353560536768
    assert config["lex"]["lr"] == 0.0006407203395820919
    assert config["hilbert"]["lr"] == 0.0009614328324244756


def test_model_and_config_are_separate_from_ordering():
    trainer = (ROOT / "ModelNet/training/train.py").read_text()
    assert 'config_key = "deepsets" if args.model == "deepsets" else args.ordering' in trainer
    assert "gradient_clip_val=args.gradient_clip_val" in trainer
    assert "eps=self.args.adam_eps" in trainer
    model = (ROOT / "ModelNet/training/utils/models.py").read_text()
    assert "nn.Dropout(input_dropout)" in model
    runner = (ROOT / "scripts/run_modelnet_classification.py").read_text()
    assert "deepsets_reference_hps" in runner
