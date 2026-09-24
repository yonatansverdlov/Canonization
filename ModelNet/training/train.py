#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import json
import shutil
import argparse
import copy
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    import lightning as L
    from lightning.pytorch.callbacks import Callback
except ImportError:
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import Callback

from torchmetrics.classification import MulticlassAccuracy

from utils.data import OrderedModelNet40
from utils.models import GlobalMLPClassifier, PointTransformerClassifier
from utils.util import IOStream


def set_seed(seed: int) -> None:
    L.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_dataset_config(dataset: str) -> Dict[str, Any]:
    if dataset == "modelnet40":
        return {
            "dataset_name": "modelnet40_ply_hdf5_2048",
            "num_classes": 40,
        }

    if dataset == "modelnet10":
        return {
            "dataset_name": "modelnet10_ply_hdf5_2048",
            "num_classes": 10,
        }

    raise ValueError(f"Unknown dataset: {dataset}")


def cli_overrides(argv):
    """
    Return argument names explicitly passed by the user.

    Examples:
        --lr 0.1        -> "lr"
        --lr=0.1        -> "lr"
        --run_5_seeds   -> "run_5_seeds"

    Priority will be:
        command line > ordering config > parser defaults
    """
    overrides = set()

    for token in argv:
        if not token.startswith("--"):
            continue

        name = token[2:].split("=")[0]
        name = name.replace("-", "_")
        overrides.add(name)

    return overrides


def apply_ordering_config(args, explicit_args):
    """
    Load args.config and apply the hyperparameters matching args.ordering.

    Example:
        --ordering hilbert
    loads:
        config["hilbert"]

    Priority:
        command line > ordering config > parser defaults

    Note:
        We intentionally do not use --preset.
        The ordering itself selects the config.
    """
    if args.ordering not in ["ply", "lex", "hilbert"]:
        return args

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Could not find config file: {args.config}")

    with open(args.config, "r") as f:
        config = json.load(f)

    if args.ordering not in config:
        raise ValueError(
            f"Ordering '{args.ordering}' not found in {args.config}. "
            f"Available configs: {list(config.keys())}"
        )

    ordering_values = config[args.ordering]

    for key, value in ordering_values.items():
        if key in explicit_args:
            continue

        if key == "ordering":
            raise ValueError(
                "Do not put 'ordering' inside the config. "
                "The config block is already selected by --ordering."
            )

        if not hasattr(args, key):
            raise ValueError(
                f"Config key '{key}' is not a valid parser argument."
            )

        setattr(args, key, value)

    return args


def _init_(args):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(os.path.join("checkpoints", args.exp_name), exist_ok=True)
    os.makedirs(os.path.join("checkpoints", args.exp_name, "models"), exist_ok=True)

    for filename in ["train.py", "models.py", "data.py", "util.py"]:
        if os.path.exists(filename):
            shutil.copy(
                filename,
                os.path.join("checkpoints", args.exp_name, f"{filename}.backup"),
            )


class ModelNetDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cfg = get_dataset_config(args.dataset)
        self.dataset_name = cfg["dataset_name"]

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)

        return DataLoader(
            OrderedModelNet40(
                partition="train",
                num_points=self.args.num_points,
                ordering=self.args.ordering,
                dataset_name=self.dataset_name,
                dataset_stride=self.args.dataset_stride,
                use_fps=self.args.use_fps,
                apply_jitter=self.args.apply_jitter,
                apply_anisotropic_scale=self.args.apply_scale,
                apply_random_permutation=self.args.apply_random_permutation,
                apply_rotation=self.args.apply_rotation,
            ),
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            persistent_workers=self.args.num_workers > 0,
            worker_init_fn=worker_init_fn if self.args.num_workers > 0 else None,
            generator=generator,
        )

    def test_dataloader(self):
        generator = torch.Generator().manual_seed(self.args.seed + 100000)
        return DataLoader(
            OrderedModelNet40(
                partition="test",
                num_points=self.args.num_points,
                ordering=self.args.ordering,
                dataset_name=self.dataset_name,
                dataset_stride=1,
                use_fps=self.args.use_fps,
                apply_jitter=False,
                apply_anisotropic_scale=False,
                apply_random_permutation=False,
                apply_rotation=False,
            ),
            batch_size=self.args.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            persistent_workers=self.args.num_workers > 0,
            worker_init_fn=worker_init_fn if self.args.num_workers > 0 else None,
            generator=generator,
        )


class LitModelNetClassifier(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args

        cfg = get_dataset_config(args.dataset)
        self.num_classes = cfg["num_classes"]

        if args.model == "global_mlp":
            self.model = GlobalMLPClassifier(
                num_classes=self.num_classes,
                num_points=args.num_points,
                num_bands=args.num_bands,
                fourier_scale=args.fourier_scale,
                dropout=args.dropout,
                ordering_type=args.ordering,
            )
        elif args.model == "point_transformer":
            self.model = PointTransformerClassifier(
                num_classes=self.num_classes,
                dim=args.trans_dim,
                depth=args.trans_depth,
                heads=args.trans_heads,
                drop_rate=args.dropout,
                drop_path_rate=args.drop_path_rate,
            )
        else:
            raise ValueError(f"Model {args.model} not implemented")

        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        self.train_acc = MulticlassAccuracy(
            num_classes=self.num_classes,
            average="micro",
        )
        self.train_avg_acc = MulticlassAccuracy(
            num_classes=self.num_classes,
            average="macro",
        )
        self.val_acc = MulticlassAccuracy(
            num_classes=self.num_classes,
            average="micro",
        )
        self.val_avg_acc = MulticlassAccuracy(
            num_classes=self.num_classes,
            average="macro",
        )
        self.test_acc = MulticlassAccuracy(
            num_classes=self.num_classes,
            average="micro",
        )
        self.test_avg_acc = MulticlassAccuracy(
            num_classes=self.num_classes,
            average="macro",
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt_choice = "sgd" if self.args.use_sgd else self.args.optimizer.lower()

        if opt_choice == "sgd":
            start_lr = self.args.lr * 100
            optimizer = optim.SGD(
                self.parameters(),
                lr=start_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        elif opt_choice == "adamw":
            start_lr = self.args.lr
            optimizer = optim.AdamW(
                self.parameters(),
                lr=start_lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            start_lr = self.args.lr
            optimizer = optim.Adam(
                self.parameters(),
                lr=start_lr,
                weight_decay=self.args.weight_decay,
            )

        min_lr = start_lr * 0.001
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.args.epochs,
            eta_min=min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _check_labels(self, label, logits):
        if label.min() < 0 or label.max() >= self.num_classes:
            raise ValueError(
                f"Bad labels for CrossEntropyLoss: "
                f"label min={label.min().item()}, "
                f"label max={label.max().item()}, "
                f"num_classes={self.num_classes}, "
                f"logits shape={tuple(logits.shape)}, "
                f"dataset={self.args.dataset}"
            )

    def training_step(self, batch, batch_idx):
        data, label = batch
        label = label.squeeze().long()

        logits = self(data)
        self._check_labels(label, logits)

        loss = self.criterion(logits, label)

        if torch.isnan(loss):
            raise RuntimeError(
                f"FATAL ERROR: NaN loss detected at epoch {self.current_epoch}"
            )

        preds = logits.argmax(dim=1)

        self.train_acc.update(preds, label)
        self.train_avg_acc.update(preds, label)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=data.size(0),
        )

        return loss

    def on_train_epoch_end(self):
        train_acc = self.train_acc.compute()
        train_avg_acc = self.train_avg_acc.compute()

        self.log("train_acc", train_acc, prog_bar=True, sync_dist=True)
        self.log("train_avg_acc", train_avg_acc, prog_bar=False, sync_dist=True)

        self.train_acc.reset()
        self.train_avg_acc.reset()

    def validation_step(self, batch, batch_idx):
        data, label = batch
        label = label.squeeze().long()

        logits = self(data)
        self._check_labels(label, logits)

        loss = self.criterion(logits, label)
        preds = logits.argmax(dim=1)

        self.val_acc.update(preds, label)
        self.val_avg_acc.update(preds, label)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=data.size(0),
        )

        return loss

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        val_avg_acc = self.val_avg_acc.compute()

        self.log("val_acc", val_acc, prog_bar=True, sync_dist=True)
        self.log("val_avg_acc", val_avg_acc, prog_bar=False, sync_dist=True)

        self.val_acc.reset()
        self.val_avg_acc.reset()

    def test_step(self, batch, batch_idx):
        data, label = batch
        label = label.squeeze().long()

        logits = self(data)
        self._check_labels(label, logits)

        preds = logits.argmax(dim=1)

        self.test_acc.update(preds, label)
        self.test_avg_acc.update(preds, label)

    def on_test_epoch_end(self):
        test_acc = self.test_acc.compute()
        test_avg_acc = self.test_avg_acc.compute()

        self.log("test_acc", test_acc, prog_bar=True, sync_dist=True)
        self.log("test_avg_acc", test_avg_acc, prog_bar=True, sync_dist=True)

        self.test_acc.reset()
        self.test_avg_acc.reset()


class TrainLogCallback(Callback):
    def __init__(self, io):
        super().__init__()
        self.io = io

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        if "train_loss" not in metrics:
            return

        train_loss = float(metrics["train_loss"].detach().cpu())
        train_acc = (
            float(metrics["train_acc"].detach().cpu())
            if "train_acc" in metrics
            else -1.0
        )
        train_avg_acc = (
            float(metrics["train_avg_acc"].detach().cpu())
            if "train_avg_acc" in metrics
            else -1.0
        )

        self.io.cprint(
            "Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f"
            % (epoch, train_loss, train_acc, train_avg_acc)
        )


def load_state_dict_robust(model: nn.Module, path: str, map_location):
    state = torch.load(path, map_location=map_location)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=True)


def run_train(args, io):
    datamodule = ModelNetDataModule(args)
    lit_model = LitModelNetClassifier(args)

    accelerator = "gpu" if args.cuda else "cpu"
    devices = args.devices if args.cuda else 1

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        deterministic=True,
        callbacks=[TrainLogCallback(io)],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )

    # Train on the full training split. The test split is not touched
    # during optimization and is evaluated exactly once after training.
    trainer.fit(lit_model, datamodule=datamodule)

    metrics = trainer.callback_metrics
    train_acc = (
        float(metrics["train_acc"].detach().cpu())
        if "train_acc" in metrics
        else float("nan")
    )

    final_model_path = os.path.join(
        "checkpoints",
        args.exp_name,
        "models",
        "model.pt",
    )
    torch.save(lit_model.model.state_dict(), final_model_path)

    test_results = trainer.test(
        lit_model,
        datamodule=datamodule,
        verbose=False,
    )
    test_acc = float(test_results[0]["test_acc"])
    gen_gap = train_acc - test_acc

    return {
        "seed": args.seed,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "gen_gap": gen_gap,
    }

def run_train_multiple_seeds(args, io):
    seeds = args.seeds
    all_results = []
    base_exp_name = args.exp_name

    for seed in seeds:
        run_args = copy.deepcopy(args)
        run_args.seed = int(seed)
        run_args.exp_name = f"{base_exp_name}_seed{seed}"

        _init_(run_args)

        seed_io = IOStream(
            os.path.join("checkpoints", run_args.exp_name, "run.log")
        )
        set_seed(run_args.seed)

        result = run_train(run_args, seed_io)
        all_results.append(result)

    test_accs = np.array(
        [r["test_acc"] for r in all_results],
        dtype=np.float64,
    )
    gen_gaps = np.array(
        [r["gen_gap"] for r in all_results],
        dtype=np.float64,
    )

    ddof = 1 if len(all_results) > 1 else 0

    test_mean = float(np.mean(test_accs))
    test_std = float(np.std(test_accs, ddof=ddof))
    gap_mean = float(np.mean(gen_gaps))
    gap_std = float(np.std(gen_gaps, ddof=ddof))

    io.cprint("")
    io.cprint("========== FINAL SUMMARY ==========")
    io.cprint(f"Dataset: {args.dataset}")
    io.cprint(f"Model: {args.ordering}")
    io.cprint(
        "Test accuracy: %.6f ± %.6f"
        % (test_mean, test_std)
    )
    io.cprint(
        "Generalization gap (train - test): %.6f ± %.6f"
        % (gap_mean, gap_std)
    )
    io.cprint("===================================")

    # Machine-readable per-seed results and mean/std for the Table 2 runner.
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "ordering": args.ordering,
        "seeds": [int(result["seed"]) for result in all_results],
        "test_acc_mean": test_mean,
        "test_acc_std": test_std,
        "gen_gap_mean": gap_mean,
        "gen_gap_std": gap_std,
        "runs": all_results,
    }
    summary_path = os.path.join(
        "checkpoints", base_exp_name, "summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    io.cprint(f"Saved summary: {summary_path}")

    return all_results

def run_test(args, io):
    datamodule = ModelNetDataModule(args)
    lit_model = LitModelNetClassifier(args)

    device = torch.device("cuda" if args.cuda else "cpu")
    load_state_dict_robust(lit_model.model, args.model_path, map_location=device)

    accelerator = "gpu" if args.cuda else "cpu"
    devices = args.devices if args.cuda else 1

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    results = trainer.test(lit_model, datamodule=datamodule, verbose=False)

    if results:
        test_acc = results[0].get("test_acc", None)
        test_avg_acc = results[0].get("test_avg_acc", None)

        if test_acc is not None and test_avg_acc is not None:
            io.cprint(
                "Test :: test acc: %.6f, test avg acc: %.6f"
                % (test_acc, test_avg_acc)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-ordered Point Cloud Recognition")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/modelnet.json",
        help="JSON config file. The block is selected by --ordering.",
    )

    parser.add_argument("--exp_name", type=str, default="Best_MLP_Hilbert")

    parser.add_argument(
        "--ordering",
        type=str,
        default="hilbert",
        choices=["lex", "hilbert", "ply", "pca"],
        help="Ordering mode. Also selects the matching config block when ordering is ply/lex/hilbert.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="global_mlp",
        choices=["global_mlp", "point_transformer"],
    )

    parser.add_argument("--dataset_stride", type=int, default=1)
    parser.add_argument("--use_fps", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--apply_jitter", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--apply_scale", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument(
        "--apply_random_permutation",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--apply_rotation", type=str2bool, nargs="?", const=True, default=False)

    parser.add_argument("--trans_dim", type=int, default=216)
    parser.add_argument("--trans_depth", type=int, default=4)
    parser.add_argument("--trans_heads", type=int, default=6)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)

    # Shared defaults across ply / lex / hilbert configs:
    parser.add_argument("--num_bands", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_points", type=int, default=1024)

    # These are loaded from presets.json according to --ordering,
    # unless explicitly overridden in the CLI.
    parser.add_argument("--fourier_scale", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.0009614328324244756)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "sgd"],
    )
    parser.add_argument("--use_sgd", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument(
        "--dataset",
        type=str,
        default="modelnet10",
        choices=["modelnet40", "modelnet10"],
    )

    parser.add_argument("--eval", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--no_cuda", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--model_path", type=str, default="")

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--pin_memory", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPU devices for Lightning. Use 1 unless you intentionally want multi-GPU.",
    )

    parser.add_argument(
        "--run_5_seeds",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Run training for 5 seeds and report mean/std.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seeds to run when --run_5_seeds is True.",
    )

    explicit_args = cli_overrides(sys.argv[1:])
    args = parser.parse_args()
    args = apply_ordering_config(args, explicit_args)

    _init_(args)

    io = IOStream(os.path.join("checkpoints", args.exp_name, "run.log"))
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    set_seed(args.seed)

    if args.cuda:
        io.cprint(f"Using GPU with devices={args.devices}")
    else:
        io.cprint("Using CPU")

    cfg = get_dataset_config(args.dataset)
    io.cprint(f"Dataset: {args.dataset}")
    io.cprint(f"Dataset folder: {cfg['dataset_name']}")
    io.cprint(f"Num classes: {cfg['num_classes']}")

    if args.ordering in ["ply", "lex", "hilbert"]:
        io.cprint(f"Loaded config by ordering: {args.ordering}")
        io.cprint(f"Config: {args.config}")

    if not args.eval:
        if args.run_5_seeds:
            run_train_multiple_seeds(args, io)
        else:
            run_train(args, io)
    else:
        run_test(args, io)