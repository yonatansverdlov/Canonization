# run_all_options.py
#
# Sweeps ALL configs across 4 model-variants:
#   1) mlp_id
#   2) mlp_sort
#   3) mlp_hilbert
#   4) deepsets  (NO permutation applied)
#
# Grid search over:
#   batch_size ∈ {16, 32, 64}
#   hidden_dims ∈ 4 options
#   lr ∈ 4 options
#   weight_decay ∈ {0.0, 1e-5}
# plus:
#   MLP also sweeps perm_mode ∈ {id, sort, hilbert}
#   DeepSets also sweeps embed_dim over "all hidden dims" scale:
#       embed_dims = sorted({hd[-1] for hd in hidden_dims_list})  -> typically [128,256,512]
#
# Loads caches created by data_creation:
#   <repo>/data_creation/modelnet{10|40}_train_P{P}_hilbm12_norm_with_perms.pt
#   <repo>/data_creation/modelnet{10|40}_test_P{P}_hilbm12_norm_with_perms.pt
#
# CLI args: ONLY
#   --P
#   --dataset_name {10,40}
#
# Outputs:
#   - CSV with one row per run (includes train/val/test loss+acc on BEST checkpoint)
#   - Prints best hyperparams per model-variant (by best_val_acc)
#   - Saves JSON with best configs for all 4 model-variants, including P and dataset_name
#
# NOTE about "train nan":
# We evaluate TRAIN metrics via trainer.validate(..., ckpt_path="best") on train_loader.
# Lightning still returns keys 'val_loss'/'val_acc' for validate(); we remap to train_*.
# If you see train nan, it's almost always because:
#   - no batches were processed (empty loader), or
#   - something crashed silently, or
#   - metric keys missing due to unusual Lightning settings.
# This script handles key-mapping robustly and prints the "last rows" from CSV if needed.

import os
import csv
import time
import math
import json
import random
import itertools
import argparse
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

PermMode = Literal["id", "sort", "hilbert"]
ModelType = Literal["mlp", "deepsets"]


# -------------------------
# Data loading
# -------------------------
def load_cache_dict(path: str) -> Dict[int, Data]:
    return torch.load(path, map_location="cpu", weights_only=False)


class CachedDataDictDataset(Dataset):
    def __init__(self, cache: Dict[int, Data]):
        self.ids = sorted(cache.keys())
        self.cache = cache
        d0 = self.cache[self.ids[0]]
        self.P = int(d0.x.shape[0])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Data:
        return self.cache[self.ids[idx]]


def split_indices(n: int, val_frac: float = 0.2, seed: int = 0) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = int(round(val_frac * n))
    return idx[n_val:], idx[:n_val]


# -------------------------
# Common utilities
# -------------------------
def apply_perm(X: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    # X: (B,P,3), perm: (B,P)
    return X.gather(1, perm.unsqueeze(-1).expand(-1, -1, 3))


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=1) == y).float().mean()


def _pretty(metrics_list: Any) -> dict:
    """
    Lightning validate/test return a list[dict]. Convert tensors->float.
    """
    if not metrics_list:
        return {}
    d = metrics_list[0]
    out = {}
    for k, v in d.items():
        out[k] = float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)
    return out


def print_metrics_table(train_m: dict, val_m: dict, test_m: dict):
    def g(d, k):
        try:
            return float(d.get(k, float("nan")))
        except Exception:
            return float("nan")

    rows = [
        ("train", g(train_m, "train_loss"), g(train_m, "train_acc")),
        ("val",   g(val_m,   "val_loss"),   g(val_m,   "val_acc")),
        ("test",  g(test_m,  "test_loss"),  g(test_m,  "test_acc")),
    ]

    print("\n+-------+------------+----------+")
    print("| split | loss       | acc      |")
    print("+-------+------------+----------+")
    for name, loss, acc in rows:
        print(f"| {name:<5} | {loss:>10.6f} | {acc:>8.6f} |")
    print("+-------+------------+----------+\n")


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool):
    return PyGDataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        drop_last=drop_last,
    )


# -------------------------
# MLP model (perm-dependent)
# -------------------------
class FlattenMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], num_classes: int, use_bn: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        return self.net(x_flat)


class PointCloudMLPLit(pl.LightningModule):
    def __init__(
        self,
        P: int,
        num_classes: int,
        perm_mode: PermMode,
        hidden_dims: Tuple[int, ...],
        lr: float,
        weight_decay: float,
        use_bn: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.P = P
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        self.perm_attr = {
            "id": "id_perm",
            "sort": "sort_perm",
            "hilbert": "hilbert_perm",
        }[perm_mode]

        self.mlp = FlattenMLP(in_dim=3 * P, hidden_dims=hidden_dims, num_classes=num_classes, use_bn=use_bn)

    def forward(self, batch: Data) -> torch.Tensor:
        B = batch.num_graphs
        X = batch.x.view(B, self.P, 3)
        perm = getattr(batch, self.perm_attr).view(B, self.P)
        Xp = apply_perm(X, perm)
        Xf = Xp.reshape(B, 3 * self.P)
        return self.mlp(Xf)

    def _step(self, batch: Data, stage: str):
        y = batch.y.view(-1)
        logits = self(batch)
        loss = F.cross_entropy(logits, y)
        acc = accuracy_top1(logits, y)
        B = int(batch.num_graphs)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=(stage == "train"), batch_size=B)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, on_epoch=True, on_step=(stage == "train"), batch_size=B)
        return loss

    def training_step(self, batch: Data, batch_idx: int):
        return self._step(batch, "train")

    def validation_step(self, batch: Data, batch_idx: int):
        self._step(batch, "val")

    def test_step(self, batch: Data, batch_idx: int):
        self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# -------------------------
# DeepSets model (perm-invariant)
# -------------------------
class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], out_dim: int, use_bn: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepSetsLit(pl.LightningModule):
    """
    phi: (3)->(embed_dim) per-point
    pool: sum or mean over points
    rho: (embed_dim)->(num_classes)
    """
    def __init__(
        self,
        P: int,
        num_classes: int,
        phi_hidden: Tuple[int, ...],
        rho_hidden: Tuple[int, ...],
        embed_dim: int,
        lr: float,
        weight_decay: float,
        use_bn: bool = True,
        pool: str = "sum",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.P = P
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.pool = pool

        self.phi = MLPBlock(in_dim=3, hidden_dims=phi_hidden, out_dim=embed_dim, use_bn=use_bn)
        self.rho = MLPBlock(in_dim=embed_dim, hidden_dims=rho_hidden, out_dim=num_classes, use_bn=use_bn)

    def forward(self, batch: Data) -> torch.Tensor:
        B = batch.num_graphs
        X = batch.x.view(B, self.P, 3)           # (B,P,3)
        Xflat = X.reshape(B * self.P, 3)         # (B*P,3)
        Z = self.phi(Xflat).view(B, self.P, -1)  # (B,P,E)

        if self.pool == "mean":
            S = Z.mean(dim=1)
        else:
            S = Z.sum(dim=1)

        return self.rho(S)

    def _step(self, batch: Data, stage: str):
        y = batch.y.view(-1)
        logits = self(batch)
        loss = F.cross_entropy(logits, y)
        acc = accuracy_top1(logits, y)
        B = int(batch.num_graphs)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=(stage == "train"), batch_size=B)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, on_epoch=True, on_step=(stage == "train"), batch_size=B)
        return loss

    def training_step(self, batch: Data, batch_idx: int):
        return self._step(batch, "train")

    def validation_step(self, batch: Data, batch_idx: int):
        self._step(batch, "val")

    def test_step(self, batch: Data, batch_idx: int):
        self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# -------------------------
# Sweep config + CSV helpers
# -------------------------
@dataclass
class SweepConfig:
    train_path: str
    test_path: str
    P_expected: int
    num_workers: int = 0
    seed: int = 0
    max_epochs: int = 100
    use_bn: bool = True
    out_dir: str = "results/runs"
    results_csv: str = "results/sweep_results_v5.csv"


def ensure_csv_header(path: str, fieldnames: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        with open(path, "r", newline="") as f:
            header = next(csv.reader(f))
        missing = [c for c in fieldnames if c not in header]
        if missing:
            raise RuntimeError(
                f"CSV header missing columns {missing}. Delete {path} or set a new results_csv filename."
            )
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_row(path: str, fieldnames: List[str], row: dict):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


# -------------------------
# Run one experiment (MLP or DeepSets)
# -------------------------
def run_one(
    *,
    cfg: SweepConfig,
    model_type: ModelType,
    perm_mode: Optional[PermMode],
    batch_size: int,
    hidden_dims: Tuple[int, ...],
    embed_dim: int,  # used only for deepsets
    lr: float,
    weight_decay: float,
    train_ds,
    val_ds,
    test_ds,
    P: int,
    num_classes: int,
) -> dict:
    hd_str = "-".join(map(str, hidden_dims))

    if model_type == "mlp":
        assert perm_mode is not None
        run_name = f"model=mlp_perm={perm_mode}_bs={batch_size}_hd={hd_str}_lr={lr:g}_wd={weight_decay:g}"
    else:
        run_name = f"model=deepsets_bs={batch_size}_phi={hd_str}_rho={hd_str}_E={embed_dim}_lr={lr:g}_wd={weight_decay:g}"

    run_dir = os.path.join(cfg.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath=run_dir,
        filename="best-{epoch:03d}-{val_acc:.4f}",
    )

    if model_type == "mlp":
        model = PointCloudMLPLit(
            P=P,
            num_classes=num_classes,
            perm_mode=perm_mode,  # type: ignore
            hidden_dims=hidden_dims,
            lr=lr,
            weight_decay=weight_decay,
            use_bn=cfg.use_bn,
        )
    else:
        model = DeepSetsLit(
            P=P,
            num_classes=num_classes,
            phi_hidden=hidden_dims,
            rho_hidden=hidden_dims,
            embed_dim=embed_dim,
            lr=lr,
            weight_decay=weight_decay,
            use_bn=cfg.use_bn,
            pool="sum",
        )

    train_loader = make_loader(train_ds, batch_size, shuffle=True,  num_workers=cfg.num_workers, drop_last=True)
    val_loader   = make_loader(val_ds,   batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)
    test_loader  = make_loader(test_ds,  batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[ckpt_cb],
        log_every_n_steps=20,
        default_root_dir=run_dir,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    t0 = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    fit_minutes = (time.time() - t0) / 60.0

    # ---- Evaluate on BEST checkpoint ----
    # validate() returns keys: val_loss/val_acc, regardless of dataloader name.
    train_raw = _pretty(trainer.validate(dataloaders=train_loader, ckpt_path="best", verbose=False))
    train_metrics = {
        "train_loss": train_raw.get("val_loss", float("nan")),
        "train_acc":  train_raw.get("val_acc",  float("nan")),
    }
    val_metrics = _pretty(trainer.validate(dataloaders=val_loader, ckpt_path="best", verbose=False))
    test_metrics = _pretty(trainer.test(dataloaders=test_loader, ckpt_path="best", verbose=False))

    best_path = ckpt_cb.best_model_path
    best_val = ckpt_cb.best_model_score.item() if ckpt_cb.best_model_score is not None else float("nan")

    row = {
        "model": model_type,
        "perm_mode": (perm_mode if model_type == "mlp" else ""),
        "batch_size": batch_size,
        "hidden_dims": str(tuple(hidden_dims)),
        "embed_dim": (embed_dim if model_type == "deepsets" else ""),
        "lr": lr,
        "weight_decay": weight_decay,
        "best_val_acc": float(best_val),
        "best_ckpt_path": best_path,
        "fit_minutes": float(fit_minutes),
        "run_name": run_name,

        "train_acc": float(train_metrics.get("train_acc", float("nan"))),
        "train_loss": float(train_metrics.get("train_loss", float("nan"))),
        "val_acc": float(val_metrics.get("val_acc", float("nan"))),
        "val_loss": float(val_metrics.get("val_loss", float("nan"))),
        "test_acc": float(test_metrics.get("test_acc", float("nan"))),
        "test_loss": float(test_metrics.get("test_loss", float("nan"))),
    }
    return row


# -------------------------
# Best selection + saving configs
# -------------------------
def find_best_by_val_acc_per_variant(results_csv: str) -> Dict[str, dict]:
    """
    Returns best row for each variant:
      - mlp_id, mlp_sort, mlp_hilbert, deepsets
    Selection: max(best_val_acc), tie-breakers:
      1) higher test_acc
      2) lower val_loss
      3) lower fit_minutes
    """
    def f(row, key, default=float("nan")):
        try:
            return float(row.get(key, default))
        except Exception:
            return default

    best_row: Dict[str, dict] = {}
    best_key: Dict[str, tuple] = {}

    with open(results_csv, "r", newline="") as fcsv:
        reader = csv.DictReader(fcsv)
        for row in reader:
            model = row.get("model", "")
            perm = row.get("perm_mode", "")

            if model == "mlp":
                variant = f"mlp_{perm}"
            elif model == "deepsets":
                variant = "deepsets"
            else:
                continue

            bva = f(row, "best_val_acc")
            if math.isnan(bva):
                continue

            test_acc = f(row, "test_acc")
            val_loss = f(row, "val_loss")
            fit_min  = f(row, "fit_minutes")

            k = (
                bva,
                (test_acc if not math.isnan(test_acc) else -1.0),
                (-val_loss if not math.isnan(val_loss) else -float("inf")),
                (-fit_min  if not math.isnan(fit_min)  else -float("inf")),
            )

            if (variant not in best_key) or (k > best_key[variant]):
                best_key[variant] = k
                best_row[variant] = row

    return best_row


def print_best_summary_for_all(best: Dict[str, dict]):
    order = ["mlp_id", "mlp_sort", "mlp_hilbert", "deepsets"]
    for name in order:
        row = best.get(name)
        if row is None:
            print(f"\n(no valid rows for {name})")
            continue

        print(f"\n================ BEST FOR {name} (by best_val_acc) ================")
        for k in ["model", "perm_mode", "batch_size", "hidden_dims", "embed_dim", "lr", "weight_decay",
                  "best_val_acc", "best_ckpt_path", "fit_minutes"]:
            v = row.get(k, "")
            if v != "" and v is not None:
                print(f"{k:12s}: {v}")
        print("---------------------------------------------------------------")

        train_m = {"train_loss": float(row.get("train_loss", float("nan"))),
                   "train_acc":  float(row.get("train_acc",  float("nan")))}
        val_m   = {"val_loss":   float(row.get("val_loss",   float("nan"))),
                   "val_acc":    float(row.get("val_acc",    float("nan")))}
        test_m  = {"test_loss":  float(row.get("test_loss",  float("nan"))),
                   "test_acc":   float(row.get("test_acc",   float("nan")))}
        print_metrics_table(train_m, val_m, test_m)


def save_best_configs_json(path: str, best: Dict[str, dict], dataset_name: str, P: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    out = {
        "dataset_name": dataset_name,
        "P": P,
        "best_configs": {}
    }

    for variant, row in best.items():
        # keep strings as-is for hidden_dims because you were saving tuple-strings in CSV
        out["best_configs"][variant] = {
            "hyperparams": {
                "model": row.get("model"),
                "perm_mode": row.get("perm_mode", ""),
                "batch_size": int(float(row.get("batch_size", "0"))),
                "hidden_dims": row.get("hidden_dims"),
                "embed_dim": row.get("embed_dim", ""),
                "lr": float(row.get("lr", "nan")),
                "weight_decay": float(row.get("weight_decay", "nan")),
            },
            "metrics_best_ckpt": {
                "best_val_acc": float(row.get("best_val_acc", "nan")),
                "train_loss": float(row.get("train_loss", "nan")),
                "train_acc": float(row.get("train_acc", "nan")),
                "val_loss": float(row.get("val_loss", "nan")),
                "val_acc": float(row.get("val_acc", "nan")),
                "test_loss": float(row.get("test_loss", "nan")),
                "test_acc": float(row.get("test_acc", "nan")),
            },
            "best_ckpt_path": row.get("best_ckpt_path", ""),
            "run_name": row.get("run_name", ""),
            "fit_minutes": float(row.get("fit_minutes", "nan")),
        }

    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved best configs JSON -> {path}\n")


# -------------------------
# CLI (ONLY P and dataset_name)
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Sweep MLP perm modes + DeepSets (sweep embed_dim too)")
    parser.add_argument("--P", type=int, required=True)
    parser.add_argument("--dataset_name", type=str, required=True, choices=["10", "40"])
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve data_creation path robustly relative to THIS script file
    HERE = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(HERE)              # assumes script is in <repo>/trainings/
    DATA_CREATION_DIR = os.path.join(REPO_ROOT, "data_creation")

    hilbert_m = 12
    tag = f"modelnet{args.dataset_name}"
    train_path = os.path.join(DATA_CREATION_DIR, f"{tag}_train_P{args.P}_hilbm{hilbert_m}_norm_with_perms.pt")
    test_path  = os.path.join(DATA_CREATION_DIR, f"{tag}_test_P{args.P}_hilbm{hilbert_m}_norm_with_perms.pt")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train cache: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test cache: {test_path}")

    num_classes = 10 if args.dataset_name == "10" else 40

    cfg = SweepConfig(
        train_path=train_path,
        test_path=test_path,
        P_expected=args.P,
        num_workers=0,
        seed=0,
        max_epochs=100,
        use_bn=True,
        out_dir=os.path.join(REPO_ROOT, "results", "runs"),
        results_csv=os.path.join(REPO_ROOT, "results", "sweep_results_v5.csv"),
    )

    # ---- GRID ----
    perm_modes: List[PermMode] = ["id", "sort", "hilbert"]
    batch_sizes = [16, 32, 64]
    hidden_dims_list = [
        (512, 256, 128),
        (1024, 512, 256),
        (2048, 1024, 512),
        (1024, 1024, 512, 256),
    ]
    lrs = [1e-4, 3e-4, 1e-3, 3e-3]
    weight_decays = [0.0, 1e-5]

    # DeepSets embed_dim sweep: "all hidden dims" scale
    embed_dims = sorted({hd[-1] for hd in hidden_dims_list})  # typically [128,256,512]
    print("DeepSets embed_dims:", embed_dims)

    fieldnames = [
        "model", "perm_mode", "batch_size", "hidden_dims", "embed_dim", "lr", "weight_decay",
        "best_val_acc", "best_ckpt_path",
        "train_acc", "train_loss", "val_acc", "val_loss", "test_acc", "test_loss",
        "fit_minutes", "run_name"
    ]
    ensure_csv_header(cfg.results_csv, fieldnames)

    pl.seed_everything(cfg.seed, workers=True)

    train_cache = load_cache_dict(cfg.train_path)
    test_cache = load_cache_dict(cfg.test_path)

    train_ds_full = CachedDataDictDataset(train_cache)
    test_ds = CachedDataDictDataset(test_cache)

    P = train_ds_full.P
    if P != cfg.P_expected:
        raise RuntimeError(f"P mismatch: cache has P={P} but expected P={cfg.P_expected}")

    train_idx, val_idx = split_indices(len(train_ds_full), val_frac=0.2, seed=cfg.seed)
    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(train_ds_full, val_idx)

    os.makedirs(cfg.out_dir, exist_ok=True)

    # MLP grid: 3*3*4*4*2 = 288
    mlp_grid = list(itertools.product(perm_modes, batch_sizes, hidden_dims_list, lrs, weight_decays))

    # DeepSets grid: 3*4*|E|*4*2 = 96*|E| => for |E|=3 => 288
    ds_grid = list(itertools.product(batch_sizes, hidden_dims_list, embed_dims, lrs, weight_decays))

    total_runs = len(mlp_grid) + len(ds_grid)
    print(f"Total runs: {total_runs} (MLP={len(mlp_grid)}, DeepSets={len(ds_grid)})")

    # ---- Run MLP variants ----
    for perm_mode, bs, hidden_dims, lr, wd in mlp_grid:
        print(f"\n=== RUN model=mlp perm={perm_mode} bs={bs} hd={hidden_dims} lr={lr} wd={wd} ===")
        row = run_one(
            cfg=cfg,
            model_type="mlp",
            perm_mode=perm_mode,
            batch_size=bs,
            hidden_dims=hidden_dims,
            embed_dim=0,  # ignored for MLP
            lr=lr,
            weight_decay=wd,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            P=P,
            num_classes=num_classes,
        )
        append_csv_row(cfg.results_csv, fieldnames, row)
        print("RESULT:", row)

    # ---- Run DeepSets variants ----
    for bs, hidden_dims, E, lr, wd in ds_grid:
        print(f"\n=== RUN model=deepsets bs={bs} phi/rho={hidden_dims} E={E} lr={lr} wd={wd} ===")
        row = run_one(
            cfg=cfg,
            model_type="deepsets",
            perm_mode=None,
            batch_size=bs,
            hidden_dims=hidden_dims,
            embed_dim=E,
            lr=lr,
            weight_decay=wd,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            P=P,
            num_classes=num_classes,
        )
        append_csv_row(cfg.results_csv, fieldnames, row)
        print("RESULT:", row)

    # ---- Print best per variant + save JSON ----
    best = find_best_by_val_acc_per_variant(cfg.results_csv)
    print_best_summary_for_all(best)

    best_json_path = os.path.join(REPO_ROOT, "results", f"best_configs_modelnet{args.dataset_name}_P{args.P}.json")
    save_best_configs_json(best_json_path, best, dataset_name=args.dataset_name, P=args.P)


if __name__ == "__main__":
    main()