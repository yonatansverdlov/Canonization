# train_from_config_and_report.py
from __future__ import annotations

import os
import glob
import json
import time
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

PermMode = Literal["id", "sort", "hilbert"]


# -------------------------
# Loading (PyTorch 2.6+)
# -------------------------
def load_cache_dict(path: str) -> Dict[int, Data]:
    return torch.load(path, map_location="cpu", weights_only=False)


# -------------------------
# Find caches from only (P, dataset_name)
# -------------------------
def find_cache_paths(P: int, dataset_name: str) -> Tuple[str, str]:
    tag = f"modelnet{dataset_name}"
    train_pat = f"{tag}_train_P{P}_hilbm*_norm_with_perms.pt"
    test_pat  = f"{tag}_test_P{P}_hilbm*_norm_with_perms.pt"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.abspath(os.path.join(script_dir, ".."))
    cwd = os.getcwd()

    candidates = [
        cwd,
        os.path.join(cwd, "data_creation"),
        script_dir,
        os.path.join(script_dir, "data_creation"),
        repo_root,
        os.path.join(repo_root, "data_creation"),
    ]

    def _glob_all(pattern: str) -> List[str]:
        hits: List[str] = []
        for d in candidates:
            hits.extend(glob.glob(os.path.join(d, pattern)))
        return sorted(set(hits))

    train_hits = _glob_all(train_pat)
    test_hits  = _glob_all(test_pat)

    if not train_hits:
        raise FileNotFoundError(
            f"Missing train cache pattern '{train_pat}'. Searched:\n" + "\n".join("  " + d for d in candidates)
        )
    if not test_hits:
        raise FileNotFoundError(
            f"Missing test cache pattern '{test_pat}'. Searched:\n" + "\n".join("  " + d for d in candidates)
        )

    preferred_root = os.path.join(repo_root, "data_creation")

    def _pick(hits: List[str]) -> str:
        pref = [h for h in hits if preferred_root in h]
        return pref[0] if pref else hits[0]

    return _pick(train_hits), _pick(test_hits)


# -------------------------
# Dataset that returns Data
# -------------------------
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


def _pretty(metrics_list):
    if not metrics_list:
        return {}
    d = metrics_list[0]
    out = {}
    for k, v in d.items():
        out[k] = float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)
    return out


# -------------------------
# Common utils
# -------------------------
def apply_perm(X: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return X.gather(1, perm.unsqueeze(-1).expand(-1, -1, 3))


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=1) == y).float().mean()


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool):
    return PyGDataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
    )


def _parse_hidden_dims(s: Any) -> Tuple[int, ...]:
    if isinstance(s, (list, tuple)):
        return tuple(int(x) for x in s)
    if isinstance(s, str):
        s2 = s.strip()
        if s2.startswith("(") and s2.endswith(")"):
            s2 = s2[1:-1].strip()
        if not s2:
            return tuple()
        parts = [p.strip() for p in s2.split(",") if p.strip()]
        return tuple(int(p) for p in parts)
    raise ValueError(f"Cannot parse hidden_dims from: {s}")


# -------------------------
# MLP (perm-dependent)
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

        # epoch metrics only (on_step=False) -> stable table later
        self.log(f"{stage}_loss", loss, prog_bar=False, on_epoch=True, on_step=False, batch_size=B)
        self.log(f"{stage}_acc",  acc,  prog_bar=False, on_epoch=True, on_step=False, batch_size=B)
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
# DeepSets (perm-invariant)
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

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=True, batch_size=B)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, on_epoch=True, on_step=True, batch_size=B)
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
# Config I/O
# -------------------------
def load_train_config(P: int, dataset_name: str, config_path: str | None) -> Tuple[dict, str]:
    if config_path is None:
        config_path = os.path.join("results", f"best_configs_modelnet{10}_P{1024}.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file: {os.path.abspath(config_path)}")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return cfg, config_path


def save_updated_config(cfg: dict, orig_path: str) -> str:
    base, ext = os.path.splitext(orig_path)
    out_path = base + "_retrained" + ext
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return out_path


# -------------------------
# Evaluation helpers (best checkpoint)
# -------------------------
def eval_on_best_ckpt(
    *,
    trainer: pl.Trainer,
    train_loader,
    test_loader,
    ckpt_path: str,
) -> Tuple[dict, dict]:
    # TRAIN via validate() => gives val_loss/val_acc keys
    tr_raw = _pretty(trainer.validate(dataloaders=train_loader, ckpt_path=ckpt_path, verbose=False))
    train_m = {
        "train_loss": tr_raw.get("val_loss", float("nan")),
        "train_acc":  tr_raw.get("val_acc",  float("nan")),
    }

    te_raw = _pretty(trainer.test(dataloaders=test_loader, ckpt_path=ckpt_path, verbose=False))
    test_m = {
        "test_loss": te_raw.get("test_loss", float("nan")),
        "test_acc":  te_raw.get("test_acc",  float("nan")),
    }
    return train_m, test_m


def print_final_table(rows: List[dict]):
    # rows: [{"name":..., "train_loss":..., ...}]
    headers = ["model", "train_loss", "train_acc", "test_loss", "test_acc"]
    print("\n=== FINAL TABLE (best val_acc checkpoint for each) ===")
    print("+---------------+------------+----------+------------+----------+")
    print("| model         | train_loss | train_acc | test_loss  | test_acc |")
    print("+---------------+------------+----------+------------+----------+")
    for r in rows:
        name = r["name"]
        tl = r.get("train_loss", float("nan"))
        ta = r.get("train_acc", float("nan"))
        tel = r.get("test_loss", float("nan"))
        tea = r.get("test_acc", float("nan"))
        print(f"| {name:<13} | {tl:>10.6f} | {ta:>8.6f} | {tel:>10.6f} | {tea:>8.6f} |")
    print("+---------------+------------+----------+------------+----------+\n")


# -------------------------
# CLI (ONLY P and dataset_name)
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--P", type=int, required=True)
    p.add_argument("--dataset_name", type=str, required=True, choices=["10", "40"])
    p.add_argument("--config", type=str, default=None, help="Path to JSON config (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    # locate caches
    train_path, test_path = find_cache_paths(P=args.P, dataset_name=args.dataset_name)
    print("Using train cache:", train_path)
    print("Using test  cache:", test_path)

    num_classes = 10 if args.dataset_name == "10" else 40

    # load config
    cfg, cfg_path = load_train_config(P=args.P, dataset_name=args.dataset_name, config_path=args.config)
    if str(cfg.get("dataset_name")) != str(args.dataset_name) or int(cfg.get("P")) != int(args.P):
        print("[warn] JSON dataset_name/P do not match CLI; continuing anyway.")

    seed = int(cfg.get("seed", 0))
    use_bn = bool(cfg.get("use_bn", True))
    best_configs = cfg.get("best_configs", {})

    # load datasets
    pl.seed_everything(seed, workers=True)
    train_cache = load_cache_dict(train_path)
    test_cache = load_cache_dict(test_path)
    train_ds_full = CachedDataDictDataset(train_cache)
    test_ds = CachedDataDictDataset(test_cache)

    if train_ds_full.P != args.P or test_ds.P != args.P:
        raise RuntimeError(f"P mismatch: cache train P={train_ds_full.P}, test P={test_ds.P}, expected P={args.P}")

    train_idx, val_idx = split_indices(len(train_ds_full), val_frac=0.2, seed=seed)
    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(train_ds_full, val_idx)

    # output dir
    out_root = cfg.get("out_dir", "results/retrained_runs")
    os.makedirs(out_root, exist_ok=True)

    final_rows: List[dict] = []

    # train each model described in config
    for name, entry in best_configs.items():
        hp = entry.get("hyperparams", {})
        model_type = hp.get("model", "")

        batch_size = int(hp["batch_size"])
        lr = float(hp["lr"])
        wd = float(hp["weight_decay"])
        max_epochs = int(cfg.get("max_epochs", 100))
        num_workers = int(cfg.get("num_workers", 0))

        # BN safety: drop_last=True for train if BN enabled
        drop_last = bool(use_bn)

        train_loader = make_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers, drop_last=drop_last)
        val_loader   = make_loader(val_ds,   batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        test_loader  = make_loader(test_ds,  batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

        run_dir = os.path.join(out_root, name)
        os.makedirs(run_dir, exist_ok=True)

        ckpt_cb = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            dirpath=run_dir,
            filename="best-{epoch:03d}-{val_acc:.4f}",
        )

        if model_type == "mlp":
            perm_mode: PermMode = hp["perm_mode"]
            hidden_dims = _parse_hidden_dims(hp["hidden_dims"])

            model = PointCloudMLPLit(
                P=args.P,
                num_classes=num_classes,
                perm_mode=perm_mode,
                hidden_dims=hidden_dims,
                lr=lr,
                weight_decay=wd,
                use_bn=use_bn,
            )
            nice_name = f"{name}"

        elif model_type == "deepsets":
            hidden_dims = _parse_hidden_dims(hp["hidden_dims"])
            embed_dim_str = str(hp.get("embed_dim", "")).strip()
            embed_dim = int(embed_dim_str) if embed_dim_str else int(hidden_dims[-1])

            model = DeepSetsLit(
                P=args.P,
                num_classes=num_classes,
                phi_hidden=hidden_dims,
                rho_hidden=hidden_dims,
                embed_dim=embed_dim,
                lr=lr,
                weight_decay=wd,
                use_bn=use_bn,
                pool="sum",
            )
            nice_name = f"{name}"

        else:
            print(f"[skip] unknown model type for {name}: {model_type}")
            continue

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices="auto",
            callbacks=[ckpt_cb],
            logger=False,
            enable_checkpointing=True,
            default_root_dir=run_dir,
            log_every_n_steps=50,
            
        )

        print(f"\n=== TRAIN {name} | model={model_type} bs={batch_size} lr={lr} wd={wd} ===")
        t0 = time.time()
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        fit_minutes = (time.time() - t0) / 60.0

        best_ckpt_path = ckpt_cb.best_model_path
        best_val_acc = ckpt_cb.best_model_score.item() if ckpt_cb.best_model_score is not None else float("nan")

        # evaluate train/test on best ckpt
        train_m, test_m = eval_on_best_ckpt(
            trainer=trainer,
            train_loader=train_loader,
            test_loader=test_loader,
            ckpt_path="best",
        )

        # store results
        row = {
            "name": nice_name,
            "train_loss": float(train_m.get("train_loss", float("nan"))),
            "train_acc":  float(train_m.get("train_acc",  float("nan"))),
            "test_loss":  float(test_m.get("test_loss",  float("nan"))),
            "test_acc":   float(test_m.get("test_acc",   float("nan"))),
        }
        final_rows.append(row)

        # update config entry
        entry["best_ckpt_path"] = best_ckpt_path
        entry["fit_minutes"] = fit_minutes
        entry["metrics_best_ckpt"] = {
            "best_val_acc": float(best_val_acc),
            "train_loss": float(row["train_loss"]),
            "train_acc":  float(row["train_acc"]),
            "test_loss":  float(row["test_loss"]),
            "test_acc":   float(row["test_acc"]),
        }

        print(f"Best val_acc: {best_val_acc:.6f}")
        print(f"Saved ckpt : {best_ckpt_path}")

    # print final table
    print_final_table(final_rows)

    # save updated config
    out_cfg_path = save_updated_config(cfg, cfg_path)
    print("Wrote updated config:", out_cfg_path)


if __name__ == "__main__":
    main()