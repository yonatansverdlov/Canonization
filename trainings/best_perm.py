# train_phi_then_flatten_with_val.py

from __future__ import annotations
import os, glob, time, argparse, random
from typing import Dict, List, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

PermMode = Literal["id", "sort", "hilbert"]


# -------------------------
# Cache utilities
# -------------------------
def load_cache_dict(path: str) -> Dict[int, Data]:
    return torch.load(path, map_location="cpu", weights_only=False)


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
        raise FileNotFoundError(f"Missing train cache '{train_pat}'. Searched:\n" + "\n".join("  "+d for d in candidates))
    if not test_hits:
        raise FileNotFoundError(f"Missing test cache '{test_pat}'. Searched:\n" + "\n".join("  "+d for d in candidates))

    return train_hits[0], test_hits[0]


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


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool):
    return PyGDataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
    )


def split_indices(n: int, val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = int(round(val_frac * n))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


# -------------------------
# Model bits
# -------------------------
def apply_perm(X: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return X.gather(1, perm.unsqueeze(-1).expand(-1, -1, X.size(-1)))


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=1) == y).float().mean()


def _parse_int_tuple(s: str) -> Tuple[int, ...]:
    s = s.strip()
    if not s:
        return tuple()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


class PhiThenFlattenNet(nn.Module):
    def __init__(
        self,
        P: int,
        num_classes: int,
        phi_hidden: Tuple[int, ...] = (64, 64),
        phi_dim: int = 16,
        head_hidden: Tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        use_tanh: bool = True,
    ):
        super().__init__()
        self.P = P
        self.phi_dim = phi_dim
        Act = nn.Tanh if use_tanh else nn.ReLU

        phi_layers: List[nn.Module] = []
        prev = 3
        for h in phi_hidden:
            phi_layers += [nn.Linear(prev, h), Act()]
            prev = h
        phi_layers += [nn.Linear(prev, phi_dim), Act()]
        self.phi = nn.Sequential(*phi_layers)

        head_layers: List[nn.Module] = []
        prev = P * phi_dim
        for h in head_hidden:
            head_layers += [nn.Linear(prev, h), Act(), nn.Dropout(dropout)]
            prev = h
        head_layers += [nn.Linear(prev, num_classes)]
        self.head = nn.Sequential(*head_layers)

    def forward(self, X: torch.Tensor, perm: torch.Tensor | None = None) -> torch.Tensor:
        B, P, _ = X.shape
        if P != self.P:
            raise RuntimeError(f"P mismatch in forward: got {P}, expected {self.P}")

        Z = self.phi(X.reshape(B * P, 3)).reshape(B, P, self.phi_dim)  # [B,P,D]
        if perm is not None:
            Z = apply_perm(Z, perm)
        Zf = Z.reshape(B, P * self.phi_dim)
        return self.head(Zf)


class LitPermMLP(pl.LightningModule):
    def __init__(
        self,
        *,
        P: int,
        num_classes: int,
        perm_mode: PermMode,
        phi_hidden: Tuple[int, ...],
        phi_dim: int,
        head_hidden: Tuple[int, ...],
        dropout: float,
        use_tanh: bool,
        lr: float,
        weight_decay: float,
        scheduler_patience: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.P = P
        self.perm_attr = {"id": "id_perm", "sort": "sort_perm", "hilbert": "hilbert_perm"}[perm_mode]

        self.net = PhiThenFlattenNet(
            P=P,
            num_classes=num_classes,
            phi_hidden=phi_hidden,
            phi_dim=phi_dim,
            head_hidden=head_hidden,
            dropout=dropout,
            use_tanh=use_tanh,
        )

    def forward(self, batch: Data) -> torch.Tensor:
        B = batch.num_graphs
        X = batch.x.view(B, self.P, 3).to(torch.float32)          # NO normalization
        perm = getattr(batch, self.perm_attr).view(B, self.P)     # cached perm
        return self.net(X, perm=perm)

    def _step(self, batch: Data, stage: str):
        y = batch.y.view(-1)
        logits = self(batch)
        loss = F.cross_entropy(logits, y)
        acc = accuracy_top1(logits, y)
        B = int(batch.num_graphs)

        # step+epoch logging for train (live bar), epoch for val/test
        on_step = (stage == "train")
        self.log(f"{stage}_loss", loss, prog_bar=True, logger=True,
                 on_step=on_step, on_epoch=True, batch_size=B)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, logger=True,
                 on_step=on_step, on_epoch=True, batch_size=B)
        return loss

    def training_step(self, batch: Data, batch_idx: int):
        return self._step(batch, "train")

    def validation_step(self, batch: Data, batch_idx: int):
        self._step(batch, "val")

    def test_step(self, batch: Data, batch_idx: int):
        self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=self.hparams.scheduler_patience
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss", "interval": "epoch", "frequency": 1},
        }


# -------------------------
# CLI + main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--P", type=int, required=True)
    p.add_argument("--dataset_name", type=str, required=True, choices=["10", "40"])
    p.add_argument("--perm", type=str, default="hilbert", choices=["id", "sort", "hilbert"])

    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.10)

    # sizes (small defaults)
    p.add_argument("--phi_hidden", type=str, default="64,64")
    p.add_argument("--phi_dim", type=int, default=16)
    p.add_argument("--head_hidden", type=str, default="256,128")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--tanh", action="store_true", default=False)

    # opt + sched
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--scheduler_patience", type=int, default=1)

    # logging
    p.add_argument("--out_dir", type=str, default="runs_phi_flat_val10")
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    train_path, test_path = find_cache_paths(P=args.P, dataset_name=args.dataset_name)
    print("Using train cache:", train_path)
    print("Using test  cache:", test_path)

    train_cache = load_cache_dict(train_path)
    test_cache  = load_cache_dict(test_path)

    train_full = CachedDataDictDataset(train_cache)
    test_ds    = CachedDataDictDataset(test_cache)

    if train_full.P != args.P or test_ds.P != args.P:
        raise RuntimeError(f"P mismatch: train cache P={train_full.P}, test cache P={test_ds.P}, expected P={args.P}")

    num_classes = 10 if args.dataset_name == "10" else 40

    # ---- 10% validation split FROM TRAIN ONLY ----
    train_idx, val_idx = split_indices(len(train_full), val_frac=args.val_frac, seed=args.seed)
    train_ds = Subset(train_full, train_idx)
    val_ds   = Subset(train_full, val_idx)

    # loaders
    train_loader = make_loader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader   = make_loader(val_ds,   batch_size=args.batch, shuffle=False,
                               num_workers=args.num_workers, drop_last=False)
    test_loader  = make_loader(test_ds,  batch_size=args.batch, shuffle=False,
                               num_workers=args.num_workers, drop_last=False)

    model = LitPermMLP(
        P=args.P,
        num_classes=num_classes,
        perm_mode=args.perm,
        phi_hidden=_parse_int_tuple(args.phi_hidden),
        phi_dim=args.phi_dim,
        head_hidden=_parse_int_tuple(args.head_hidden),
        dropout=args.dropout,
        use_tanh=args.tanh,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
    )

    run_name = f"modelnet{args.dataset_name}_P{args.P}_perm{args.perm}"
    logger = TensorBoardLogger(save_dir=args.out_dir, name=run_name)
    lrmon = LearningRateMonitor(logging_interval="epoch")

    ckpt_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:03d}-{val_acc:.4f}",
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=logger,
        callbacks=[lrmon, ckpt_cb],
        enable_checkpointing=True,
        enable_progress_bar=True,
        log_every_n_steps=20,
        default_root_dir=os.path.join(args.out_dir, run_name),
    )

    print(f"\n=== TRAIN (val={args.val_frac:.0%}) | perm={args.perm} P={args.P} bs={args.batch} lr={args.lr} wd={args.weight_decay} ===")
    t0 = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Fit minutes: {(time.time()-t0)/60.0:.2f}")
    print("Best ckpt:", ckpt_cb.best_model_path)
    print("Best val_acc:", float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else float("nan"))

    # Evaluate best checkpoint
    print("\n=== FINAL EVAL (best checkpoint) ===")
    tr = trainer.test(dataloaders=train_loader, ckpt_path="best", verbose=False)[0]
    va = trainer.test(dataloaders=val_loader,   ckpt_path="best", verbose=False)[0]
    te = trainer.test(dataloaders=test_loader,  ckpt_path="best", verbose=False)[0]

    print(f"Train acc: {float(tr.get('test_acc', float('nan'))):.4f}   Train loss: {float(tr.get('test_loss', float('nan'))):.4f}")
    print(f"Val   acc: {float(va.get('test_acc', float('nan'))):.4f}   Val   loss: {float(va.get('test_loss', float('nan'))):.4f}")
    print(f"Test  acc: {float(te.get('test_acc', float('nan'))):.4f}   Test  loss: {float(te.get('test_loss', float('nan'))):.4f}")

    print("\nTensorBoard logdir:")
    print(" ", logger.log_dir)


if __name__ == "__main__":
    main()