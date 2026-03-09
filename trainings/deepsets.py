import os
import math
import numpy as np
import torch
import torch.nn as nn
import lightning as L
import argparse
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform, Compose
import torch_geometric.transforms as T


# -------------------------
# Repo-faithful transforms
# -------------------------
class FixedSubsample(BaseTransform):
    """
    Match repo behavior:
      - train: pick ONE fixed random permutation of indices, take [::downsample]
      - test : deterministic slice 1::downsample
    Assumes data.pos has at least base_points points (we'll make base_points=10000 via pre_transform).
    """
    def __init__(self, split: str, downsample: int, base_points: int = 10000, seed: int = 0):
        super().__init__()
        assert split in ("train", "test")
        self.split = split
        self.downsample = downsample
        self.base_points = base_points

        if split == "train":
            rng = np.random.RandomState(seed)
            perm = rng.permutation(base_points)[::downsample]
            self.idx = torch.from_numpy(perm.astype(np.int64))
        else:
            self.idx = slice(1, None, downsample)

    def forward(self, data):
        if data.pos.size(0) < self.base_points:
            raise RuntimeError(
                f"Expected at least {self.base_points} points before FixedSubsample, got {data.pos.size(0)}. "
                "Use pre_transform=SamplePoints(10000) to match repo."
            )
        data.pos = data.pos[: self.base_points]
        if isinstance(self.idx, slice):
            data.pos = data.pos[self.idx]
        else:
            data.pos = data.pos[self.idx]
        return data


class RepoStandardize(BaseTransform):
    """
    repo modelnet.py:
      clipper = mean(abs(x))
      clip to +/- 100*clipper
      z-score using mean/std over all coords
    """
    def __init__(self):
        super().__init__()

    def forward(self, data):
        x = data.pos
        clipper = x.abs().mean()
        x = x.clamp(-100.0 * clipper, 100.0 * clipper)
        mean = x.mean()
        std = x.std(unbiased=False).clamp(min=1e-12)
        data.pos = (x - mean) / std
        return data


class RepoAugment(BaseTransform):
    """
    repo augment:
      theta ~ U(-0.1,0.1)*pi rotation around z
      per-axis scale ~ rand*0.45 + 0.8  (approx [0.8,1.25])
    """
    def __init__(self):
        super().__init__()

    def forward(self, data):
        x = data.pos  # [P,3]

        theta = (torch.rand(1, device=x.device) * 0.2 - 0.1) * math.pi
        c, s = torch.cos(theta), torch.sin(theta)
        R = x.new_tensor([[c.item(), -s.item(), 0.0],
                          [s.item(),  c.item(), 0.0],
                          [0.0,      0.0,      1.0]])
        x = x @ R.T

        scale = torch.rand(3, device=x.device) * 0.45 + 0.8
        x = x * scale

        data.pos = x
        return data


# -------------------------
# Repo-faithful model: PermEqui1_max + tanh + max pool
# -------------------------
class PermEqui1Max(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):  # [B,P,C]
        xm = x.max(dim=1, keepdim=True).values
        return self.Gamma(x - xm)


class DTanh(nn.Module):
    def __init__(self, d_dim, x_dim=3):
        super().__init__()
        self.phi = nn.Sequential(
            PermEqui1Max(x_dim, d_dim), nn.Tanh(),
            PermEqui1Max(d_dim, d_dim), nn.Tanh(),
            PermEqui1Max(d_dim, d_dim), nn.Tanh(),
        )
        self.ro = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(d_dim, d_dim), nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(d_dim, 40),
        )

    def forward(self, x):  # [B,P,3]
        phi_out = self.phi(x)
        pooled = phi_out.max(dim=1).values
        return self.ro(pooled)


# -------------------------
# Lightning module (no val)
# -------------------------
class LitRepoStyle(L.LightningModule):
    def __init__(self, num_points: int, network_dim=256, lr=1e-3, weight_decay=1e-7, adam_eps=1e-3, max_epochs=1000):
        super().__init__()
        self.save_hyperparameters()
        self.model = DTanh(network_dim, x_dim=3)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_points = num_points

    def _dense(self, batch):
        # Because we enforce fixed points per item, PyG Batch will stack to [B*P,3]
        B = int(batch.y.size(0))
        expected = B * self.num_points
        if batch.pos.size(0) != expected:
            raise RuntimeError(
                f"Expected {expected} points in batch.pos but got {batch.pos.size(0)}. "
                "Check SamplePoints(10000) + FixedSubsample."
            )
        x = batch.pos.view(B, self.num_points, 3)
        y = batch.y.view(-1)
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self._dense(batch)
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_epoch=True,batch_size=x.size(0))
        self.log("train_acc", acc, prog_bar=True, on_epoch=True,batch_size=x.size(0))
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = self._dense(batch)
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True, on_epoch=True,batch_size=x.size(0))
        self.log("test_acc", acc, prog_bar=True, on_epoch=True,batch_size=x.size(0))

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, eps=self.hparams.adam_eps)
        milestones = list(range(400, self.hparams.max_epochs, 400))  # 400, 800 if max_epochs=1000
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, required=True, choices=["10", "40"])
    p.add_argument("--config", type=str, default=None, help="Path to JSON config (optional)")
    return p.parse_args()

def main():
    # ---- Repo-like knobs ----
    DOWN = 2          # 2 -> ~5000 points from 10000 (paper-ish); 10 -> ~1000 (repo default)
    BASE = 10000      # make this 10000 to mimic their H5 setup
    P = BASE // DOWN  # number of points used by the network
    NETWORK_DIM = 512 if DOWN == 2 else 256
    BATCH = 64
    EPOCHS = 1000
    SEED = 0
    args = parse_args()
    dataset_name = args.dataset_name
    
    L.seed_everything(SEED, workers=True)

    # pre_transform caches processed samples on disk (fast epochs)
    pre = T.SamplePoints(BASE)

    train_tf = Compose([
        FixedSubsample("train", downsample=DOWN, base_points=BASE, seed=SEED),
        RepoStandardize(),
        RepoAugment(),  # train only
    ])
    test_tf = Compose([
        FixedSubsample("test", downsample=DOWN, base_points=BASE, seed=SEED),
        RepoStandardize(),
    ])

    root = os.path.expanduser(f"./data/ModelNet{dataset_name}")
    train_ds = ModelNet(root=root, name=f"{dataset_name}", train=True, pre_transform=pre, transform=train_tf)
    test_ds  = ModelNet(root=root, name=f"{dataset_name}", train=False, pre_transform=pre, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=8,
                              pin_memory=torch.cuda.is_available(), persistent_workers=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=8,
                              pin_memory=torch.cuda.is_available(), persistent_workers=True, drop_last=False)

    lit = LitRepoStyle(num_points=P, network_dim=NETWORK_DIM, lr=1e-3, weight_decay=1e-7, adam_eps=1e-3, max_epochs=EPOCHS)

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        max_epochs=EPOCHS,
        gradient_clip_val=5.0,   # repo: clip_grad(..., 5)
        enable_checkpointing=False,
        log_every_n_steps=20,
    )
    trainer.fit(lit, train_loader)
    trainer.test(lit, dataloaders=test_loader, verbose=True)


if __name__ == "__main__":
    main()