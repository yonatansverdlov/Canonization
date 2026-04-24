from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from hilbertcurve.hilbertcurve import HilbertCurve


PermField = Literal["sort_perm", "hilbert_perm"]


@dataclass
class CloudDistanceConfig:
    P: int
    train_batch: int = 4096
    device: str | None = None


# -------------------------
# Locate caches
# -------------------------
def find_cache_paths(P: int, dataset_name: str, hilbert_m: int) -> Tuple[str, str]:
    tag = f"modelnet{dataset_name}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    base_dir = os.path.join(repo_root, "data_creation", "data")

    train_path = os.path.join(base_dir, f"{tag}_train_P{P}_hilbm{hilbert_m}_norm_with_perms.pt")
    test_path = os.path.join(base_dir, f"{tag}_test_P{P}_hilbm{hilbert_m}_norm_with_perms.pt")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train cache: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test cache: {test_path}")

    return train_path, test_path


# -------------------------
# Cache loader + dataset wrapper
# -------------------------
def load_cache_dict(path: str) -> Dict[int, Data]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict[int, Data] in {path}, got {type(obj)}")
    return obj


class CachedDataDictDataset(Dataset):
    def __init__(self, cache: Dict[int, Data]):
        self.cache = cache
        self.keys = sorted(cache.keys())
        if not self.keys:
            raise ValueError("Empty cache")
        d0 = cache[self.keys[0]]
        self.P = int(d0.x.shape[0])

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Data:
        return self.cache[self.keys[idx]]


# -------------------------
# Perm helpers
# -------------------------
@torch.no_grad()
def apply_perm_single(X: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return X.index_select(0, perm)


@torch.no_grad()
def apply_perm_batch(Y: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    B, P, C = Y.shape
    idx = perm.unsqueeze(-1).expand(B, P, C)
    return torch.gather(Y, dim=1, index=idx)


# -------------------------
# Distance
# -------------------------
@torch.no_grad()
def mean_pointwise_l2_batch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    diff = Y - X.unsqueeze(0)
    per_point = torch.linalg.norm(diff, dim=-1)
    return per_point.mean(dim=-1)


@torch.no_grad()
def mean_pointwise_l2_perm_batch(
    X: torch.Tensor,
    x_perm: torch.Tensor,
    Y: torch.Tensor,
    y_perm: torch.Tensor,
) -> torch.Tensor:
    Xp = apply_perm_single(X, x_perm)
    Yp = apply_perm_batch(Y, y_perm)
    return mean_pointwise_l2_batch(Xp, Yp)


# -------------------------
# Recompute permutations
# -------------------------
@torch.no_grad()
def lexsort_perm_xyz(X: torch.Tensor) -> torch.Tensor:
    perm = torch.arange(X.shape[0], device=X.device)
    perm = perm[torch.argsort(X[perm, 2], stable=True)]
    perm = perm[torch.argsort(X[perm, 1], stable=True)]
    perm = perm[torch.argsort(X[perm, 0], stable=True)]
    return perm


@torch.no_grad()
def affine_normalize_min_then_max(X: torch.Tensor) -> torch.Tensor:
    m = X.min()
    Z = X - m
    M = Z.max()
    if M <= 0:
        return torch.zeros_like(X)
    return Z / M


@torch.no_grad()
def hilbert_perm_from_points(X: torch.Tensor, hilbert_m: int) -> torch.Tensor:
    Xn = affine_normalize_min_then_max(X).cpu()
    side = (1 << hilbert_m) - 1

    Q = torch.clamp(torch.floor(Xn * side), min=0, max=side).to(torch.int64)
    hc = HilbertCurve(p=hilbert_m, n=3)

    distances = hc.distances_from_points(Q.tolist())
    distances = torch.tensor(distances, dtype=torch.int64)

    perm = torch.argsort(distances, stable=True)
    return perm


# -------------------------
# Expand train set with noisy copies
# -------------------------
@torch.no_grad()
def group_train_by_class_with_noisy_copies(
    train_ds: Dataset,
    mu: float,
    num_copies: int,
    hilbert_m: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    For each train sample s, create exactly num_copies noisy copies:
        s_noisy = s + mu * N(0,1)

    The original clean sample is NOT kept.
    For each noisy copy, recompute both sort_perm and hilbert_perm.
    """
    buckets: Dict[int, Dict[str, list]] = {}

    for d in train_ds:
        if not hasattr(d, "x") or not hasattr(d, "y"):
            raise ValueError("Train Data must have fields 'x' and 'y'")

        cls = int(d.y.item())
        b = buckets.setdefault(cls, {"x": [], "sort_perm": [], "hilbert_perm": []})

        x0 = d.x.cpu().contiguous()

        for _ in range(num_copies):
            noise = mu * torch.randn_like(x0)
            x_noisy = (x0 + noise).contiguous()

            sort_perm = lexsort_perm_xyz(x_noisy)
            hilbert_perm = hilbert_perm_from_points(x_noisy, hilbert_m)

            b["x"].append(x_noisy)
            b["sort_perm"].append(sort_perm.cpu().contiguous())
            b["hilbert_perm"].append(hilbert_perm.cpu().contiguous())

    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for cls, b in buckets.items():
        out[cls] = {k: torch.stack(v, dim=0).contiguous() for k, v in b.items()}
    return out


# -------------------------
# Score computation
# -------------------------
@torch.no_grad()
def score_perm_metric(
    train_by_class: Dict[int, Dict[str, torch.Tensor]],
    test_ds: Dataset,
    cfg: CloudDistanceConfig,
    perm_field: PermField,
) -> float:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vals: List[float] = []

    for t_idx, td in enumerate(test_ds):
        cls = int(td.y.item())
        if cls not in train_by_class:
            raise RuntimeError(f"Class {cls} appears in test but not in train.")

        if not hasattr(td, perm_field):
            raise RuntimeError(f"Test sample missing '{perm_field}'")

        pack = train_by_class[cls]

        X = td.x.to(device=device, dtype=torch.float32).contiguous()
        x_perm = getattr(td, perm_field).to(device=device, non_blocking=True)

        Y_cpu = pack["x"]
        Yperm_cpu = pack[perm_field]
        N = Y_cpu.size(0)

        best = None
        for s in range(0, N, cfg.train_batch):
            Y = Y_cpu[s:s + cfg.train_batch].to(device, non_blocking=True)
            y_perm = Yperm_cpu[s:s + cfg.train_batch].to(device, non_blocking=True)

            d_batch = mean_pointwise_l2_perm_batch(X, x_perm, Y, y_perm)
            bmin = d_batch.min()
            best = bmin if best is None else torch.minimum(best, bmin)

        vals.append(float(best.item()))

        if (t_idx + 1) % 50 == 0:
            print(f"[{perm_field}] processed {t_idx + 1}/{len(test_ds)}")

    return float(sum(vals) / len(vals))


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute noisy-train NN scores per mu for l2_sorted and l2_hilbert"
    )
    p.add_argument("--P", type=int, required=True)
    p.add_argument("--dataset_name", type=str, required=True, choices=["10", "40"])
    p.add_argument("--hilbert_m", type=int, default=12)

    p.add_argument(
        "--mus",
        type=float,
        nargs="+",
        required=True,
        help="Compute results separately for each mu in this list.",
    )

    p.add_argument(
        "--num_copies",
        type=int,
        default=5,
        help="Number of noisy copies per original train sample. Original is not kept.",
    )

    p.add_argument("--train_batch", type=int, default=4096)
    return p.parse_args()


def main():
    args = parse_args()

    train_path, test_path = find_cache_paths(args.P, args.dataset_name, hilbert_m=args.hilbert_m)
    print("Using train cache:", train_path)
    print("Using test  cache:", test_path)

    train_cache = load_cache_dict(train_path)
    test_cache = load_cache_dict(test_path)

    train_ds = CachedDataDictDataset(train_cache)
    test_ds = CachedDataDictDataset(test_cache)

    if train_ds.P != args.P or test_ds.P != args.P:
        raise RuntimeError(
            f"P mismatch: train P={train_ds.P}, test P={test_ds.P}, expected P={args.P}"
        )

    cfg = CloudDistanceConfig(P=args.P, train_batch=args.train_batch, device=None)

    original_train_size = len(train_ds)
    expanded_train_size = original_train_size * args.num_copies

    print(f"\nOriginal train size: {original_train_size}")
    print(f"Noisy train size per mu: {expanded_train_size} (= {original_train_size} x {args.num_copies})")

    all_results = {}

    for mu in args.mus:
        print(f"\n----- mu = {mu} -----")
        print("Building noisy train copies...")

        train_by_class = group_train_by_class_with_noisy_copies(
            train_ds=train_ds,
            mu=mu,
            num_copies=args.num_copies,
            hilbert_m=args.hilbert_m,
        )

        print("Computing l2_sorted...")
        l2_sorted = score_perm_metric(train_by_class, test_ds, cfg, "sort_perm")

        print("Computing l2_hilbert...")
        l2_hilbert = score_perm_metric(train_by_class, test_ds, cfg, "hilbert_perm")

        all_results[mu] = {
            "l2_sorted": l2_sorted,
            "l2_hilbert": l2_hilbert,
        }

    print("\n================ FINAL DISTANCES FOR EACH MU ================\n")
    for mu in args.mus:
        print(f"mu = {mu}")
        print(f"  l2_sorted  : {all_results[mu]['l2_sorted']}")
        print(f"  l2_hilbert : {all_results[mu]['l2_hilbert']}")
        print()

    print("============================================================\n")


if __name__ == "__main__":
    main()