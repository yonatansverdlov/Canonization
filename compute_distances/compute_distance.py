# nn_scores_all_metrics.py
#
# Computes class-restricted NN scores for cached ModelNet perms:
#   score = mean_{X in test} min_{Y in train, same class} d(X,Y)
#
# Metrics:
#   l2         : mean_i ||X[i]-Y[i]||_2
#   l2_sorted  : mean_i ||X[sort_perm_X][i]-Y[sort_perm_Y][i]||_2   (cached perms)
#   l2_hilbert : mean_i ||X[hilbert_perm_X][i]-Y[hilbert_perm_Y][i]||_2 (cached perms)
#   l2_wass    : min_{pi} mean_i ||X[i]-Y[pi(i)]||_2  (Hungarian / assignment)

from __future__ import annotations

import os
import glob
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from scipy.optimize import linear_sum_assignment


PermField = Literal["id_perm", "sort_perm", "hilbert_perm"]


@dataclass
class CloudDistanceConfig:
    P: int
    train_batch: int = 4096
    device: str | None = None


# -------------------------
# Locate caches (only P + dataset_name)
# -------------------------
def find_cache_paths(P: int, dataset_name: str) -> Tuple[str, str]:
    tag = f"modelnet{dataset_name}"
    train_pat = f"{tag}_train_P{P}_hilbm*_norm_with_perms.pt"
    test_pat  = f"{tag}_test_P{P}_hilbm*_norm_with_perms.pt"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    candidates = [
        os.getcwd(),
        os.path.join(os.getcwd(), "data_creation"),
        script_dir,
        os.path.join(script_dir, "data_creation"),
        repo_root,
        os.path.join(repo_root, "data_creation"),
    ]

    def _glob_all(pattern: str) -> List[str]:
        hits: List[str] = []
        for d in candidates:
            hits.extend(glob.glob(os.path.join(d, pattern)))
        return hits

    train_hits = _glob_all(train_pat)
    test_hits = _glob_all(test_pat)

    if not train_hits:
        raise FileNotFoundError(f"Missing train cache pattern '{train_pat}' in:\n" + "\n".join("  "+d for d in candidates))
    if not test_hits:
        raise FileNotFoundError(f"Missing test cache pattern '{test_pat}' in:\n" + "\n".join("  "+d for d in candidates))

    # If multiple matches exist, just take the first (you can delete old ones to avoid ambiguity)
    return train_hits[0], test_hits[0]


# -------------------------
# Cache loader + dataset wrapper
# -------------------------
def load_cache_dict(path: str) -> Dict[int, Data]:
    obj = torch.load(path, map_location="cpu",weights_only=False)
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
    # Y: (B,P,3), perm: (B,P)
    B, P, C = Y.shape
    idx = perm.unsqueeze(-1).expand(B, P, C)
    return torch.gather(Y, dim=1, index=idx)


# -------------------------
# Distances (match your old reductions)
# -------------------------
@torch.no_grad()
def mean_pointwise_l2_batch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Matches old l2_distance:
      (1/P) * sum_i ||X[i] - Y[b,i]||_2
    X: (P,3)
    Y: (B,P,3)
    returns (B,)
    """
    diff = Y - X.unsqueeze(0)                # (B,P,3)
    per_point = torch.linalg.norm(diff, dim=-1)  # (B,P)
    return per_point.mean(dim=-1)

@torch.no_grad()
def mean_pointwise_l2_perm_batch(
    X: torch.Tensor, x_perm: torch.Tensor,
    Y: torch.Tensor, y_perm: torch.Tensor
) -> torch.Tensor:
    """
    mean_i || X[x_perm][i] - Y[y_perm][i] ||_2
    X: (P,3), x_perm: (P,)
    Y: (B,P,3), y_perm: (B,P)
    """
    Xp = apply_perm_single(X, x_perm)    # (P,3)
    Yp = apply_perm_batch(Y, y_perm)     # (B,P,3)
    return mean_pointwise_l2_batch(Xp, Yp)

@torch.no_grad()
def wass_hungarian_batch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Matches your old l2_wass_distance:
      out[b] = min_pi mean_i ||X[i] - Y[b, pi(i)]||_2
    using SciPy Hungarian on CPU.

    X: (P,3) on device
    Y: (B,P,3) on device
    returns (B,) on same device as X
    """
    # cost matrices: C[b,i,j] = ||X[i] - Y[b,j]||_2
    C = torch.cdist(X.unsqueeze(0), Y, p=2)  # (B,P,P)

    B = C.shape[0]
    out = torch.empty(B, dtype=C.dtype, device=C.device)

    C_cpu = C.detach().cpu().numpy()
    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(C_cpu[b])
        out[b] = C[b, row_ind, col_ind].mean()  # mean, not sum
    return out


# -------------------------
# Group train by class once (CPU)
# -------------------------
@torch.no_grad()
def group_train_by_class(train_ds: Dataset) -> Dict[int, Dict[str, torch.Tensor]]:
    buckets: Dict[int, Dict[str, list]] = {}

    for d in train_ds:
        for field in ("x", "y", "id_perm", "sort_perm", "hilbert_perm"):
            if not hasattr(d, field):
                raise ValueError(f"Train Data missing '{field}'")

        cls = int(d.y.item())
        b = buckets.setdefault(cls, {"x": [], "id_perm": [], "sort_perm": [], "hilbert_perm": []})
        b["x"].append(d.x.cpu().contiguous())
        b["id_perm"].append(d.id_perm.cpu().contiguous())
        b["sort_perm"].append(d.sort_perm.cpu().contiguous())
        b["hilbert_perm"].append(d.hilbert_perm.cpu().contiguous())

    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for cls, b in buckets.items():
        out[cls] = {k: torch.stack(v, dim=0).contiguous() for k, v in b.items()}
    return out


# -------------------------
# Score computations (class-restricted NN)
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
        pack = train_by_class[cls]

        X = td.x.to(device=device, dtype=torch.float32).contiguous()
        x_perm = getattr(td, perm_field).to(device=device, non_blocking=True)

        Y_cpu = pack["x"]
        Yperm_cpu = pack[perm_field]
        N = Y_cpu.size(0)

        best = None
        for s in range(0, N, cfg.train_batch):
            Y = Y_cpu[s:s+cfg.train_batch].to(device, non_blocking=True)
            y_perm = Yperm_cpu[s:s+cfg.train_batch].to(device, non_blocking=True)

            d_batch = mean_pointwise_l2_perm_batch(X, x_perm, Y, y_perm)
            bmin = d_batch.min()
            best = bmin if best is None else torch.minimum(best, bmin)

        vals.append(float(best.item()))
        if (t_idx + 1) % 50 == 0:
            print(f"[{perm_field}] processed {t_idx+1}/{len(test_ds)}")

    return float(sum(vals) / len(vals))


@torch.no_grad()
def score_l2_identity(
    train_by_class: Dict[int, Dict[str, torch.Tensor]],
    test_ds: Dataset,
    cfg: CloudDistanceConfig,
) -> float:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vals: List[float] = []

    for t_idx, td in enumerate(test_ds):
        cls = int(td.y.item())
        pack = train_by_class[cls]

        X = td.x.to(device=device, dtype=torch.float32).contiguous()
        Y_cpu = pack["x"]
        N = Y_cpu.size(0)

        best = None
        for s in range(0, N, cfg.train_batch):
            Y = Y_cpu[s:s+cfg.train_batch].to(device, non_blocking=True)
            d_batch = mean_pointwise_l2_batch(X, Y)
            bmin = d_batch.min()
            best = bmin if best is None else torch.minimum(best, bmin)

        vals.append(float(best.item()))
        if (t_idx + 1) % 50 == 0:
            print(f"[l2] processed {t_idx+1}/{len(test_ds)}")

    return float(sum(vals) / len(vals))


@torch.no_grad()
def score_wasserstein(
    train_by_class: Dict[int, Dict[str, torch.Tensor]],
    test_ds: Dataset,
    cfg: CloudDistanceConfig,
    wass_batch: int = 32,  # smaller chunk, Hungarian is expensive
) -> float:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vals: List[float] = []

    for t_idx, td in enumerate(test_ds):
        cls = int(td.y.item())
        pack = train_by_class[cls]

        X = td.x.to(device=device, dtype=torch.float32).contiguous()
        Y_cpu = pack["x"]
        N = Y_cpu.size(0)

        best = None
        for s in range(0, N, wass_batch):
            Y = Y_cpu[s:s+wass_batch].to(device, non_blocking=True)
            d_batch = wass_hungarian_batch(X, Y)  # (B,)
            bmin = d_batch.min()
            best = bmin if best is None else torch.minimum(best, bmin)

        vals.append(float(best.item()))
        if (t_idx + 1) % 10 == 0:
            print(f"[l2_wass] processed {t_idx+1}/{len(test_ds)}")

    return float(sum(vals) / len(vals))


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compute NN scores for all metrics using cached Data perms")
    p.add_argument("--P", type=int, required=True)
    p.add_argument("--dataset_name", type=str, required=True, choices=["10", "40"])
    return p.parse_args()


def main():
    args = parse_args()

    train_path, test_path = find_cache_paths(args.P, args.dataset_name)
    print("Using train cache:", train_path)
    print("Using test  cache:", test_path)

    # Old code hardcoded m=10 in l2_sorted_hilbert_distance.
    # If your cache isn't hilbm10, your "hilbert" score will differ vs that old run.
    if "hilbm10" not in os.path.basename(train_path):
        print("WARNING: your old l2_sorted_hilbert_distance used m=10.")
        print("         Your cache filename suggests a different hilbert_m, so hilbert results may differ.")

    train_cache = load_cache_dict(train_path)
    test_cache = load_cache_dict(test_path)
    train_ds = CachedDataDictDataset(train_cache)
    test_ds = CachedDataDictDataset(test_cache)

    if train_ds.P != args.P or test_ds.P != args.P:
        raise RuntimeError(f"P mismatch: train P={train_ds.P}, test P={test_ds.P}, expected P={args.P}")

    cfg = CloudDistanceConfig(P=args.P, train_batch=4096, device=None)

    print("\nGrouping train by class (one-time CPU)...")
    train_by_class = group_train_by_class(train_ds)

    results = {}

    print("\nComputing l2 (identity order)...")
    results["l2"] = score_l2_identity(train_by_class, test_ds, cfg)

    print("\nComputing l2_sorted (cached sort_perm)...")
    results["l2_sorted"] = score_perm_metric(train_by_class, test_ds, cfg, "sort_perm")

    print("\nComputing l2_hilbert (cached hilbert_perm)...")
    results["l2_hilbert"] = score_perm_metric(train_by_class, test_ds, cfg, "hilbert_perm")

    print("\nComputing l2_wass (Hungarian assignment)...")
    results["l2_wass"] = score_wasserstein(train_by_class, test_ds, cfg, wass_batch=32)

    print("\n================ SUMMARY ================")
    for k in ["l2", "l2_sorted", "l2_hilbert", "l2_wass"]:
        print(f"{k:10s} : {results[k]}")
    print("========================================\n")


if __name__ == "__main__":
    main()