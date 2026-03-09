# build_modelnet10_perms_cache.py
#
# Creates cached PyG Data objects for ModelNet10 train/test.
# For each sample we save:
#   data.x            : (P,3) normalized point cloud (float32)
#   data.sort_perm    : (P,)  lexicographic permutation by (x,y,z) on normalized x
#   data.id_perm      : (P,)  identity permutation
#   data.hilbert_perm : (P,)  permutation by Hilbert index (m-bit quantization) on normalized x
#   data.y            : (1,)  class label (long)

import os
from typing import Dict

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints

from hilbertcurve.hilbertcurve import HilbertCurve


# -------------------------
# Your normalization: subtract global min, divide by shifted global max
# -------------------------
@torch.no_grad()
def affine_normalize_min_then_max(X: torch.Tensor) -> torch.Tensor:
    """
    X: (P,3)
    m = min over all entries
    Z = X - m
    M = max over all entries in Z
    X' = Z / M
    """
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"Expected (P,3), got {tuple(X.shape)}")

    m = X.amin()          # scalar
    Z = X - m
    M = Z.amax()          # scalar
    return Z / M


# -------------------------
# Lexicographic permutation (x, then y, then z)
# -------------------------
@torch.no_grad()
def lex_perm_xyz(X: torch.Tensor) -> torch.Tensor:
    """
    X: (P,3)
    returns perm: (P,) such that X[perm] is lex-sorted by (x,y,z).
    """
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"Expected (P,3), got {tuple(X.shape)}")

    P = X.shape[0]
    idx = torch.arange(P, device=X.device)

    for k in (2, 1, 0):  # stable sort by z, then y, then x
        idx = idx[torch.argsort(X[idx, k], stable=True)]

    return idx.to(torch.long)


# -------------------------
# Hilbert permutation (m bits per coordinate)
# -------------------------
_HC_CACHE: Dict[int, HilbertCurve] = {}

def _get_hc(m: int) -> HilbertCurve:
    hc = _HC_CACHE.get(m)
    if hc is None:
        hc = HilbertCurve(m, 3)
        _HC_CACHE[m] = hc
    return hc


@torch.no_grad()
def hilbert_perm_3d(X: torch.Tensor, m: int) -> torch.Tensor:
    """
    X: (P,3) normalized (typically in [0,1])
    m: bits per coordinate

    returns perm: (P,) such that X[perm] is ordered by Hilbert index of quantized coords.
    """
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"Expected (P,3), got {tuple(X.shape)}")
    if m <= 0:
        raise ValueError("m must be positive")

    hc = _get_hc(m)
    scale = (1 << m) - 1

    Q = torch.floor(X * float(scale)).to(torch.int64)  # (P,3)
    h_list = hc.distances_from_points(Q.cpu().tolist())

    perm = torch.tensor(
        sorted(range(len(h_list)), key=h_list.__getitem__),
        dtype=torch.long,
        device=X.device,
    )
    return perm


# -------------------------
# Build caches
# -------------------------
@torch.no_grad()
def build_split_cache(ds: ModelNet, P: int, hilbert_m: int) -> Dict[int, Data]:
    out: Dict[int, Data] = {}

    for i, d in enumerate(ds):
        x = d.pos.to(dtype=torch.float32).contiguous()  # (P,3)
        if x.shape != (P, 3):
            raise RuntimeError(f"Expected x shape {(P,3)}, got {tuple(x.shape)}")

        x = affine_normalize_min_then_max(x)

        sort_perm = lex_perm_xyz(x)
        hilb_perm = hilbert_perm_3d(x, m=hilbert_m)
        id_perm = torch.arange(P, dtype=torch.long)

        data = Data(
            x=x.cpu(),
            sort_perm=sort_perm.cpu(),
            id_perm=id_perm.cpu(),
            hilbert_perm=hilb_perm.cpu(),
            y=d.y.to(torch.long).view(1).cpu(),
        )
        data.sample_id = torch.tensor([i], dtype=torch.long)  # optional
        out[i] = data

        if (i + 1) % 200 == 0:
            print(f"processed {i+1}/{len(ds)}")

    return out


import os
import argparse
import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints

# assumes you already have: build_split_cache(train_ds, P, hilbert_m)

def parse_args():
    parser = argparse.ArgumentParser(description="Build ModelNet cache with Hilbert perms")
    parser.add_argument("--P", type=int, default=1024,
                        help="Number of points to sample per shape")
    parser.add_argument("--dataset_name", type=str, default="10", choices=["10", "40"],
                        help="ModelNet subset: 10 or 40")
    parser.add_argument("--hilbert_m", type=int, default=12,
                        help="Hilbert discretization parameter (recommend ~10–20)")
    parser.add_argument("--force_reload", action="store_true",default=True,
                        help="Force re-processing of the dataset")
    parser.add_argument("--datasets_root", type=str, default="data/datasets",
                        help="Base datasets folder (will use ModelNet10/ModelNet40 under it)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Root folder convention: ~/datasets/ModelNet10 or ~/datasets/ModelNet40
    root = os.path.expanduser(os.path.join(args.datasets_root, f"ModelNet{args.dataset_name}"))

    P = args.P
    hilbert_m = args.hilbert_m
    force_reload = args.force_reload

    pre_transform = SamplePoints(P)
    train_ds = ModelNet(root=root, name=args.dataset_name, train=True,
                        pre_transform=pre_transform, force_reload=force_reload)
    test_ds  = ModelNet(root=root, name=args.dataset_name, train=False,
                        pre_transform=pre_transform, force_reload=force_reload)

    print("Building train cache...")
    train_cache = build_split_cache(train_ds, P=P, hilbert_m=hilbert_m)

    print("Building test cache...")
    test_cache = build_split_cache(test_ds, P=P, hilbert_m=hilbert_m)

    tag = f"modelnet{args.dataset_name}"
    train_path = f"{tag}_train_P{P}_hilbm{hilbert_m}_norm_with_perms.pt"
    test_path  = f"{tag}_test_P{P}_hilbm{hilbert_m}_norm_with_perms.pt"

    torch.save(train_cache, train_path)
    torch.save(test_cache, test_path)

    print("Saved:")
    print(" ", train_path)
    print(" ", test_path)


if __name__ == "__main__":
    main()