import os
import glob
import json
import shutil
import argparse
from typing import Dict

import h5py
import numpy as np
import torch
from hilbertcurve.hilbertcurve import HilbertCurve


# -------------------------
# Normalization
# -------------------------
@torch.no_grad()
def affine_normalize_min_then_max(X: torch.Tensor) -> torch.Tensor:
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"Expected (P,3), got {tuple(X.shape)}")

    m = X.amin()
    Z = X - m
    M = Z.amax()

    # Clamp denominator to prevent division-by-zero
    return Z / torch.clamp(M, min=1e-8)


# -------------------------
# Permutation Generators
# -------------------------
@torch.no_grad()
def lex_perm_xyz(X: torch.Tensor) -> torch.Tensor:
    P = X.shape[0]
    idx = torch.arange(P, device=X.device)

    # Stable sort by z, then y, then x
    for k in (2, 1, 0):
        idx = idx[torch.argsort(X[idx, k], stable=True)]

    return idx.to(torch.long)


_HC_CACHE: Dict[int, HilbertCurve] = {}


def _get_hc(m: int) -> HilbertCurve:
    hc = _HC_CACHE.get(m)
    if hc is None:
        hc = HilbertCurve(m, 3)
        _HC_CACHE[m] = hc
    return hc


@torch.no_grad()
def hilbert_perm_3d(X: torch.Tensor, m: int) -> torch.Tensor:
    hc = _get_hc(m)
    scale = (1 << m) - 1

    Q = torch.floor(X * float(scale)).to(torch.int64)
    h_list = hc.distances_from_points(Q.cpu().tolist())

    perm = torch.tensor(
        sorted(range(len(h_list)), key=h_list.__getitem__),
        dtype=torch.long,
        device=X.device,
    )
    return perm


# -------------------------
# HDF5 Processing
# -------------------------
def process_h5_file(input_path: str, lex_output_path: str, hilbert_output_path: str, hilbert_m: int,
                    device: torch.device):
    """Reads an h5 file, canonicalizes the data, and saves to two new h5 files."""
    with h5py.File(input_path, 'r') as f:
        data = f['data'][:]  # Shape: (N, 2048, 3)
        label = f['label'][:]  # Shape: (N, 1)

        # Check if normals exist in the dataset
        has_normal = 'normal' in f
        if has_normal:
            normal = f['normal'][:]

    N, P, _ = data.shape

    # Prepare output arrays
    lex_data = np.zeros_like(data)
    hilbert_data = np.zeros_like(data)

    if has_normal:
        lex_normal = np.zeros_like(normal)
        hilbert_normal = np.zeros_like(normal)

    # Process each point cloud
    for i in range(N):
        x = torch.tensor(data[i], dtype=torch.float32, device=device)

        # 1. Normalize
        x_norm = affine_normalize_min_then_max(x)

        # 2. Get Permutations (using the normalized coordinates)
        lex_idx = lex_perm_xyz(x_norm)
        hilbert_idx = hilbert_perm_3d(x_norm, hilbert_m)

        # 3. Apply permutations and convert back to numpy
        # Note: We are saving the *normalized* coordinates here, mirroring your original script.
        lex_data[i] = x_norm[lex_idx].cpu().numpy()
        hilbert_data[i] = x_norm[hilbert_idx].cpu().numpy()

        # If normals exist, they must be physically reordered by the exact same permutations
        if has_normal:
            n_tensor = torch.tensor(normal[i], dtype=torch.float32, device=device)
            lex_normal[i] = n_tensor[lex_idx].cpu().numpy()
            hilbert_normal[i] = n_tensor[hilbert_idx].cpu().numpy()

    # Save Lexicographical version
    with h5py.File(lex_output_path, 'w') as f_lex:
        f_lex.create_dataset('data', data=lex_data)
        f_lex.create_dataset('label', data=label)
        if has_normal:
            f_lex.create_dataset('normal', data=lex_normal)

    # Save Hilbert version
    with h5py.File(hilbert_output_path, 'w') as f_hilb:
        f_hilb.create_dataset('data', data=hilbert_data)
        f_hilb.create_dataset('label', data=label)
        if has_normal:
            f_hilb.create_dataset('normal', data=hilbert_normal)


# -------------------------
# Main Execution
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Canonicalize ModelNet40 HDF5 dataset")
    parser.add_argument("--data_dir", type=str, default="data/modelnet40_ply_hdf5_2048",
                        help="Path to the original hdf5 directory")
    parser.add_argument("--hilbert_m", type=int, default=12,
                        help="Hilbert discretization parameter")
    args = parser.parse_args()

    # Setup directories
    base_dir = os.path.dirname(args.data_dir.rstrip('/'))
    if not base_dir:
        base_dir = "data"

    lex_dir = os.path.join(base_dir, "modelnet40_lex_hdf5_2048")
    hilbert_dir = os.path.join(base_dir, "modelnet40_hilbert_hdf5_2048")

    os.makedirs(lex_dir, exist_ok=True)
    os.makedirs(hilbert_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Process all .h5 files
    h5_files = glob.glob(os.path.join(args.data_dir, "*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {args.data_dir}. Check your path!")
        return

    for idx, h5_path in enumerate(h5_files):
        filename = os.path.basename(h5_path)
        lex_out = os.path.join(lex_dir, filename)
        hilbert_out = os.path.join(hilbert_dir, filename)

        print(f"[{idx + 1}/{len(h5_files)}] Processing {filename}...")
        process_h5_file(h5_path, lex_out, hilbert_out, args.hilbert_m, device)

    # 2. Copy metadata files (.txt, .json) so dataloaders still work seamlessly
    meta_files = glob.glob(os.path.join(args.data_dir, "*.txt")) + \
                 glob.glob(os.path.join(args.data_dir, "*.json"))

    print("Copying metadata files...")
    for meta_file in meta_files:
        filename = os.path.basename(meta_file)
        shutil.copy(meta_file, os.path.join(lex_dir, filename))
        shutil.copy(meta_file, os.path.join(hilbert_dir, filename))

    print("\n✅ Done! Datasets created at:")
    print(f"  Lexicographical: {lex_dir}")
    print(f"  Hilbert:         {hilbert_dir}")


if __name__ == "__main__":
    main()