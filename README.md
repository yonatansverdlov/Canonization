cat > README.md << 'EOF'
# Canonization

Code for building cached point-cloud datasets with canonical permutations for ModelNet.

This repository currently includes a script that preprocesses ModelNet10 or ModelNet40 point clouds, applies normalization, computes several point permutations, and saves the results as cached PyTorch Geometric Data objects.

## What the script does

The script build_modelnet10_perms_cache.py

1. Loads ModelNet10 or ModelNet40 using PyTorch Geometric
2. Samples a fixed number of points from each mesh
3. Applies affine normalization
4. Computes three permutations for each point cloud
   - identity permutation
   - lexicographic permutation by x y z
   - Hilbert-curve permutation after quantization
5. Saves the processed train and test splits as pt cache files

## Saved fields

For each sample, the cache stores

- data.x  
  normalized point cloud of shape P by 3 with float32

- data.sort_perm  
  permutation of shape P corresponding to lexicographic sorting by x y z

- data.id_perm  
  identity permutation of shape P

- data.hilbert_perm  
  permutation of shape P corresponding to Hilbert ordering

- data.y  
  class label

Optionally, each sample also stores

- data.sample_id

## Main script

build_modelnet10_perms_cache.py

## Installation

Install the required packages with

pip install -r requirements.txt

Depending on your machine and CUDA setup, you may want to install PyTorch and PyTorch Geometric using their official installation instructions.

## Requirements

Main dependencies

- Python
- PyTorch
- PyTorch Geometric
- hilbertcurve

## Example usage

Build a cache for ModelNet10 with 1024 sampled points and Hilbert parameter 12

python build_modelnet10_perms_cache.py --P 1024 --dataset_name 10 --hilbert_m 12

Build a cache for ModelNet40

python build_modelnet10_perms_cache.py --P 1024 --dataset_name 40 --hilbert_m 12

Specify a custom datasets root

python build_modelnet10_perms_cache.py --datasets_root data/datasets --dataset_name 10

## Arguments

- --P  
  Number of sampled points per shape

- --dataset_name  
  Which ModelNet subset to use 10 or 40

- --hilbert_m  
  Number of bits per coordinate for Hilbert quantization

- --force_reload  
  Force reprocessing of the dataset

- --datasets_root  
  Base directory where ModelNet data is stored

## Output files

The script saves files of the form

- modelnet10_train_P{P}_hilbm{hilbert_m}_norm_with_perms.pt
- modelnet10_test_P{P}_hilbm{hilbert_m}_norm_with_perms.pt

or for ModelNet40

- modelnet40_train_P{P}_hilbm{hilbert_m}_norm_with_perms.pt
- modelnet40_test_P{P}_hilbm{hilbert_m}_norm_with_perms.pt

## Notes

- The normalization subtracts the global minimum over all coordinates and divides by the shifted global maximum
- Hilbert ordering is computed after discretizing the normalized coordinates
- The cache is stored as a Python dictionary mapping sample index to a PyG Data object

## Current status

This repository is under active development.
More scripts training code and experiment documentation will be added.
EOF