# Canonization

<p align="center">
  <img src="hilbert.png" width="700">
</p>

This repository contains the experiments for studying canonization and invariant learning across point clouds, image rotations, and neural-network weight spaces.

## Contents

- [Canonization](#canonization)
  - [Contents](#contents)
  - [Installation](#installation)
  - [ModelNet](#modelnet)
    - [Data Setup](#data-setup)
    - [Covering Number Experiment](#covering-number-experiment)
  - [Rotated MNIST](#rotated-mnist)
  - [Deep Weight Spaces](#deep-weight-spaces)

## Installation

Clone the repository:

```bash
git clone https://github.com/yonatansverdlov/Canonization.git
cd Canonization
```

Create and activate the conda environment:

```bash
conda create -n canon python=3.10 -y
conda activate canon
```

Install all required dependencies:

```bash
pip install -r requirements.txt
```
## ModelNet

### Data Setup

Download and prepare the required ModelNet datasets by running:

```bash
python scripts/setup_modelnet.py
```
### Covering Number Experiment

Run the experiment on ModelNet10:

```bash
python scripts/run_modelnet_covering.py --dataset 10
```

Run the experiment on ModelNet40:

```bash
python scripts/run_modelnet_covering.py --dataset 40
```

## Rotated MNIST

### Training

```bash
python scripts/run_rotated_mnist.py --model cnn
python scripts/run_rotated_mnist.py --model average
python scripts/run_rotated_mnist.py --model learned_can
python scripts/run_rotated_mnist.py --model frozen_can
```

Each experiment is run over 5 random seeds.

### Distance Computation

```bash
python scripts/run_rotated_mnist_distances.py
```

The experiment reports `l2`, `group`, and `can_frozen`; `can_learned` is also reported when a learned canonization checkpoint is available. The distance computation uses seed `0` by default.

## Deep Weight Spaces

### Data Setup

Download the original DWS MNIST-INR and Fashion-MNIST-INR datasets and convert them to PyG `Data` objects:

```bash
python scripts/setup_dws_data.py
```

Each processed sample stores the same INR twice:

- `raw`: the original PyTorch state dict.
- `canon`: the canonized state dict.

The canonization proceeds sequentially from input to output. For each hidden layer, neurons are sorted lexicographically using the key `[incoming weights, bias, sorted outgoing weights]`. If this gives permutation (P_l), the rows of (W_l) and (b_l) are permuted and the same permutation is propagated to the columns of (W_{l+1}). The output layer is not sorted.

The processed split is fixed to 55,000 train, 5,000 validation, and 10,000 test examples using split seed `0` by default.

## Reproducibility

Seeded experiments use deterministic Python, NumPy, PyTorch, CUDA, and DataLoader settings where supported. Rotated MNIST `learned_can` is not guaranteed to be bitwise deterministic on CUDA because its Kornia rotation uses CUDA `grid_sample` backward.

