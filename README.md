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
```

Each experiment is run over 5 random seeds.

### Distance Computation

```bash
python scripts/run_rotated_mnist_distances.py
```
