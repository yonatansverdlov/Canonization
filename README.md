# Canonization

<p align="center">
  <img src="hilbert.png" width="700">
</p>

This repository contains the experiments for studying canonization and invariant learning across point clouds, image rotations, and neural-network weight spaces.

## Contents

- [Installation](#installation)
- [ModelNet](#modelnet)
  - [Covering Number Experiment](#covering-number-experiment)
  - [Point-Cloud Classification](#point-cloud-classification)
  - [PCA Sign Canonization](#pca-sign-canonization)
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
