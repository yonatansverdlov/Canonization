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
  - [Citation](#citation)
  - [Contact](#contact)

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
### Table 2: ModelNet classification

Two commands run all three models (Hilbert, Lex-Sort, and MLP without sorting)
on ModelNet40 and ModelNet10. Each model runs with seeds 0–4:

```bash
python scripts/run_modelnet_classification.py --dataset 40
python scripts/run_modelnet_classification.py --dataset 10
```

The MLP baseline uses `--ordering ply` internally (the original, unsorted
point order); the other two use `hilbert` and `lex`, respectively. Each
ordering retains its own hyperparameters from
`ModelNet/training/configs/modelnet.json`.

The combined results are saved as
`results/modelnet/table2_modelnet40.{json,csv}` and
`results/modelnet/table2_modelnet10.{json,csv}`, with per-model summaries
under `ModelNet/training/checkpoints/modelnet{40,10}_{hilbert,lex,ply}/`
and per-seed checkpoints under matching `_seed0` through `_seed4` folders.

To run only one model, the original `--ordering hilbert|lex|ply` option
remains available.

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

```bash
python scripts/setup_dws_data.py
```

### Training

MNIST-INR:

```bash
python scripts/run_dws.py --dataset mnist --model mlp
python scripts/run_dws.py --dataset mnist --model can_mlp
python scripts/run_dws.py --dataset mnist --model dwsnet
```

Fashion-MNIST-INR:

```bash
python scripts/run_dws.py --dataset fmnist --model mlp
python scripts/run_dws.py --dataset fmnist --model can_mlp
python scripts/run_dws.py --dataset fmnist --model dwsnet
```

## Reproducibility

Seeded experiments use deterministic Python, NumPy, PyTorch, CUDA, and DataLoader settings where supported. Rotated MNIST `learned_can` is not guaranteed to be bitwise deterministic on CUDA because its Kornia rotation uses CUDA `grid_sample` backward.

## Citation

If you find this code useful, please cite:

```bibtex
@article{sverdlov2026canonize,
  title={When and How to Canonize: A Generalization Perspective},
  author={Sverdlov, Yonatan and Friedman, Benjamin and Hordan, Snir and Dym, Nadav},
  journal={arXiv preprint arXiv:2605.11008},
  year={2026}
}
```

## Contact

For questions, feedback, or collaboration opportunities, feel free to reach out:

📧 **Email:** [yonatans@campus.technion.ac.il](mailto:yonatans@campus.technion.ac.il)

If you encounter issues or have suggestions, please open an issue on the [GitHub repository](https://github.com/yonatansverdlov/Canonization).
