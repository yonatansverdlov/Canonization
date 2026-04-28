# Canonization Experiments

This repository contains two sets of experiments:

1. ModelNet
2. rotatedMNIST

## Setup

Clone the repository:

    git clone https://github.com/yonatansverdlov/Canonization.git
    cd Canonization

Create the conda environment:

    conda create -n canon python=3.10 pip -y

Activate it:

    conda activate canon

Install PyTorch with CUDA 12.1:

    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Install the remaining required packages:

    python -m pip install torch-geometric lightning kornia numpy scipy hilbertcurve h5py tqdm

## Experiments

### ModelNet

The ModelNet experiments are divided into two parts:

1. Covering number experiments
2. Training experiments

#### Covering Number Experiments

First create the data.

From the repository root, run:

    cd ModelNet/data_creation
    python create_data.py --dataset_name 10 --P 256

For ModelNet40, run:

    python create_data.py --dataset_name 40 --P 256

Here:

    dataset_name = 10 or 40
    P = number of sampled points per shape

Then run the distance computation.

From the repository root, run:

    cd ../compute_distances
    python compute_distance.py --dataset_name 10 --P 256

For ModelNet40, run:

    python compute_distance.py --dataset_name 40 --P 256

The script compares:

    regular distance
    group distance
    sort distance
    Hilbert distance
#### Training Experiments

The training code uses the HDF5 ModelNet40 point-cloud dataset.

Download the data from Kaggle:

[ModelNet40 Kaggle Dataset](https://www.kaggle.com/datasets/leiting0/modelnet40)

Put the extracted data here:

    ModelNet/training/data/modelnet40_ply_hdf5_2048/

To create ModelNet10 from ModelNet40, run:

    cd ModelNet/training
    python create_h5_modelnet10_from_modelnet40.py

In order to train ModelNet10:

    python train.py --dataset modelnet10 --ordering hilbert --run_5_seeds true
    python train.py --dataset modelnet10 --ordering lex --run_5_seeds true
    python train.py --dataset modelnet10 --ordering ply --run_5_seeds true

In order to train ModelNet40:

    python train.py --dataset modelnet40 --ordering hilbert --run_5_seeds true
    python train.py --dataset modelnet40 --ordering lex --run_5_seeds true
    python train.py --dataset modelnet40 --ordering ply --run_5_seeds true

The training script reports train/test accuracy for each seed and then mean/std over all seeds.
### rotatedMNIST

The rotatedMNIST experiments contain two scripts:

1. Training
2. Distance computation

#### Training

From the repository root, run:

    cd mnist
    python train.py --model_type cnn

The available model types are:

    cnn
    average
    learned_can

By default, the script runs with one seed.

To reproduce the paper setting, use five seeds:

    python train.py --model_type cnn --num_seeds 5
    python train.py --model_type average --num_seeds 5
    python train.py --model_type learned_can --num_seeds 5

#### Distance Computation

From the repository root, run:

    cd mnist
    python compute_distance.py