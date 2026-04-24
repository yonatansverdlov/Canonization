# Canonization Experiments

This repository contains two sets of experiments:

1. ModelNet
2. rotatedMNIST

## ModelNet

The ModelNet experiments are divided into two parts:

1. Covering number experiments
2. Training experiments

### Environment setup

Create the conda environment:

    conda env create -f environment.yml

Activate it:

    conda activate canonization

Alternatively, install the requirements manually:

    pip install -r requirements.txt

### Covering Number Experiments

First create the data.

From the repository root, run:

    cd ModelNet/data_creation
    python create_data.py --dataset_name 10 --P 256

For ModelNet40, run:

    python create_data.py --dataset_name 40 --P 256

Here:

    dataset_name = 10 or 40
    P = number of sampled points per shape

Then run the distance computation:

    cd ../compute_distances
    python compute_distances.py --dataset_name 10 --P 256

The script compares:

    regular distance
    group distance
    sort distance
    Hilbert distance