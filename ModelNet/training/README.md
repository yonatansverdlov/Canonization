# ModelNet40 Canonicalization Experiments

This repository contains the code and experimental setup for studying the effects of point cloud canonicalization on the ModelNet40 dataset. The project evaluates canonicalization techniques using a global MLP and conducts a comprehensive rotation study across various frame-alignment models.

## 📂 Data Preparation

Before running the experiments, ensure you have downloaded the **ModelNet40** dataset in HDF5 format. The data should be extracted and placed in the following directory: `~/data/modelnet40_ply_hdf5_2048/`.

Your directory structure must look exactly like this:

```text
~/data/modelnet40_ply_hdf5_2048$ ls
fps_cache                     ply_data_test1.h5             ply_data_train_0_id2file.json  ply_data_train2.h5             ply_data_train_3_id2file.json  shape_names.txt
ply_data_test0.h5             ply_data_test_1_id2file.json  ply_data_train1.h5             ply_data_train_2_id2file.json  ply_data_train4.h5             test_files.txt
ply_data_test_0_id2file.json  ply_data_train0.h5            ply_data_train_1_id2file.json  ply_data_train3.h5             ply_data_train_4_id2file.json  train_files.txt
```
🚀 Experiment 1: Effect of Canonicalization
The main experiment evaluates a global_mlp model on the ModelNet40 dataset to measure the impact of point sequence ordering. The training script is pre-configured with optimized default hyperparameters.

Local Run
To run the main experiment locally using the optimized Hilbert ordering baseline, execute:
```text
Bash
python train.py \
    --exp_name "Best_MLP_Hilbert_Local" \
    --epochs 100 \
    --batch_size 256 \
    --ordering "hilbert"
```
(Note: Parameters like learning rate, weight decay, and dropout are already set as optimal defaults in the script).

🌀 Experiment 2: Rotation Study
The second experiment investigates how different canonicalization models (e.g., PurePCA, FrameAveraging, SkewnessCanon, RandomFrame) handle random point cloud rotations.

Local Run
To test a single model locally (e.g., Model 2: FrameAveraging) with rotation applied, use the following command:
```text
Bash
python train_rot.py \
    --model 2 \
    --exp_name "Best_FrameAveraging_Local" \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.00052445 \
    --weight_decay 0.00000354 \
    --apply_rotation True
```
🛠️ Requirements
Ensure you are using Python 3.13+. It is highly recommended to set up a virtual environment before installing the dependencies.

To install all required packages, run:

```text
Bash
pip install -r requirements.txt