#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation


def augment_anisotropic_scale(pointcloud):
    scales = np.random.uniform(0.8, 1.2, size=(3,)).astype(np.float32)
    return pointcloud * scales


def augment_so3_rotation(pointcloud):
    """
    Centers the point cloud and applies a uniformly sampled
    random proper rotation (SO(3)).
    """
    # 1. Center the point cloud at the origin
    pointcloud = pointcloud - np.mean(pointcloud, axis=0, keepdims=True)

    # 2. Generate random SO(3) matrix
    R = Rotation.random().as_matrix().astype(np.float32)

    # 3. Apply rotation
    return np.dot(pointcloud, R)


def farthest_point_sample(xyz, npoint):
    """
    Numpy implementation of Farthest Point Sampling.
    Returns the INDICES of the sampled points.
    """
    N, C = xyz.shape
    centroids = np.zeros(npoint, dtype=int)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)

    return centroids


def load_ordered_data(partition, data_root="data"):
    dataset_name = "modelnet40_ply_hdf5_2048"
    dataset_dir = os.path.join(data_root, dataset_name)

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Could not find dataset directory at {dataset_dir}")

    all_data = []
    all_label = []

    file_pattern = os.path.join(dataset_dir, f'ply_data_{partition}*.h5')
    files = glob.glob(file_pattern)

    if not files:
        raise FileNotFoundError(f"No h5 files found matching: {file_pattern}")

    for h5_name in files:
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            all_data.append(data)
            all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label


class OrderedModelNet40(Dataset):
    def __init__(self, num_points, partition='train', ordering='lex', data_root='data',
                 jitter_sigma=0.02, jitter_clip=0.04, dataset_stride=1, use_fps=False,
                 apply_jitter=False, apply_anisotropic_scale=False,
                 apply_random_permutation=False, apply_rotation=False):
        """
        Dataloader for Point Clouds.
        """
        self.num_points = num_points
        self.partition = partition
        self.ordering = ordering

        # Augmentation parameters & toggles
        self.apply_jitter = apply_jitter
        self.apply_anisotropic_scale = apply_anisotropic_scale
        self.apply_random_permutation = apply_random_permutation
        self.apply_rotation = apply_rotation
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip

        # --- FPS Pre-computation and Caching Logic ---
        dataset_name = "modelnet40_ply_hdf5_2048"
        cache_dir = os.path.join(data_root, dataset_name, "fps_cache")
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(cache_dir, f"{partition}_fps_{num_points}.npz")

        if use_fps:
            if os.path.exists(cache_file):
                print(f"Loading pre-computed FPS data from {cache_file}...")
                cached_data = np.load(cache_file)
                self.data = cached_data['data']
                self.label = cached_data['label']
            else:
                print(f"FPS Cache not found. Generating {num_points}-point FPS downsampling...")
                raw_data, raw_label = load_ordered_data(partition, data_root=data_root)
                num_samples = raw_data.shape[0]
                fps_data = np.zeros((num_samples, num_points, 3), dtype=np.float32)

                for i in range(num_samples):
                    if i % 500 == 0:
                        print(f"Processing {i}/{num_samples}...")

                    pc = raw_data[i]
                    fps_idx = farthest_point_sample(pc, num_points)
                    fps_data[i] = pc[fps_idx]

                self.data = fps_data
                self.label = raw_label
                np.savez_compressed(cache_file, data=self.data, label=self.label)
        else:
            self.data, self.label = load_ordered_data(partition, data_root=data_root)

        # Subset the dataset based on the provided dataset_stride
        if dataset_stride > 1:
            self.data = self.data[::dataset_stride]
            self.label = self.label[::dataset_stride]

        self.data = self.data.astype(np.float32, copy=False)

    def __getitem__(self, item):
        pointcloud = self.data[item].copy()
        label = self.label[item]

        # Fallback to stride if FPS wasn't used and shape doesn't match
        if pointcloud.shape[0] != self.num_points:
            stride = pointcloud.shape[0] // self.num_points
            pointcloud = pointcloud[::stride][:self.num_points]

        # --- Apply Strict SO(3) Rotation (Applied to Train AND Test if toggled) ---
        if self.apply_rotation:
            pointcloud = augment_so3_rotation(pointcloud)

        # --- Apply Requested Augmentations (Train Only) ---
        if self.partition == 'train':
            # 1. Anisotropic Scale
            if self.apply_anisotropic_scale:
                pointcloud = augment_anisotropic_scale(pointcloud)

            # 2. Jitter
            if self.apply_jitter:
                noise = np.random.normal(0, self.jitter_sigma, size=pointcloud.shape).astype(np.float32)
                noise = np.clip(noise, -self.jitter_clip, self.jitter_clip)
                pointcloud += noise

        # --- Random Permutation ---
        if self.apply_random_permutation:
            perm = np.random.permutation(self.num_points)
            pointcloud = pointcloud[perm]

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    print("Testing Train Partition with Full SO(3) Rotations...")
    train = OrderedModelNet40(
        1024,
        partition='train',
        ordering='hilbert',
        dataset_stride=3,
        use_fps=True,
        apply_jitter=True,
        apply_anisotropic_scale=True,
        apply_random_permutation=True,
        apply_rotation=True
    )

    for data, label in train:
        print("Point Cloud Shape:", data.shape)
        print("Label Shape:", label.shape)
        break