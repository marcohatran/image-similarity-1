import os
from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib.image import imread

from image_similarity.utils import cached_property


class Dataset(object):
    train_images = np.array([])
    train_labels = np.array([])
    test_images = np.array([])
    test_labels = np.array([])

    @cached_property
    def unique_label(self) -> np.ndarray:
        return np.unique(self.train_labels)

    @cached_property
    def map_train(self):
        return {
            label: np.flatnonzero(self.train_labels == label)
            for label in self.unique_label
        }

    def get_triplet_batch(
        self, batch_size
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        anchor_indices, positive_indices, negative_indices = [], [], []
        for _ in range(batch_size):
            anchor, positive, negative = self._get_triplet()
            anchor_indices.append(anchor)
            positive_indices.append(positive)
            negative_indices.append(negative)
        return [
            self.train_images[anchor_indices],
            self.train_images[positive_indices],
            self.train_images[negative_indices],
        ]

    def _get_triplet(self) -> Tuple[int, int, int]:
        label_l, label_r = np.random.choice(self.unique_label, 2, replace=False)
        anchor, positive = np.random.choice(self.map_train[label_l], 2, replace=False)
        negative = np.random.choice(self.map_train[label_r])
        return anchor, positive, negative

    def __repr__(self):
        return "\n".join(
            [
                f"{self.__class__.__name__}",
                f"  Train images: {self.train_images.shape}",
                f"  Train labels: {self.train_labels.shape}",
                f"  Test images: {self.test_images.shape}",
                f"  Test labels: {self.test_labels.shape}",
                f"  Unique labels: {self.unique_label}",
            ]
        )


class DirectoryDataset(Dataset):
    def __init__(self, dataset_path: Path):
        super().__init__()

        images, labels = self._load_dataset(dataset_path)
        images = images / 255.0

    def _load_dataset(self, dataset_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        images = []
        labels = []
        for directory in os.listdir(dataset_path):
            for image in sorted((dataset_path / directory).glob("*.jpg")):
                images.append(np.squeeze(np.asarray(imread(image))))
                labels.append(directory)
        return np.array(images), np.array(labels)
