import os
from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib.image import imread

from image_similarity.utils import cached_property


class Dataset(object):
    def __init__(self, train_images, train_labels, test_images, test_labels: Path):
        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels

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
            self.train_images[anchor_indices, :],
            self.train_images[positive_indices, :],
            self.train_images[negative_indices, :],
        ]

    def _get_triplet(self) -> Tuple[int, int, int]:
        label_l, label_r = np.random.choice(self.unique_label, 2, replace=False)
        anchor, positive = np.random.choice(self.map_train[label_l], 2, replace=False)
        negative = np.random.choice(self.map_train[label_r])
        return anchor, positive, negative


class DirectoryDataset(Dataset):
    def __init__(self, dataset_path: Path):
        super().__init__()

        images, labels = self._load_dataset(dataset_path)

    def _load_dataset(self, dataset_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        images = []
        labels = []
        for directory in os.listdir(dataset_path):
            for image in sorted((dataset_path / directory).glob("*.jpg")):
                images.append(np.squeeze(np.asarray(imread(image))))
                labels.append(directory)
        return np.array(images), np.array(labels)

    def _preprocess(self, images: np.ndarray, labels: np.ndarray):
        images = _normalize(images)


def _normalize(image: np.ndarray):
    # min_value = np.min(array)
    # max_value = np.max(array)
    # array = (array - min_value) / (max_value - min_value)
    image / 255.0
    return image
