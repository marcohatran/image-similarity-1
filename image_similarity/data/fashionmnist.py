import numpy as np
from tensorflow.keras.datasets import fashion_mnist

from .dataset import Dataset


class FashionMnistDataset(Dataset):
    def __init__(self):
        (self.train_images, self.train_labels), (
            self.test_images,
            self.test_labels,
        ) = fashion_mnist.load_data()
        self.train_images = np.expand_dims(self.train_images, axis=3) / 255.0
        self.test_images = np.expand_dims(self.test_images, axis=3) / 255.0
