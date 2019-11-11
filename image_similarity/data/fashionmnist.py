from .dataset import Dataset


class FashionMnistDataset(Dataset):
    def __init__(self):
        {
            label: np.flatnonzero(self.train_labels == label)
            for label in self.unique_label
        }
