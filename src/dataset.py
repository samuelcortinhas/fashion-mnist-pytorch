import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
# from torchvision import transforms


def load_data(train_path, test_path):
    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Pre-process
    y = train_data.label
    y_test = test_data.label
    X = train_data.drop("label", axis=1) / 255
    X_test = test_data.drop("label", axis=1) / 255

    # Reshape
    X = X.values.reshape(-1, 1, 28, 28)
    X_test = X_test.values.reshape(-1, 1, 28, 28)

    # Create validation set
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.9, test_size=0.1, random_state=0
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


# Having problems with torchvision on my pc (skipping for now)

# def get_transforms(degrees=30, scale=(0.8, 1.2), ratio=(1, 1)):
#     # Data augmentation
#     data_transforms = transforms.Compose(
#         [
#             transforms.RandomRotation(degrees=degrees),
#             transforms.RandomResizedCrop(size=(28, 28), scale=scale, ratio=ratio),
#         ]
#     )
#     return data_transforms


class FashionMNIST(Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.values) if y is not None else None
        self.transform = transform

    # Get item in position given by index
    def __getitem__(self, index):
        sample = self.X[index]
        if self.transform:
            sample = self.transform(sample)

        if self.y is not None:
            return sample, self.y[index]
        else:
            return sample

    # Length of dataset
    def __len__(self):
        return self.X.shape[0]
