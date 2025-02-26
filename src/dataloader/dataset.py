import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from utils.helpers import normalize_image

class WildfireDataset(Dataset):
    def __init__(self, hdf5_path):
        with h5py.File(hdf5_path, "r") as file:
            self.images = file["images"][:]
            self.labels = file["labels"][:]

        self.images = normalize_image(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label
