import h5py
import torch
from torch.utils.data import Dataset

class WildfireDataset(Dataset):
    def __init__(self, hdf5_path="data/wildfire_dataset.h5"):
        with h5py.File(hdf5_path, "r") as hdf:
            self.data = hdf["wildfire_data"][:]
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  # Using self as the target for now
