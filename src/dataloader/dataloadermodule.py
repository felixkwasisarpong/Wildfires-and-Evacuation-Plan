import os
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, random_split
import glob
from .FireSpreadDataset import FireSpreadDataset
from typing import List, Optional, Union
from torch.utils.data import Dataset
from config import config
class WildfireDataset(Dataset):
    def __init__(self, data_dir: str, batch_size: int, n_leading_observations: int, n_leading_observations_test_adjustment: int,
                 crop_side_length: int, load_from_hdf5: bool, num_workers: int, remove_duplicate_features: bool,
                 features_to_keep: Union[Optional[List[int]], str] = None, return_doy: bool = False,
                 data_fold_id: int = 0):

        self.n_leading_observations_test_adjustment = n_leading_observations_test_adjustment
        self.data_fold_id = data_fold_id
        self.return_doy = return_doy
        self.features_to_keep = features_to_keep if type(features_to_keep) != str else None
        self.remove_duplicate_features = remove_duplicate_features
        self.num_workers = num_workers
        self.load_from_hdf5 = load_from_hdf5
        self.crop_side_length = crop_side_length
        self.n_leading_observations = n_leading_observations
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self):
        train_years, val_years, test_years = self.split_fires(self.data_fold_id)
        self.train_dataset = FireSpreadDataset(data_dir=self.data_dir, included_fire_years=train_years,
                                               n_leading_observations=self.n_leading_observations,
                                               n_leading_observations_test_adjustment=None,
                                               crop_side_length=self.crop_side_length,
                                               load_from_hdf5=self.load_from_hdf5, is_train=True,
                                               remove_duplicate_features=self.remove_duplicate_features,
                                               features_to_keep=self.features_to_keep, return_doy=self.return_doy,
                                               stats_years=train_years)
        self.val_dataset = FireSpreadDataset(data_dir=self.data_dir, included_fire_years=val_years,
                                             n_leading_observations=self.n_leading_observations,
                                             n_leading_observations_test_adjustment=None,
                                             crop_side_length=self.crop_side_length,
                                             load_from_hdf5=self.load_from_hdf5, is_train=True,
                                             remove_duplicate_features=self.remove_duplicate_features,
                                             features_to_keep=self.features_to_keep, return_doy=self.return_doy,
                                             stats_years=train_years)
        self.test_dataset = FireSpreadDataset(data_dir=self.data_dir, included_fire_years=test_years,
                                              n_leading_observations=self.n_leading_observations,
                                              n_leading_observations_test_adjustment=self.n_leading_observations_test_adjustment,
                                              crop_side_length=self.crop_side_length,
                                              load_from_hdf5=self.load_from_hdf5, is_train=False,
                                              remove_duplicate_features=self.remove_duplicate_features,
                                              features_to_keep=self.features_to_keep, return_doy=self.return_doy,
                                              stats_years=train_years)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    @staticmethod
    def split_fires(data_fold_id):
        """Split the years into train/val/test set.

        Args:
            data_fold_id (int): Index of the respective split to choose.

        Returns:
            tuple: Lists of years for training, validation, and testing.
        """
        folds = [(2018, 2019, 2020, 2021),
                 (2018, 2019, 2021, 2020),
                 (2018, 2020, 2019, 2021),
                 (2018, 2020, 2021, 2019),
                 (2018, 2021, 2019, 2020),
                 (2018, 2021, 2020, 2019),
                 (2019, 2020, 2018, 2021),
                 (2019, 2020, 2021, 2018),
                 (2019, 2021, 2018, 2020),
                 (2019, 2021, 2020, 2018),
                 (2020, 2021, 2018, 2019),
                 (2020, 2021, 2019, 2018)]

        train_years = list(folds[data_fold_id][:2])
        val_years = list(folds[data_fold_id][2:3])
        test_years = list(folds[data_fold_id][3:4])

        print(f"Using the following dataset split:\nTrain years: {train_years}, Val years: {val_years}, Test years: {test_years}")

        return train_years, val_years, test_years