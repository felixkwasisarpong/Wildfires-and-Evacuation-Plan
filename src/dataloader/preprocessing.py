import os
import h5py
import rasterio
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
from dataloader.FireSpreadDataset import FireSpreadDataset
from config import config
from pathlib import Path
class Preprocessor:
    def __init__(self, base_dir, output_dir):
        self.base_dir = base_dir
        self.output_dir = output_dir

    years = [2018, 2019, 2020, 2021]
    dataset = FireSpreadDataset(data_dir=config.paths["geotiff_path"], included_fire_years=years, n_leading_observations=1, crop_side_length=128, load_from_hdf5=False, is_train=True, remove_duplicate_features=False, stats_years=(2018,2019))
    data_gen = dataset.get_generator_for_hdf5()
    hdf5_path = config.paths['hdf5_path']                              
    for y in years:
        target_dir = f"{hdf5_path}/{y}"
        Path(target_dir).mkdir(parents=True, exist_ok=True)

    for year, fire_name, img_dates, lnglat, imgs in tqdm(data_gen):

        target_dir = f"{hdf5_path}/{year}"
        h5_path = f"{target_dir}/{fire_name}.hdf5"

        if Path(h5_path).is_file():
            print(f"File {h5_path} already exists, skipping...")
            continue

        with h5py.File(h5_path, "w") as f:
            dset = f.create_dataset("data", imgs.shape, data=imgs)
            dset.attrs["year"] = year
            dset.attrs["fire_name"] = fire_name
            dset.attrs["img_dates"] = img_dates
            dset.attrs["lnglat"] = lnglat