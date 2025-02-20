import h5py
import rasterio
import numpy as np
import os

class Preprocessor:
    def __init__(self, geotiff_path="data/wildfire_data.tif", hdf5_path="data/wildfire_dataset.h5"):
        self.geotiff_path = geotiff_path
        self.hdf5_path = hdf5_path

    def convert_geotiff_to_hdf5(self):
        if os.path.exists(self.hdf5_path):
            print("HDF5 file already exists.")
            return

        with rasterio.open(self.geotiff_path) as src:
            data = src.read()
            print(f"Loaded GeoTIFF with shape: {data.shape}")

        with h5py.File(self.hdf5_path, "w") as hdf:
            hdf.create_dataset("wildfire_data", data=data)
        print("Converted GeoTIFF to HDF5.")
