import os
from dataloader.extract import WildfireDataExtractor
from dataloader.preprocessing import Preprocessor
from Training.trainer import train
from config import config
def main():
    extractor = WildfireDataExtractor()
    geotiff_path = config.paths["geotiff_path"]
    hdf5_path = config.paths["hdf5_path"]

    if not extractor.check_existing_data(hdf5_path):
        print("Extracting wildfire dataset...")
        extractor.extract_data()

        print("Converting GeoTIFF to HDF5...")
        preprocessor = Preprocessor(geotiff_path, hdf5_path)
        preprocessor.convert_geotiff_to_hdf5()

    print("Starting model training...")
    train()

if __name__ == "__main__":
    main()
