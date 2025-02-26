import os
from dataloader.extract import WildfireDataExtractor
from dataloader.preprocessing import Preprocessor
from Training.trainer import Trainer
import ee
from config import config
def main():
    ee.Authenticate()
    ee.Initialize(project='wildfire-451604')
    region = config.gee["region"]
    extractor = WildfireDataExtractor(region=region)
    geotiff_path = config.paths["geotiff_path"]
    hdf5_path = config.paths["hdf5_path"]

    # if not extractor.check_existing_data(geotiff_path):
    #     print("Extracting wildfire dataset...")
    #     extractor.extract_data()

    #     print("Converting GeoTIFF to HDF5...")
    #     preprocessor = Preprocessor(geotiff_path, hdf5_path)
    #     preprocessor.convert_geotiff_to_hdf5()
    # if not os.path.exists(hdf5_path):
    #     preprocessor = Preprocessor(geotiff_path, hdf5_path)
    # print("Starting model training...")
    Trainer().train()
    print("Model training complete.")

if __name__ == "__main__":
    main()
