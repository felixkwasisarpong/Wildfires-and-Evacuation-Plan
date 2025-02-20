import ee
import os
from config import config
class WildfireDataExtractor:
    def __init__(self, start_date=config.gee["start_date"], end_date=config.gee["end_date"], region=None):
        ee.Initialize()
        self.start_date = start_date
        self.end_date = end_date
        self.region = region if region else ee.Geometry.Rectangle([-125, 32, -114, 42])

    def extract_data(self):
        fire_strength = ee.ImageCollection("MODIS/006/MCD14ML").filterDate(self.start_date, self.end_date).mean().clip(self.region)
        vegetation = ee.ImageCollection("MODIS/006/MOD13A1").select("NDVI").filterDate(self.start_date, self.end_date).mean().clip(self.region)
        precipitation = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").select("pr").filterDate(self.start_date, self.end_date).mean().clip(self.region)
        wind_speed = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").select("vs").filterDate(self.start_date, self.end_date).mean().clip(self.region)
        elevation = ee.Image("USGS/SRTMGL1_003").clip(self.region)

        combined = fire_strength.addBands([
            vegetation, precipitation, wind_speed, elevation
        ])

        task = ee.batch.Export.image.toDrive(
            image=combined,
            description="Wildfire_Dataset",
            folder="Wildfire_Data",
            scale=500,
            region=self.region.getInfo(),
            fileFormat="GeoTIFF"
        )
        task.start()
        print("Exporting dataset to Google Drive...")

    def check_existing_data(self, path=config.paths["hdf5_path"]):
        return os.path.exists(path)
