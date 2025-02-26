import datetime
import time
import ee
import os
from config import config
from google.cloud import storage

class WildfireDataExtractor:
    def __init__(self, region, start_date=config.gee["start_date"], end_date=config.gee["end_date"]):
        ee.Initialize()
        self.start_date = start_date
        self.end_date = end_date
        self.region = ee.Geometry.Polygon(region)  # Ensure region is a GeoJSON object
        self.location = config.paths["data_dir"]  # Define location name if needed

        # Initialize data sources
        self.srtm = ee.Image("USGS/SRTMGL1_003")
        self.landcover = ee.ImageCollection("MODIS/061/MCD12Q1")
        self.weather = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
        self.weather_forecast = ee.ImageCollection('NOAA/GFS0P25')
        self.drought = ee.ImageCollection("GRIDMET/DROUGHT")
        self.viirs = ee.ImageCollection('NOAA/VIIRS/001/VNP09GA')
        self.viirs_af = ee.FeatureCollection('projects/grand-drive-285514/assets/afall')
        self.viirs_veg_idx = ee.ImageCollection("NOAA/VIIRS/001/VNP13A1")

    def extract_data(self):
        """Exports wildfire-related data to Google Cloud Storage and downloads it if successful."""
        if self.check_existing_data():
            print("Data already exists locally. Skipping extraction.")
            return

        img = self.compute_daily_features(self.start_date, self.end_date, self.region).max().toFloat()
        task = ee.batch.Export.image.toCloudStorage(
            image=img,
            description="Wildfire_Dataset",
            bucket=config.get("output_bucket"),
            fileNamePrefix=f"{config.get('cloud_folder')}/wildfire_data",
            scale=500,
            region=self.region.getInfo()["coordinates"],
            fileFormat="GeoTIFF",
            maxPixels=1e13
        )

        task.start()
        print(f'Starting export task (ID: {task.id})...')

        # Wait for task completion
        while task.status()['state'] in ['READY', 'RUNNING']:
            print(f"Export in progress... Current status: {task.status()['state']}")
            time.sleep(30)

        if task.status()['state'] == 'COMPLETED':
            print("Export completed successfully. Initiating download...")
            self.download_data_from_gcloud_to_local()
        else:
            print(f"Export failed with status: {task.status()['state']} - {task.status().get('error_message', '')}")

    def compute_daily_features(self, start_time: str, end_time: str, geometry: ee.Geometry):
        """Computes daily wildfire-related features."""
        today = datetime.datetime.strptime(start_time, '%Y-%m-%d')
        today_timestamp = int(today.timestamp()) * 1000

        weather = self.weather.filterDate(start_time, end_time).filterBounds(geometry)
        precipitation = weather.select('pr').median().rename("total precipitation")
        wind_velocity = weather.select('vs').median().rename("wind speed")
        temperature_min = weather.select('tmmn').median().rename("minimum temperature")
        temperature_max = weather.select('tmmx').median().rename("maximum temperature")
        specific_humidity = weather.select('sph').median().rename("specific humidity")

        weather_forecast = self.weather_forecast.filterDate(start_time, end_time).filterBounds(geometry)
        forecast_temperature = weather_forecast.select("temperature_2m_above_ground").mean().rename("forecast temperature")
        forecast_specific_humidity = weather_forecast.select("specific_humidity_2m_above_ground").mean().rename("forecast specific humidity")

        elevation = self.srtm.select('elevation')
        slope = ee.Terrain.slope(elevation)
        aspect = ee.Terrain.aspect(elevation)

        drought_index = self.drought.filter(ee.Filter.lte("system:time_start", today_timestamp))\
            .filter(ee.Filter.gte("system:time_end", today_timestamp)).select('pdsi').median()

        igbp_land_cover = self.landcover.filterDate(start_time[:4] + '-01-01', start_time[:4] + '-12-31')\
            .filterBounds(geometry).select('LC_Type1').median()

        def add_acq_hour(feature):
            acq_time_str = ee.String(feature.get("acq_time"))
            acq_time_int = ee.Number.parse(acq_time_str)
            return feature.set({"acq_hour": acq_time_int})

        viirs_img = self.viirs.filterDate(start_time, end_time).filterBounds(geometry).select(['M11', 'I2', 'I1']).median()
        viirs_veg_idx = self.viirs_veg_idx.filterDate(start_time, end_time).filterBounds(geometry).select(['NDVI', 'EVI2']).reduce(ee.Reducer.last())
        viirs_af_img = self.viirs_af.map(add_acq_hour).filterBounds(geometry) \
            .filter(ee.Filter.gte('acq_date', start_time[:-6])) \
            .filter(ee.Filter.lt('acq_date', (
                datetime.datetime.strptime(end_time[:-6], '%Y-%m-%d') + datetime.timedelta(1)).strftime(
                '%Y-%m-%d'))) \
            .filter(ee.Filter.neq('confidence', 'l')).map(self.get_buffer) \
            .reduceToImage(['acq_hour'], ee.Reducer.last()) \
            .rename(['active fire'])

        return ee.ImageCollection(ee.Image([
            viirs_img, viirs_veg_idx, precipitation, wind_velocity, temperature_min, temperature_max, specific_humidity,
            slope, aspect, elevation, drought_index, igbp_land_cover, forecast_temperature, forecast_specific_humidity,
            viirs_af_img
        ]))

    def check_existing_data(self, path=config.paths["hdf5_path"]):
        """Checks if the data already exists locally."""
        return os.path.exists(path)

    def get_buffer(self, feature):
        """Returns buffered feature with a set radius."""
        return feature.buffer(375 / 2).bounds()

    def download_data_from_gcloud_to_local(self):
        """Downloads exported data from Google Cloud to local storage if not already present."""
        if self.check_existing_data():
            print("Data already exists locally. Skipping download.")
            return

        year = str(self.start_date[:4])
        blob_prefix = f"{year}/{self.location}/"
        destination_dir = f"data/{year}/{self.location}/"

        os.makedirs(destination_dir, exist_ok=True)
        self.download_blob(config.get('output_bucket'), blob_prefix, destination_dir)

        

    def download_blob(self, bucket_name: str, blob_name: str, destination_file_name: str):
        """Downloads files from Google Cloud Storage."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=blob_name)

        for blob in blobs:
            filename = f"{blob.name.split('/')[2].replace('.tif', '')}_{blob.name.split('/')[1]}.tif"
            blob.download_to_filename(os.path.join(destination_file_name, filename))
            print(f"Blob {filename} downloaded to {destination_file_name}.")


    def download_data_from_gcloud_to_local(self):  
        if self.check_existing_data():
            print("Data already exists locally. Skipping download.")
            return

        year = str(self.start_date[:4])  # Extract the year from start_date
        blob_prefix = f"{year}/{self.location}/"
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(config.get('output_bucket'))
        blobs = bucket.list_blobs(prefix=blob_prefix)

        destination_dir = f"data/{year}/"
        os.makedirs(destination_dir, exist_ok=True)

        for blob in blobs:
            # Extract the date from the filename
            filename_parts = blob.name.split('/')
            if len(filename_parts) < 3:
                continue  # Skip if the blob name doesn't match expected format
            
            date_str = filename_parts[2].replace('.tif', '')  # Extract the date
            destination_file = os.path.join(destination_dir, f"{date_str}.tif")
            
            blob.download_to_filename(destination_file)
            print(f"Downloaded {date_str}.tif to {destination_dir}")