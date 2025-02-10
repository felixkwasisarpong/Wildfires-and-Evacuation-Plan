import pandas as pd
from sklearn.model_selection import train_test_split
from dataloader.climate_data_loader import ClimateDataLoader
from dataloader.modis_data_loader import ModisDataLoader
from preprocessing.data_merger import DataMerger
from preprocessing.filter import DateFilter
from models.wildfire_predictor import WildfirePredictor

# Load and filter MODIS fire data
modis_loader = ModisDataLoader(url="https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_USA_contiguous_and_Hawaii_7d.csv")
modis_loader.load_data()
modis_loader.filter_data()
fires_data = modis_loader.get_data()

# Load and filter climate data
climate_loader = ClimateDataLoader(url="https://www.ncdc.noaa.gov/cdo-web/api/v2/data.csv")
climate_loader.load_data()
climate_loader.filter_data()
climate_data = climate_loader.get_data()

# Merge data and create ground truth
data_merger = DataMerger(fire_data=fires_data, climate_data=climate_data)
data_merger.merge_data()
data_merger.create_fire_labels()
merged_data = data_merger.get_data()

# Filter by date range
date_filter = DateFilter(start_date="2024-06-01", end_date="2024-08-31")
filtered_data = date_filter.filter_by_date(merged_data)

# Define features (X) and target (y)
X = filtered_data[['TEMP', 'WIND_SPEED', 'HUMIDITY']]
y = filtered_data['fire_tomorrow']

# Split data into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
wildfire_predictor = WildfirePredictor(X_train, y_train, X_test, y_test)
wildfire_predictor.train_model()
wildfire_predictor.evaluate_model()
