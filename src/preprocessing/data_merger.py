import pandas as pd

class DataMerger:
    def __init__(self, fire_data, climate_data):
        self.fire_data = fire_data
        self.climate_data = climate_data
        self.merged_data = None
    
    def merge_data(self):
        """
        Merges fire data with climate data.
        """
        self.merged_data = pd.merge(self.fire_data, self.climate_data, on=['date', 'latitude', 'longitude'], how='left')
        print("✅ Data merged successfully!")
    
    def create_fire_labels(self):
        """
        Creates ground truth labels: 1 if fire exists today, else 0.
        """
        self.merged_data['fire_today'] = 1
        self.merged_data['fire_tomorrow'] = self.merged_data['fire_today'].shift(-1, fill_value=0)
        print("✅ Ground truth labels created!")
    
    def get_data(self):
        return self.merged_data
