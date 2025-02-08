import pandas as pd

class ModisDataLoader:
    def __init__(self, url):
        self.url = url
        self.data = None
    
    def load_data(self):
        """
        Loads MODIS active fire data from the provided URL.
        """
        self.data = pd.read_csv(self.url)
        print("✅ MODIS fire data loaded!")
    
    def filter_data(self):
        """
        Filters data for relevant columns and high-confidence fires.
        """
        self.data = self.data[['latitude', 'longitude', 'acq_date', 'confidence', 'frp']]
        self.data.rename(columns={'acq_date': 'date'}, inplace=True)
        self.data = self.data[self.data['confidence'] >= 80]  # High-confidence fires only
        self.data['date'] = pd.to_datetime(self.data['date'])
        print("✅ MODIS fire data filtered!")
    
    def get_data(self):
        return self.data

