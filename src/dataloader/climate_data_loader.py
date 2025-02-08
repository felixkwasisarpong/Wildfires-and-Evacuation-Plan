import pandas as pd

class ClimateDataLoader:
    def __init__(self, url):
        self.url = url
        self.data = None
    
    def load_data(self):
        """
        Loads climate data from the provided URL.
        """
        self.data = pd.read_csv(self.url)
        print("✅ Climate data loaded!")
    
    def filter_data(self):
        """
        Filters data for relevant columns.
        """
        self.data = self.data[['DATE', 'LATITUDE', 'LONGITUDE', 'TEMP', 'WIND_SPEED', 'HUMIDITY']]
        self.data.rename(columns={'DATE': 'date', 'LATITUDE': 'latitude', 'LONGITUDE': 'longitude'}, inplace=True)
        self.data['date'] = pd.to_datetime(self.data['date'])
        print("✅ Climate data filtered!")
    
    def get_data(self):
        return self.data
