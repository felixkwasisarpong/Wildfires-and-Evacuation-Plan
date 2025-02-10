import pandas as pd

class DateFilter:
    def __init__(self, start_date, end_date):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
    
    def filter_by_date(self, data):
        """
        Filters the dataset by the provided date range.
        """
        return data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]
    
