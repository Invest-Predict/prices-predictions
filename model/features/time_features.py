import pandas as pd

class TimeFeaturesMixin:
    def __init__(self):
        self.df: pd.DataFrame() = None
        self.cat_features: list = []
        self.numeric_features: list = []

    def insert_time_features(self):
        # Добавляет минуты, дни, часы 
        # Убрать может раз вообще незначимо
                                  
        self.df['hours'] = self.df['utc'].dt.hour
        self.df['day'] = self.df['utc'].dt.day_of_year
        self.df['minute'] = (self.df['utc'].dt.minute + 60 * self.df['hours'])

        if 'hours' not in self.cat_features:
            self.cat_features += ['hours', 'day', 'minute']
    
    def insert_holidays(self):
    
        self.df['is_holiday'] = self.df['utc'].dt.strftime('%m-%d').isin(['12-31', '01-01', '01-07', '02-14', '02-23', '03-08', '05-01']).astype('int')
        self.df['pred_holiday'] = self.df['utc'].dt.strftime('%m-%d').isin(['11-11', '11-30', '12-30', '02-13', '02-22', '03-07', '04-30']).astype('int') # + black friday
        self.df['week_pred_holiday'] = self.df['utc'].dt.strftime('%m-%d').isin(['12-24', '02-07', '02-16', '03-01', '04-24']).astype('int')

        if 'is_holiday' not in self.cat_features:
            self.cat_features += ['is_holiday', 'pred_holiday', 'week_pred_holiday']
    
    def insert_seasons(self):
        def get_season(x):
            if x < 3 or x == 12:
                return 1
            elif x < 6:
                return 2
            elif x < 9:
                return 3
            else:
                return 4

        self.df['day_time'] = self.df['utc'].dt.hour.where(self.df['utc'].dt.hour > 12, 0)  # morning
        self.df['day_time'] = self.df['day_time'].mask(self.df['day_time'] > 18, 2)  # evening
        self.df['day_time'] = self.df['day_time'].mask(self.df['day_time'] > 2, 1) # afternoon


        self.df['season'] = self.df['utc'].dt.month.apply(lambda x: get_season(x))

        if 'season' not in self.cat_features:
            self.cat_features += ['day_time', 'season']
    