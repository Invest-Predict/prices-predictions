import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt 

def butter_filter(data, cutoff_frequency, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


class TrendFeaturesMixin:
    # в модерации весь класс 
    def __init__(self):
        self.df: pd.DataFrame = None
        self.cat_features: list = []
        self.numeric_features: list = []

    def insert_butter_filter(self, fs=144, cutoff_frequency=5, order=4, visualise=False):
            self.df['butter_filter_trend'] = butter_filter(self.df['close'].values, cutoff_frequency, fs, order)
            self.df['butter_filter_std'] = self.df['close'] - self.df['butter_filter_trend']
            if 'butter_filter_trend' not in self.numeric_features:
                self.numeric_features += ['butter_filter_trend', 'butter_filter_std']

            if visualise:
                plt.plot(self.df['utc'], self.df['open'], label='Original')
                plt.plot(self.df['utc'], self.df['butter_filter_trend'], label='Filtered Trend', color='red')
                plt.plot(self.df['utc'], self.df['butter_filter_std'], label='Short Term Component', color='blue')
                if self.df.shape[0] > 10:
                    plt.xticks(self.df['utc'][::self.df.shape[0] // 10], rotation=45)
                else:
                    plt.xticks(rotation=45)
                plt.legend()
                plt.show()
    
    def insert_trend_rsi(self, windows = [3, 6, 18], filters = ['butter']): # TODO generalize for several filters
        if 'butter_filter_trend' not in self.df.columns:
            self.insert_butter_filter()
        
        for i in windows:
            self.df[f'butter_filter_trend_rsi_{i}'] = self.df['butter_filter_trend'] - self.df['butter_filter_trend'].shift(i)
            self.df[f'close_normed_butter_filter_trend_rsi_{i}'] = self.df['close'] / self.df[f'butter_filter_trend_rsi_{i}']

            if f'butter_filter_trend_rsi_{i}' not in self.numeric_features:
                self.numeric_features += [f'close_normed_butter_filter_trend_rsi_{i}']
                # self.numeric_features += [f'butter_filter_trend_rsi_{i}', f'close_normed_butter_filter_trend_rsi_{i}']

    def insert_trend_rolling_means(self, windows = [3, 6, 18]):
        if 'butter_filter_trend' not in self.df.columns:
            self.insert_butter_filter()
        
        for i in windows:
            self.df[f'butter_filter_trend_ma_{i}'] = (self.df['butter_filter_trend']).ewm(span=i).mean()
            self.df[f'close_normed_butter_filter_trend_ma_{i}'] = self.df['close'] / self.df[f'butter_filter_trend_ma_{i}']

            if f'butter_filter_trend_ma_{i}' not in self.numeric_features:
                self.numeric_features += [f'close_normed_butter_filter_trend_ma_{i}']
                # self.numeric_features += [f'butter_filter_trend_ma_{i}', f'close_normed_butter_filter_trend_ma_{i}']
    
    def insert_trend_deviation(self):
        if 'butter_filter_trend' not in self.df.columns:
            self.insert_butter_filter()
        
        self.df['butter_filter_trend_deviation'] = (self.df['close'] - self.df['butter_filter_trend']) / self.df['butter_filter_trend']
        self.df['close_normed_butter_filter_trend_deviation'] = self.df['close'] / self.df['butter_filter_trend_deviation']

        if 'butter_filter_trend_deviation' not in self.numeric_features:
            self.numeric_features += ['close_normed_butter_filter_trend_deviation']
            # self.numeric_features += ['butter_filter_trend_deviation', 'close_normed_butter_filter_trend_deviation']