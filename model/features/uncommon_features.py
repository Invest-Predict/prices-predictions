import pandas as pd
import numpy as np

class UncommonFeaturesMixin:
    def __init__(self):
        self.df: pd.DataFrame() = None
        self.cat_features: list = []
        self.numeric_features: list = []

    def insert_bollinger(self): # TODO add other windows
        if 'ma_16' not in self.df.columns:
            self.insert_rolling_means([16])
        self.df['SD'] =np.sqrt(np.square((self.df["ma_16"] - self.df["close"])).rolling(16, closed='left').sum())
        self.df['upper_bollinger'] = self.df["ma_16"] + self.df['SD'] * 2
        self.df['low_bollinger'] = self.df["ma_18"] - self.df['SD'] * 2

        self.df['close_normed_upper_bollinger'] = self.df['close']/self.df['upper_bollinger']
        self.df['close_normed_low_bollinger'] = self.df['close']/self.df['low_bollinger']

        if 'SD' not in self.numeric_features:
                self.numeric_features += ['SD', 'close_normed_upper_bollinger', 'close_normed_low_bollinger']
    
    def insert_stochastic_oscillator(self, windows_st = [3, 6, 18]): # TODO add other windows

        self.df['stochastic_oscillator'] = (self.df['close'] - self.df['low'].shift(18)) / (self.df['high'].shift(18) - self.df['low'].shift(18))
        self.df['close_normed_stochastic_oscillator'] = self.df['close'] / self.df['stochastic_oscillator']
        
        if 'stochastic_oscillator' not in self.numeric_features:
            self.numeric_features += ['stochastic_oscillator', 'close_normed_stochastic_oscillator']

        for i in windows_st:
            self.df[f'stochastic_oscillator_ma_{i}'] = self.df['stochastic_oscillator'].rolling(i).mean()
            self.df[f'close_normed_stochastic_oscillator_ma_{i}'] = self.df['close'] / self.df[f'stochastic_oscillator_ma_{i}']

            if f'stochastic_oscillator_ma_{i}' not in self.numeric_features:
                self.numeric_features += [f'stochastic_oscillator_ma_{i}', f'close_normed_stochastic_oscillator_ma_{i}']
    
    def insert_random_prediction(self): # TODO think, if other normalization is needed
        self.df['target_predict'] = np.exp(np.log1p(self.df['close'].shift(15)) + 4 * np.random.normal(1)) - self.df['close']
        if 'target_predict' not in self.numeric_features:
            self.numeric_features += ['target_predict']