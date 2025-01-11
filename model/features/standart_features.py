import pandas as pd

class StandartFeaturesMixin:
    def __init__(self):
        self.df: pd.DataFrame() = None
        self.cat_features: list = []
        self.numeric_features: list = []

    def insert_shifts_norms(self, windows_shifts_norms = [3, 6, 18]):
        """
        Добавляет нормализованные значения цены с учетом сдвигов.

        Параметры:
            windows_shifts_norms (list): Список сдвигов для нормализации.
        """
        for i in windows_shifts_norms:
            self.df[f'close_norms_{i}'] = self.df['close']/self.df['close'].shift(i)
            self.df[f'close_high_norms_{i}'] = self.df['close']/self.df['close'].shift(i)
            self.df[f'high_norms_{i}'] = self.df['high']/self.df['high'].shift(i)
            self.df[f'low_norms_{i}'] = self.df['low']/self.df['low'].shift(i)

            if f'close_norms_{i}' not in self.numeric_features:
                self.numeric_features += [f'close_norms_{i}', f'close_high_norms_{i}', f'high_norms_{i}', f'low_norms_{i}']
    
    def insert_rolling_means(self, windows_ma = [3, 6, 18]):
        """
        Добавляет скользящие средние для указанных окон.

        Параметры:
            windows_ma (list): Список размеров окон для скользящих средних.
        """
        # скользящие средние 
        for i in windows_ma:
            self.df[f'ma_{i}'] = self.df['close'].rolling(window = i, closed="left").mean()
            self.df[f'close_normed_ma_{i}'] = self.df['close']/self.df[f'ma_{i}']

            if f'ma_{i}' not in self.numeric_features:
                self.numeric_features += [f'ma_{i}', f'close_normed_ma_{i}']
    
    def insert_exp_rolling_means(self, windows_ema = [3, 6, 18]):
        """
        Добавляет экспоненциальные скользящие средние для указанных окон.

        Параметры:
            windows_ema (list): Список размеров окон для EMA.
        """
        # экспоненциальные скользящие средние
        for i in windows_ema:
            self.df[f'ema_{i}'] = (self.df['close']).ewm(span=i).mean()
            self.df[f'close_normed_ema_{i}'] = self.df['close']/self.df[f'ema_{i}']

            if f'ema_{i}' not in self.numeric_features:
                self.numeric_features += [f'ema_{i}', f'close_normed_ema_{i}']
    
    def insert_rsi(self, windows_rsi = [3, 6, 18]):
        for i in windows_rsi:
            self.df[f'rsi_{i}'] = self.df['close'] - self.df['close'].shift(i)
            self.df[f'close_normed_rsi_{i}'] = self.df['close']/self.df[f'rsi_{i}']

            if f'rsi_{i}' not in self.numeric_features:
                self.numeric_features += [f'rsi_{i}', f'close_normed_rsi_{i}']
    
    def insert_high_low_diff(self, windows_hl = [3, 6, 18]):
        for i in windows_hl:
            self.df[f'hl_diff_{i}'] = self.df['high'].shift(i) - self.df['low'].shift(i)
            self.df[f'close_normed_hl_diff_{i}'] = self.df['close']/self.df[f'hl_diff_{i}']

            if f'hl_diff_{i}' not in self.numeric_features:
                self.numeric_features += [f'hl_diff_{i}', f'close_normed_hl_diff_{i}']
    