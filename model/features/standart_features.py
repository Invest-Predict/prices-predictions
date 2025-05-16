import pandas as pd
import numpy as np

class StandartFeaturesMixin:
    def __init__(self):
        self.df: pd.DataFrame = None
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
            self.df[f'close_high_norms_{i}'] = self.df['close']/self.df['high'].shift(i)
            self.df[f'close_low_norms_{i}'] = self.df['close']/self.df['low'].shift(i)
            self.df[f'high_norms_{i}'] = self.df['high']/self.df['high'].shift(i)
            self.df[f'low_norms_{i}'] = self.df['low']/self.df['low'].shift(i)

            if f'close_norms_{i}' not in self.numeric_features:
                self.numeric_features += [f'close_norms_{i}', f'close_high_norms_{i}', 
                                          f'high_norms_{i}', f'low_norms_{i}', f'close_low_norms_{i}']
    
    def insert_rsi(self, windows_rsi = [3, 6, 18]):
        """
        Добавляет значения индекса относительной силы (RSI).

        Параметры:
            windows_rsi (list): Список временных окон для расчета RSI.
        """
        for i in windows_rsi:
            self.df[f'rsi_{i}'] = self.df['close'] - self.df['close'].shift(i)
            self.df[f'close_normed_rsi_{i}'] = self.df['close']/self.df[f'rsi_{i}']

            if f'rsi_{i}' not in self.numeric_features:
                # self.numeric_features += [f'rsi_{i}', f'close_normed_rsi_{i}']
                self.numeric_features += [f'rsi_{i}']

    
    def insert_high_low_diff(self, windows_hl = [3, 6, 18]):
        """
        Добавляет разницу между high и low в свече ценами для заданных временных окон.

        Параметры:
            windows_hl (list): Список временных окон для расчета разницы high-low.
        """
        for i in windows_hl:
            self.df[f'hl_diff_{i}'] = self.df['high'].shift(i) - self.df['low'].shift(i)
            self.df[f'close_normed_hl_diff_{i}'] = self.df['close']/self.df[f'hl_diff_{i}']

            if f'hl_diff_{i}' not in self.numeric_features:
                # self.numeric_features += [f'hl_diff_{i}', f'close_normed_hl_diff_{i}']
                self.numeric_features += [f'hl_diff_{i}']
    