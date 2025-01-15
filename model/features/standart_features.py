import pandas as pd
import numpy as np

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
    
    def insert_rolling_means(self, windows_ma):
        """
        Добавляет скользящие средние для указанных окон.

        Параметры:
            windows_ma (list): Список размеров окон для скользящих средних.
        """
        # скользящие средние 
        for i in windows_ma:
            self.df[f'ma_{i}'] = self.df['close'].rolling(window = i, closed="left").mean()
            column_name = f'ma_{i}'
            for idx in range(i):
                if pd.isna(self.df.at[idx, column_name]):
                    self.df.at[idx, column_name] = self.df['close'][:idx + 1].mean()
                    
            # как будто хочется везде так сделать, но хз
            self.df[f'close_normed_ma_{i}'] = self.df['close']/self.df[f'ma_{i}']
            self.df[f'low_normed_ma_{i}'] = self.df['low']/self.df[f'ma_{i}']
            self.df[f'high_normed_ma_{i}'] = self.df['high']/self.df[f'ma_{i}']

            if f'ma_{i}' not in self.numeric_features:
                self.numeric_features += [f'ma_{i}', 
                                          f'close_normed_ma_{i}'
                                          f'low_normed_ma_{i}', 
                                          f'high_normed_ma_{i}']
    
    def insert_exp_rolling_means(self, windows_ema):
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

    def insert_weighted_rolling_means(self, windows_wma):
        """
        Добавляет взвешенные скользящие средние для указанных окон.

        Параметры:
            windows_wma (list): Список размеров окон для WMA.
        """
        # взвешенные скользящие средние
        for i in windows_wma:
            weights = np.arange(1, i + 1)
            self.df[f'wma_{i}'] = self.df['close'].rolling(window=i).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
            self.df[f'close_normed_wma_{i}'] = self.df['close'] / self.df[f'wma_{i}']

            if f'wma_{i}' not in self.numeric_features:
                self.numeric_features += [f'wma_{i}', f'close_normed_wma_{i}']

    
    def insert_hull_moving_average(self, windows_hma):
        """
        Добавляет скользящую среднюю Халла для указанных окон.

        Параметры:
            windows_hma (list): Список размеров окон для HMA.
        """
        for i in windows_hma:
            half_length = int(i / 2)
            sqrt_length = int(np.sqrt(i))

            wma_half = self.df['close'].rolling(window=half_length).apply(lambda prices: np.dot(prices, np.arange(1, half_length + 1)) / np.arange(1, half_length + 1).sum(), raw=True)
            wma_full = self.df['close'].rolling(window=i).apply(lambda prices: np.dot(prices, np.arange(1, i + 1)) / np.arange(1, i + 1).sum(), raw=True)
            hma = 2 * wma_half - wma_full
            self.df[f'hma_{i}'] = hma.rolling(window=sqrt_length).mean()

            self.df[f'close_normed_hma_{i}'] = self.df['close'] / self.df[f'hma_{i}']

            if f'hma_{i}' not in self.numeric_features:
                self.numeric_features += [f'hma_{i}', f'close_normed_hma_{i}']

    # def insert_adaptive_moving_average(self, windows_ama):
    #     """
    #     Добавляет адаптивную скользящую среднюю (Adaptive Moving Average, AMA).

    #     Параметры:
    #         windows_ama (list): Список размеров окон для AMA.
    #     """
    #     for i in windows_ama:
    #         n_fast = 2 / (2 + 1)  # Коэффициент для быстрой EMA (обычно 2)
    #         n_slow = 2 / (30 + 1)  # Коэффициент для медленной EMA (обычно 30)

    #         # Вычисление изменения цены (Efficiency Ratio)
    #         price_change = abs(self.df['close'] - self.df['close'].shift(i))
    #         volatility = self.df['close'].rolling(window=i).apply(lambda x: np.sum(abs(np.diff(x))), raw=True)
    #         efficiency_ratio = price_change / volatility

    #         # Вычисление сглаживающего коэффициента (Smoothing Constant)
    #         smoothing_constant = (efficiency_ratio * (n_fast - n_slow) + n_slow) ** 2

    #         # Вычисление AMA
    #         ama = [self.df['close'].iloc[0]]  # Инициализация AMA начальным значением цены
    #         for j in range(1, len(self.df)):
    #             ama.append(ama[-1] + smoothing_constant.iloc[j] * (self.df['close'].iloc[j] - ama[-1]))

    #         self.df[f'ama_{i}'] = ama
    #         self.df[f'close_normed_ama_{i}'] = self.df['close'] / self.df[f'ama_{i}']

    #         if f'ama_{i}' not in self.numeric_features:
    #             self.numeric_features += [f'ama_{i}', f'close_normed_ama_{i}']
    
    def insert_rsi(self, windows_rsi):
        for i in windows_rsi:
            self.df[f'rsi_{i}'] = self.df['close'] - self.df['close'].shift(i)
            self.df[f'close_normed_rsi_{i}'] = self.df['close']/self.df[f'rsi_{i}']

            if f'rsi_{i}' not in self.numeric_features:
                self.numeric_features += [f'rsi_{i}', f'close_normed_rsi_{i}']
    
    def insert_high_low_diff(self, windows_hl):
        for i in windows_hl:
            self.df[f'hl_diff_{i}'] = self.df['high'].shift(i) - self.df['low'].shift(i)
            self.df[f'close_normed_hl_diff_{i}'] = self.df['close']/self.df[f'hl_diff_{i}']

            if f'hl_diff_{i}' not in self.numeric_features:
                self.numeric_features += [f'hl_diff_{i}', f'close_normed_hl_diff_{i}']
    