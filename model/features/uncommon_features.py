import pandas as pd
import numpy as np

class UncommonFeaturesMixin:
    def __init__(self):
        self.df: pd.DataFrame() = None
        self.cat_features: list = []
        self.numeric_features: list = []

    def insert_bollinger(self): # TODO add other windows
        """
        Добавляет индикатор полос Боллинджера и нормализованные значения цены относительно полос.

        Полосы Боллинджера рассчитываются на основе скользящего среднего и стандартного отклонения.
        Создаются следующие признаки:
            - 'SD': Стандартное отклонение.
            - 'upper_bollinger': Верхняя полоса Боллинджера.
            - 'low_bollinger': Нижняя полоса Боллинджера.
            - 'close_normed_upper_bollinger': Нормализованное значение цены относительно верхней полосы.
            - 'close_normed_low_bollinger': Нормализованное значение цены относительно нижней полосы.
        """
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
        """
        Добавляет стохастический осциллятор и его нормализованные значения.

        Стохастический осциллятор рассчитывается на основе минимальных и максимальных значений цены за указанное окно.
        Создаются следующие признаки:
            - 'stochastic_oscillator': Значение стохастического осциллятора.
            - 'close_normed_stochastic_oscillator': Нормализованное значение цены относительно осциллятора.
            - Для каждого окна в windows_st:
                - 'stochastic_oscillator_ma_{i}': Скользящее среднее осциллятора.
                - 'close_normed_stochastic_oscillator_ma_{i}': Нормализованное значение цены относительно скользящего среднего.
        
        Параметры:
            windows_st (list): Список окон для расчета скользящих средних осциллятора.
        """
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
        """
        Добавляет случайный прогноз целевой переменной.

        Признак:
            - 'target_predict': Случайный прогноз на основе экспоненциального преобразования и смещения.
        """
        self.df['target_predict'] = np.exp(np.log1p(self.df['close'].shift(15)) + 4 * np.random.normal(1)) - self.df['close']
        if 'target_predict' not in self.numeric_features:
            self.numeric_features += ['target_predict']