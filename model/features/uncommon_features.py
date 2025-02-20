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
        
        if 'close_normed_stochastic_oscillator' not in self.numeric_features:
            # self.numeric_features += ['close_normed_stochastic_oscillator']
            self.numeric_features += ['stochastic_oscillator']

        for i in windows_st:
            self.df[f'stochastic_oscillator_ma_{i}'] = self.df['stochastic_oscillator'].rolling(i).mean()
            self.df[f'close_normed_stochastic_oscillator_ma_{i}'] = self.df['close'] / self.df[f'stochastic_oscillator_ma_{i}']

            if f'closed_normed_stochastic_oscillator_ma_{i}' not in self.numeric_features:
                self.numeric_features += [f'close_normed_stochastic_oscillator_ma_{i}']
                # self.numeric_features += [f'stochastic_oscillator_ma_{i}', f'close_normed_stochastic_oscillator_ma_{i}']
    
    def insert_random_prediction(self): # TODO think, if other normalization is needed
        """
        Добавляет случайный прогноз целевой переменной.

        Признак:
            - 'target_predict': Случайный прогноз на основе экспоненциального преобразования и смещения.
        """
        # self.df['target_predict'] = np.exp(np.log1p(self.df['close'].shift(15)) + 4 * np.random.normal(1)) - self.df['close']
        # if 'target_predict' not in self.numeric_features:
        #     self.numeric_features += ['target_predict']

        n = self.df.shape[0]
        sum_s, sum_t = np.log1p(self.df['close'][0]), self.df['volume'][0]
        for i in range(n - 1):
            sum_s += np.log1p(self.df['close'][i + 1]) - np.log1p(self.df['close'][i])
            sum_t += self.df['volume'][i + 1]

        sum_s, sum_t = sum_s / n, sum_t / n
        a = sum_s / sum_t

        su = (np.log1p(self.df['close'][0]) - a * self.df['volume'][0]) / np.sqrt(self.df['volume'][0])
        for i in range(n - 1):
            diff_t = self.df['volume'][i + 1]
            su += (np.log1p(self.df['close'][i + 1]) - np.log1p(self.df['close'][i]) - a * diff_t) / np.sqrt(diff_t)
        su /= n

        sigma2 = 0

        for i in range(n - 1):
            diff_t = self.df['volume'][i + 1]
            sigma2 += ((np.log1p(self.df['close'][i + 1]) - np.log1p(self.df['close'][i]) - a * diff_t) / np.sqrt(diff_t) - su) ** 2

        sigma2 /= n

        m = 20000000
        s = [0] * (m + 1)
        x_t = [0] * (m + 1)
        eps = np.random.choice([-1, 1], m)
        for i in range(1, m + 1):
            s[i] = s[i - 1] + eps[i - 1]

        def f(t):
            k = t * m
            if abs(k - int(k)) < 1e-6:
                return np.sqrt(m) * (s[int(k)] / np.sqrt(m))
            else:
                return  np.sqrt(m) * (s[int(k)] / np.sqrt(m) + (k - int(k)) * eps[int(k) + 1] / np.sqrt(m))

        log_s0 = np.log1p(self.df['close'][0])
        s_n = [log_s0] * n
        for i in range(1, n):
            # print(a * df['T_n'][i], "-", f(df['T_n'][i] / m) * np.sqrt(sigma2))
            s_n[i] = log_s0 + a * self.df['volume'][i] + f(self.df['volume'][i] / m) * np.sqrt(sigma2)
        
        self.df['target_predict'] = np.exp(s_n)
        if 'target_predict' not in self.numeric_features:
                self.numeric_features += ['target_predict']

    def insert_angle(self, other_name : str):
        """
        Добавляет признак 'angle_{other_name}' - угол между ценой закрытия target-акции и ценой закрытия акции other_name.
        """
        self.df[f'angle_{other_name}'] = np.arctan(self.df["close"] - self.df["close"].shift()) - np.arctan(self.df[f"close_{other_name}"] - self.df[f"close_{other_name}"].shift())


    def insert_angle_ln(self, other_name : str):
        """
        Добавляет признак 'angle_ln_{other_name}' - угол на графике логарифмов, чтобы убрать зависимость от масштаба.
        """
        self.df[f'angle_ln_{other_name}'] = np.arctan(np.log(self.df["close"]) - np.log(self.df["close"].shift())) - np.arctan(np.log(self.df[f"close_{other_name}"]) - np.log(self.df[f"close_{other_name}"]).shift())