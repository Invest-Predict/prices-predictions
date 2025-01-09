import pandas as pd
import polars as pl
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import adfuller
from datetime import date

# Здесь все признаки и все по датафрейму

# to do:
# Добавить все признаки, которые можно сюда добавить 


class FinData():
    """
    Класс для обработки финансовых данных. 
    Позволяет загружать данные, фильтровать их по времени, добавлять признаки, 
    визуализировать и подготавливать таргет для моделей машинного обучения.
    """
    def __init__(self, df_path, column_names=None):
        """
        Инициализирует объект FinData, загружая данные из CSV-файла.

        Параметры:
            df_path (str): Путь к CSV-файлу с данными.
        """
        self.df = pd.read_csv(df_path)

        self.df = self.df.rename(columns={'Yandex open' : 'open', 
                                          'Yandex close' : 'close', 
                                          'Yandex high' : 'high', 
                                          'Yandex low' : 'low', 
                                          'Yandex volume' : 'volume'}, 
                                 errors='ignore')
        
        self.df.utc = pd.to_datetime(self.df.utc).dt.tz_localize(None)
        self.df.drop_duplicates(inplace=True)
        self.target : str

        self.cat_features = []
        self.numeric_features = ['open', 'close', 'high', 'low', 'volume']

        
    def make_binary_class_target(self):
        """
        Добавляет два типа таргетов: бинарный и трехклассовый.

        direction_binary: 0 или 1 в зависимости от изменения цены закрытия.
        direction: 0, 1 или 2 для обозначения направления изменения цены (вниз, без изменений, вверх).
        """
        # добавляет два варианта таргета - 2 и 3 класса 

        # self.df = pl.from_pandas(self.df)
        # self.df = self.df.with_columns(
        #     pl.when(pl.col('close').shift(-1) > pl.col('close')).then(2)
        #     .when(pl.col('close').shift(-1) == pl.col('close')).then(1)
        #     .otherwise(0)
        #     .alias("direction"),
        #     pl.when(pl.col('close').shift(-1) > pl.col('close')).then(1)
        #     .otherwise(0)
        #     .alias("direction_binary")
        # )
        # self.df = self.df.to_pandas()

        self.df["direction_binary"] = (self.df['close'].shift(-1) > self.df['close']).astype('int')

    def restrict_time_down(self, year=2024, month=9, day=11, date=None):
        # обрезает датасет по времени ОТ
        if date is not None:
            self.df = self.df[self.df >= date]
            return 

    def restrict_time_down(self, year, month, day):
        """
        Обрезает данные, оставляя только записи начиная с указанной даты.

        Параметры:
            year (int): Год.
            month (int): Месяц.
            day (int): День.
        """
        self.df = pl.from_pandas(self.df)
        self.df = self.df.filter(pl.col("utc") >= pl.datetime(year, month, day))
        self.df = self.df.to_pandas()

    def restrict_time_up(self, year, month, day):
        """
        Обрезает данные, оставляя только записи до указанной даты.

        Параметры:
            year (int): Год.
            month (int): Месяц.
            day (int): День.
        """
        self.df = pl.from_pandas(self.df)
        self.df = self.df.filter(pl.col("utc") <= pl.datetime(year, month, day))
        self.df = self.df.to_pandas()

    def restrict_time_up_stupidly(self, months=2, days=0):
        # берёт первую дату в датасете (пусть это 2024.09.11) и оберзает все даты большие чем 2024.09.11 + months + days

        last_day = self.df['utc'][0] + pd.DateOffset(months=months, days=days)
        self.restrict_time_up(date=last_day)

    def set_target(self, target):
        """
        Устанавливает таргет для анализа.

        Параметры:
            target (str): Название столбца с таргетом.
        """
        # если понадобиться, а так я обычно не пользуюсь
        self.target = target
    
    def visualize_time_frame(self,
                             year_start, month_start, day_start, 
                             year_end, month_end, day_end, 
                             columns = ['candle'], candle_freq=None,
                             cmap=None, line_kwargs=None):
        """
        Визуализирует данные за указанный временной интервал.

        Параметры:
            year_start (int): Год начала.
            month_start (int): Месяц начала.
            day_start (int): День начала.
            year_end (int): Год конца.
            month_end (int): Месяц конца.
            day_end (int): День конца.
            columns (list(str)): Список столбцов, которые нужно визуализировать; 'candle' визуализирует свечи целиком.
            candle_freq (str): Частота свечей, которую передаем в pd.Grouper. None - не меняем интервал.
            cmap (str | Colormap): Название или объект Colormap. 
            line_kwargs (dict): Аргументы, которые передаются в plt.plot.
        """
        if line_kwargs is None:
            line_kwargs = {}

        if cmap is not None:
            cmap = plt.get_cmap(cmap)

        vis_data = pl.from_pandas(self.df)
        vis_data = vis_data.filter((pl.col("utc") <= pl.datetime(year_end, month_end, day_end)) 
                                   & (pl.col("utc") >= pl.datetime(year_start, month_start, day_start)))
        vis_data = vis_data.to_pandas()
        plt.figure(figsize=(12, 6))
        for i, column in enumerate(columns):
            if column == 'candle':
                candle_vis_data = vis_data.set_index('utc')[['open', 'close', 'high', 'low']]
                if candle_freq is not None:
                    candle_vis_data = candle_vis_data.groupby(pd.Grouper(freq=candle_freq)).agg({'open': 'first', 'close': 'last', 'high': 'max', 'low': 'min'})
                
                up = candle_vis_data[candle_vis_data['close'] > candle_vis_data['open']].dropna()
                down = candle_vis_data[candle_vis_data['close'] < candle_vis_data['open']].dropna()

                num_days = (date(year_end, month_end, day_end) - date(year_start, month_start, day_start)).days + 1

                width_wide = num_days / (len(candle_vis_data) + 3)
                width_narrow = width_wide / 5
                edge_width = 40 / (len(candle_vis_data) + 3)

                col_up = 'green'
                col_down = 'red'
                edge_color = 'black'

                plt.bar(up.index, up['high']-up['low'], width_narrow, bottom=up['low'], color=col_up, edgecolor=edge_color, linewidth=edge_width) 
                plt.bar(up.index, up['close']-up['open'], width_wide, bottom=up['open'], color=col_up, edgecolor=edge_color, linewidth=edge_width)

                plt.bar(down.index, down['high']-down['low'], width_narrow, bottom=down['low'], color=col_down, edgecolor=edge_color, linewidth=edge_width) 
                plt.bar(down.index, down['open']-down['close'], width_wide, bottom=down['close'], color=col_down, edgecolor=edge_color, linewidth=edge_width)

            else:
                if cmap is not None:
                    line_kwargs['color'] = cmap(i / max(1, len(columns) - 1))
                plt.plot(vis_data['utc'], vis_data[column], label=column, **line_kwargs)

        plt.legend(facecolor='lightgrey', edgecolor='black', title='Columns')

    
    # Добавление признаков 
    def insert_shifts_norms(self, windows_shifts_norms):
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
    
    def insert_bollinger(self): # TODO add other windows
        if 'ma_16' not in self.df.columns:
            self.insert_rolling_means([16])
        self.df['SD'] =np.sqrt(np.square((self.df["ma_16"] - self.df["close"])).rolling(16, closed='left').sum())
        self.df['upper_bollinger'] = self.df["ma_16"] + self.df['SD'] * 2
        self.df['low_bollinger'] = self.df["ma_18"] - self.df['SD'] * 2

        self.df['close_normed_upper_bollinger'] = self.df['close']/self.df['upper_bollinger']
        self.df['close_normed_low_bollinger'] = self.df['close']/self.df['low_bollinger']

        if 'SD' not in self.numeric_features:
                self.numeric_features += ['SD', 'low_bollinger', 'upper_bollinger', 'close_normed_upper_bollinger', 'close_normed_low_bollinger']
    
    def insert_high_low_diff(self, windows_hl = [3, 6, 18]):
        for i in windows_hl:
            self.df[f'hl_diff_{i}'] = self.df['high'].shift(i) - self.df['low'].shift(i)
            self.df[f'close_normed_hl_diff_{i}'] = self.df['close']/self.df[f'hl_diff_{i}']

            if f'hl_diff_{i}' not in self.numeric_features:
                self.numeric_features += [f'hl_diff_{i}', f'close_normed_hl_diff_{i}']
    
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
    
    def insert_all(self, common_windows= None):
        if common_windows is None:
            common_windows = [3, 6, 18]
        self.insert_shifts_norms(common_windows)
        self.insert_time_features()
        self.insert_holidays()
        self.insert_rolling_means(common_windows)
        self.insert_exp_rolling_means(common_windows)
        self.insert_rsi(common_windows)
        self.insert_bollinger()
        self.insert_high_low_diff(common_windows)
        self.insert_stochastic_oscillator(common_windows)
        self.insert_random_prediction()

    def get_columns(self):
        return self.df.columns
    
    def check_stationarity(self, columns):
        """
        Проверяет стационарность столбцов с использованием тестов Phillips-Perron (PP),
        теста Дики-Фуллера (ADF).

        Параметры:
            columns (list): Список названий столбцов для проверки на стационарность.

        Выводит результаты тестов для каждого столбца. Если хотя бы один тест обнаруживает
        нестационарность, выводит сообщение с указанием тестов и их p-value.
        """
        for column in columns:
            if column in self.df.columns:
                series = self.df[column].dropna()

                # Результаты тестов
                pp_result = PhillipsPerron(series)
                adf_result = adfuller(series, autolag='AIC')

                # Проверка p-value для каждого теста
                non_stationary_tests = []

                if pp_result.pvalue > 0.05:
                    non_stationary_tests.append(f"Phillips-Perron (p-value: {pp_result.pvalue:.5f})")

                if adf_result[1] > 0.05:
                    non_stationary_tests.append(f"ADF (p-value: {adf_result[1]:.5f})")

                # Вывод результатов
                if non_stationary_tests:
                    print(f"Столбец: {column}")
                    print("  Нестационарность обнаружена в следующих тестах:")
                    for test in non_stationary_tests:
                        print(f"    - {test}")
            else:
                print(f"Столбец '{column}' не найден в DataFrame.")
    