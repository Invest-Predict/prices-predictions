from types import NoneType
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import adfuller, zivot_andrews
from datetime import date

from .features import StandartFeaturesMixin, TimeFeaturesMixin, TrendFeaturesMixin, UncommonFeaturesMixin, SmoothingFeaturesMixin
from .features import SmallFeaturesMixin

# Здесь все признаки и все по датафрейму

class FinData(StandartFeaturesMixin, TimeFeaturesMixin, TrendFeaturesMixin, 
              UncommonFeaturesMixin, SmoothingFeaturesMixin, SmallFeaturesMixin):
    """
    Класс для обработки финансовых данных. 
    Позволяет загружать данные, фильтровать их по времени, добавлять признаки, 
    визуализировать и подготавливать таргет для моделей машинного обучения.
    """
    def __init__(self, df, column_names=None, fill_skips = False):
        """
        Инициализирует объект FinData, загружая данные из CSV-файла.

        Параметры:
            df (str | pd.Dataframe): Путь к CSV-файлу или pd.Dataframe с данными .
        """
        if isinstance(df, pd.DataFrame):
            self.df = df.copy()
        else:
            self.df = pd.read_csv(df)

        if column_names is not None:
            self.df = self.df.rename(columns=column_names, errors='ignore')
        
        self.df.utc = pd.to_datetime(self.df.utc).dt.tz_localize(None)
        self.df.drop_duplicates(inplace=True) 
        self.target : list

        self.cat_features = []
        self.numeric_features = ['volume'] # я бы остальное по умолчанию не стала добавлять, потому что оно не нормировано 

        if fill_skips:
            self._fill_skips()

        self.make_binary_class_target(target_name="direction_binary_0", ind = 0)
        self.make_binary_class_target(target_name="direction_binary_1", ind = 1)

    def _fill_skips(self):
        self.df.set_index('utc', inplace=True)

        # Создаем полный временной ряд с минутными интервалами
        full_index = pd.date_range(start=self.df.index.min(), end=self.df.index.max(), freq='T')

        # Фильтруем временной ряд, оставляя только минуты с 7:00 до 21:00
        full_index = full_index[(full_index.time >= dt.time(7, 0)) & (full_index.time <= dt.time(21, 0))]

        # Реиндексируем исходный DataFrame по полному временному ряду
        df_reindexed = self.df.reindex(full_index)
        # Заполняем пропущенные значения
        df_reindexed['volume'] = df_reindexed['volume'].fillna(0)  # volume = 0 для пропущенных свечей
        df_reindexed['close'] = df_reindexed['close'].ffill()  # close_{t-1} для пропущенных свечей
        df_reindexed['open'] = df_reindexed['close'].shift(1).fillna(df_reindexed['close'])  # open_t = close_{t-1}
        df_reindexed['high'] = df_reindexed['close'].shift(1).fillna(df_reindexed['close'])  # high_t = close_{t-1}
        df_reindexed['low'] = df_reindexed['close'].shift(1).fillna(df_reindexed['close'])  # low_t = close_{t-1}

        # Убедимся, что open, high, low, close равны close_{t-1} для пропущенных свечей
        df_reindexed['open'] = df_reindexed['close'].shift(1).fillna(df_reindexed['close'])
        df_reindexed['high'] = df_reindexed['close'].shift(1).fillna(df_reindexed['close'])
        df_reindexed['low'] = df_reindexed['close'].shift(1).fillna(df_reindexed['close'])

        # Сбрасываем индекс, чтобы 'utc' стал столбцом
        df_reindexed.reset_index(inplace=True)
        df_reindexed.rename(columns={'index': 'utc'}, inplace=True)
        self.df = df_reindexed

    def make_binary_class_target(self, target_name, ind):
        """ 
        Создаёт бинарный таргет на основе изменения цены закрытия.

        Параметры:
            target_name (str): Название колонки для таргета.
        """
        if ind == 0:
            self.df[target_name] = (self.df['close'].shift(-1) > self.df['close']).astype('int')
        elif ind == 1:
            self.df[target_name] = (self.df['close'].shift(-1) >= self.df['close']).astype('int')
        self.target = [target_name]
    

    def make_both_binary_class_target(self, target_name='direction_binary'):
        """ 
        Создаёт оба бинарный таргета на основе изменения цены закрытия. (Без учёта параметра ind)

        Параметры:
            target_name (str): Название колонки для таргета.
        """
        self.df[target_name + '_0'] = (self.df['close'].shift(-1) > self.df['close']).astype('int')
        self.df[target_name + '_1'] = (self.df['close'].shift(-1) >= self.df['close']).astype('int')

    # def make_long_strat_target(self, target_name, commission):
    #     self.df["vol_up"] = (self.df['close'].shift(-1) - self.df['close']) / ((self.df['close'].shift(-1) + self.df['close']) / 2)
    #     self.df[target_name] = self.df[target_name] = np.where(self.df["vol_up"] > commission * 2, 1, 0)


    # def make_short_strat_target(self, target_name, commission):
    #     self.df["vol_down"] = (- self.df['close'].shift(-1) + self.df['close']) / ((self.df['close'].shift(-1) + self.df['close']) / 2)
    #     self.df[target_name] = (self.df["vol_down"] > commission*2).astype('int')


    def get_numeric_features(self):
        """
        Возвращает список добавленных к текущему моменту числовых признаков.

        Возвращает:
            list: Список названий числовых колонок.
        """
        return self.numeric_features
    
    def get_cat_features(self):
        """
        Возвращает список добавленных к текущему моменту категориальных признаков.

        Возвращает:
            list: Список названий категориальных колонок.
        """
        return self.cat_features
    
    def keep_specific_time(self, start_time : dt.time, end_time : dt.time):
        self.df = self.df[(self.df['utc'].dt.time >= start_time) & (self.df['utc'].dt.time <= end_time)]

    def restrict_time_down(self, date : dt.datetime = None, months = 2, days = 0):
        """
        Ограничивает данные, оставляя строки после указанной даты.

        Параметры:
            date (datetime, optional): Дата отсечения. Если не указано, используется текущая дата минус заданный интервал.
            months (int): Количество месяцев до текущей даты для отсечения.
            days (int): Количество дней до текущей даты для отсечения.
        """
        if date is None:
            date = self.df['utc'].iloc[-1] - pd.DateOffset(months=months, days=days)
        self.df = self.df[self.df["utc"] >= date].reset_index().drop(columns=['index'])

    def restrict_time_up(self, date : dt.datetime = None, months = 2, days = 0):
        """
        Ограничивает данные, оставляя строки до указанной даты.

        Параметры:
            date (datetime, optional): Дата отсечения. Если не указано, используется текущая дата минус заданный интервал.
            months (int): Количество месяцев до текущей даты для отсечения.
            days (int): Количество дней до текущей даты для отсечения.
        """
        if date is None:
            date = self.df['utc'].iloc[-1] - pd.DateOffset(months=months, days=days)
        self.df = self.df[self.df["utc"] <= date].reset_index().drop(columns=['index'])

    # Функция для увеличения интервала свечей
    def merge_candles(self, freq):
        """
        Объединяет свечи в соответствии с указанной частотой.

        Параметры:
            freq (str): Частота объединения (например, '1H' для объединения по часам).
        """
        self.df = self.df.set_index('utc').groupby(pd.Grouper(freq=freq)).agg({'open': 'first', 'close': 'last', 'high': 'max', 'low': 'min', 'volume': 'sum'}).dropna().reset_index()

    def visualize_time_frame(self,
                             datetime_start, 
                             datetime_end, 
                             columns = ['candle'], candle_freq=None,
                             predictor=None,
                             cmap=None, line_kwargs=None):
        """
        Визуализирует данные за указанный временной интервал.

        Параметры:
            datetime_start (datetime): Datetime начала.
            datetime_end (datetime): Datetime конца.
            columns (list(str)): Список столбцов, которые нужно визуализировать; 'candle' визуализирует свечи целиком.
            candle_freq (str): Частота свечей для отрисовки, которую передаем в pd.Grouper. None - не меняем интервал.
            predictor (Callable): Функция или вызываемый объект, предсказания которого визуализируются.
            cmap (str | Colormap): Название или объект Colormap. 
            line_kwargs (dict): Аргументы, которые передаются в plt.plot.
        """

        if predictor is not None:
            if 'candle' not in columns:
                raise ValueError('Must draw candles in order to draw predictions.')
            if candle_freq is not None:
                raise ValueError("Frequencies of predictions and candles don't align.")

        if line_kwargs is None:
            line_kwargs = {}

        if cmap is not None:
            cmap = plt.get_cmap(cmap)

        vis_data = self.df
        vis_data = vis_data[(vis_data['utc'] <= datetime_end) & (vis_data['utc'] >= datetime_start)]

        plt.figure(figsize=(12, 6))
        legend = False
        for i, column in enumerate(columns):
            if column == 'candle':
                candle_vis_data = vis_data.set_index('utc')[['open', 'close', 'high', 'low']]

                if candle_freq is not None:
                    candle_vis_data = candle_vis_data.groupby(pd.Grouper(freq=candle_freq)).agg({'open': 'first', 'close': 'last', 'high': 'max', 'low': 'min'})
                
                up = candle_vis_data[candle_vis_data['close'] > candle_vis_data['open']].dropna()
                down = candle_vis_data[candle_vis_data['close'] < candle_vis_data['open']].dropna()

                width_wide = (min(pd.Series(candle_vis_data.index).shift(-1) - candle_vis_data.index)).total_seconds() / 86400 # типо % дня
                width_narrow = width_wide / 5
                edge_width = 40 / (len(candle_vis_data) + 3)

                col_up = 'green'
                col_down = 'red'
                edge_color = 'black'

                plt.bar(up.index, up['high']-up['low'], width_narrow, bottom=up['low'], color=col_up, edgecolor=edge_color, linewidth=edge_width) 
                plt.bar(up.index, up['close']-up['open'], width_wide, bottom=up['open'], color=col_up, edgecolor=edge_color, linewidth=edge_width)

                plt.bar(down.index, down['high']-down['low'], width_narrow, bottom=down['low'], color=col_down, edgecolor=edge_color, linewidth=edge_width) 
                plt.bar(down.index, down['open']-down['close'], width_wide, bottom=down['close'], color=col_down, edgecolor=edge_color, linewidth=edge_width)

                if predictor is not None:
                    pred = predictor(vis_data)

                    # сопоставим каждой свече предсказание для неё и достанем отдельно положительные и отрицательные
                    candles_pos = candle_vis_data[1:][pred[:-1] == 1]
                    candles_neg = candle_vis_data[1:][pred[:-1] == 0]

                    marker_size = edge_width * 80

                    shift = (max(candle_vis_data['high']) - min(candle_vis_data['low'])) / 15

                    plt.scatter(x=candles_pos.index, y=candles_pos['high'] + shift, marker='^', color=col_up, s=marker_size, edgecolors=edge_color, linewidths=edge_width)
                    plt.scatter(x=candles_neg.index, y=candles_neg['low'] - shift, marker='v', color=col_down, s=marker_size, edgecolors=edge_color, linewidths=edge_width)

            else:
                legend = True
                if cmap is not None:
                    line_kwargs['color'] = cmap(i / max(1, len(columns) - 1))
                plt.plot(vis_data['utc'], vis_data[column], label=column, **line_kwargs)

        if legend:
            plt.legend(facecolor='lightgrey', edgecolor='black', title='Columns')


    def insert_all(self, features_settings : dict | None = None, mini_features  = None):
        if features_settings == None:
            standart_windows = list(range(1, 20))
            self.insert_shifts_norms(standart_windows)
            self.insert_rolling_means(standart_windows)
            self.insert_exp_rolling_means(standart_windows)
            self.insert_rsi(standart_windows)
            self.insert_bollinger()
            self.insert_high_low_diff(standart_windows)
            # self.insert_stochastic_oscillator(standart_windows)
        else: 
            features = list(features_settings.keys())
            if "shifts_norms" in features:
                self.insert_shifts_norms(features_settings["shifts_norms"])
            if "ma" in features:
                self.insert_rolling_means(features_settings["ma"])
            if "ema" in features:
                self.insert_exp_rolling_means(features_settings["ema"])
            if "boll" in features:
                self.insert_bollinger()
            if "rsi" in features:
                self.insert_rsi(features_settings["rsi"])
            if "hl_diff" in features:
                self.insert_high_low_diff(features_settings["hl_diff"])
            if "stoch_osc" in features:
                self.insert_stochastic_oscillator(features_settings["stoch_osc"])
            if "rand_pred" in features:
                self.insert_random_prediction()
            if "mini_features" in features:
                self.insert_small_close_shifts(features_settings["mini_features"])

        return self.numeric_features, self.cat_features


    def get_columns(self):
        """
        Возвращает список всех колонок текущего DataFrame.

        Возвращает:
            list: Список названий всех колонок DataFrame.
        """
        return self.df.columns
    
    def check_stationarity(self, columns):
            """
            Проверяет стационарность столбцов с использованием тестов Phillips-Perron (PP),
            теста Дики-Фуллера (ADF), и теста Зиво-Эндрюса (Zivot-Andrews).

            Параметры:
                columns (list): Список названий столбцов для проверки на стационарность.

            Выводит результаты тестов для каждого столбца. Если хотя бы один тест обнаруживает
            нестационарность, выводит сообщение с указанием тестов и их p-value.
            """
            for column in columns:
                if column in self.df.columns:
                    series = self.df[column].dropna()
                    try:
                        # результаты тестов
                        pp_result = PhillipsPerron(series)
                        adf_result = adfuller(series, autolag='AIC')
                        
                        # za_result = zivot_andrews(series, trim=0.15)

                        # проверка p-value для каждого теста
                        non_stationary_tests = []
                        if pp_result.pvalue > 0.05:
                            non_stationary_tests.append(f"Phillips-Perron (p-value: {pp_result.pvalue:.5f})")
                        if adf_result[1] > 0.05:
                            non_stationary_tests.append(f"ADF (p-value: {adf_result[1]:.5f})")
                        # if za_result and za_result[1] > 0.05:
                        #     non_stationary_tests.append(f"Zivot-Andrews (p-value: {za_result[1]:.5f})")

                        if non_stationary_tests:
                            print(f"Столбец: {column}")
                            print("  Нестационарность обнаружена в следующих тестах:")
                            for test in non_stationary_tests:
                                print(f"    - {test}")
                    except Exception as e:
                        print(f"Ошибка при обработке столбца '{column}': {e}")
                else:
                    print(f"Столбец '{column}' не найден в DataFrame.")

    def print_correlations(self, columns: list, column_with=None):
        """
        Выводит корреляции Пирсона указанных столбцов DataFrame с заданной колонкой.
        Если колонка не указана, используется таргет по умолчанию.

        Параметры:
            columns (list): Список названий столбцов, для которых вычисляются корреляции.
            column_with (str, optional): Название колонки, с которой вычисляются корреляции. 
                                         Если не указано, используется self.target.

        Выводит:
            Корреляции в порядке возрастания.
        """
        if column_with is None:
            column_with = self.target[0]

        if column_with not in self.df.columns:
            print(self.df.columns)
            raise ValueError(f"Колонка '{column_with}' не найдена в DataFrame.")

        correlations = self.df[columns + [column_with]].corr()[column_with].drop(index=column_with).sort_values()
        print("Корреляции столбцов с колонкой '{}':".format(column_with))
        print(correlations)

    




