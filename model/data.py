import pandas as pd
import polars as pl
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import adfuller, zivot_andrews
from datetime import date

from .features import StandartFeaturesMixin, TimeFeaturesMixin, TrendFeaturesMixin, UncommonFeaturesMixin, SmoothingFeaturesMixin

# Здесь все признаки и все по датафрейму

class FinData(StandartFeaturesMixin, TimeFeaturesMixin, TrendFeaturesMixin, UncommonFeaturesMixin, SmoothingFeaturesMixin):
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

        if column_names is not None:
            self.df = self.df.rename(columns=column_names, errors='ignore')
        
        self.df.utc = pd.to_datetime(self.df.utc).dt.tz_localize(None)
        self.df.drop_duplicates(inplace=True) 
        self.target : list

        self.cat_features = []
        self.numeric_features = ['volume'] # я бы остальное по умолчанию не стала добавлять, потому что оно не нормировано 
        self.make_binary_class_target(target_name="direction_binary")

    def make_binary_class_target(self, target_name):
        """
        Создаёт бинарный таргет на основе изменения цены закрытия.

        Параметры:
            target_name (str): Название колонки для таргета.
        """
        self.df[target_name] = (self.df['close'].shift(-1) > self.df['close']).astype('int')
        self.target = [target_name]

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

    def restrict_time_by_candles(self, date: dt.datetime, number_before, num_after):
        # режем по количеству свечей 
        pass

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
        self.df = self.df.set_index('utc').groupby(pd.Grouper(freq=freq)).agg({'open': 'first', 'close': 'last', 'high': 'max', 'low': 'min'}).dropna().reset_index()

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

            
    def insert_all(self, windows_shifts_norms=None, 
                         windows_ma=None, 
                         windows_ema=None,
                         windows_rsi=None, 
                         windows_high_low_diff=None, 
                         windows_stoch_osc=None, 
                         common_windows=[3, 6, 18]):
        """
        Вставляет в DataFrame все реализованные признаки.

        Параметры:
            windows_shifts_norms (list, optional): Список временных окон для нормализованных сдвигов.
            windows_ma (list, optional): Список временных окон для расчёта простых скользящих средних.
            windows_ema (list, optional): Список временных окон для расчёта экспоненциальных скользящих средних.
            windows_rsi (list, optional): Список временных окон для расчёта индикатора RSI.
            windows_high_low_diff (list, optional): Список временных окон для расчёта разницы между high и low.
            windows_stoch_osc (list, optional): Список временных окон для стохастического осциллятора.
            common_windows (list): Список временных окон по умолчанию для вставки нескольких признаков.
        """
        self.insert_shifts_norms(common_windows if windows_shifts_norms==None else windows_shifts_norms)
        self.insert_time_features()
        self.insert_holidays()
        self.insert_seasons()
        self.insert_rolling_means(common_windows if windows_ma==None else windows_ma)
        self.insert_exp_rolling_means(common_windows if windows_ema==None else windows_ema)
        self.insert_rsi(common_windows if windows_rsi==None else windows_rsi)
        self.insert_bollinger()
        self.insert_high_low_diff(common_windows if windows_high_low_diff==None else windows_high_low_diff)
        self.insert_stochastic_oscillator(common_windows if windows_stoch_osc==None else windows_stoch_osc)
        self.insert_random_prediction() # мамочки 
        # self.insert_butter_filter()
        # self.insert_trend_rsi()
        # self.insert_trend_rolling_means()
        # self.insert_trend_deviation()

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

    




