import pandas as pd
import polars as pl
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

# Здесь все признаки и все по датафрейму

# to do: визуализации свечей в visualize_time_frame 
# Добавить все признаки, которые можно сюда добавить 


class FinData():
    """
    Класс для обработки финансовых данных. 
    Позволяет загружать данные, фильтровать их по времени, добавлять признаки, 
    визуализировать и подготавливать таргет для моделей машинного обучения.
    """
    def __init__(self, df_path = 'datasets/T_yandex_10min.csv'):
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

        
    def make_binary_class_target(self):
        """
        Добавляет два типа таргетов: бинарный и трехклассовый.

        direction_binary: 0 или 1 в зависимости от изменения цены закрытия.
        direction: 0, 1 или 2 для обозначения направления изменения цены (вниз, без изменений, вверх).
        """
        # добавляет два варианта таргета - 2 и 3 класса 
        self.df = pl.from_pandas(self.df)
        self.df = self.df.with_columns(
            pl.when(pl.col('close').shift(-1) > pl.col('close')).then(2)
            .when(pl.col('close').shift(-1) == pl.col('close')).then(1)
            .otherwise(0)
            .alias("direction"),
            pl.when(pl.col('close').shift(-1) > pl.col('close')).then(1)
            .otherwise(0)
            .alias("direction_binary")
        )
        self.df = self.df.to_pandas()

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


    def set_tardet(self, target):
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
                             column, type="line"):
        """
        Визуализирует данные за указанный временной интервал.

        Параметры:
            year_start (int): Год начала.
            month_start (int): Месяц начала.
            day_start (int): День начала.
            year_end (int): Год конца.
            month_end (int): Месяц конца.
            day_end (int): День конца.
            column (str): Название столбца для визуализации.
            type (str): Тип графика (по умолчанию "line").
        """
        vis_data = pl.from_pandas(self.df)
        vis_data = vis_data.filter((pl.col("utc") <= pl.datetime(year_end, month_end, day_end)) 
                                   & (pl.col("utc") >= pl.datetime(year_start, month_start, day_start)))
        vis_data = vis_data.to_pandas()
        plt.figure(figsize=(12, 6))
        plt.plot(vis_data['utc'], vis_data[column], label=f'Data {column}')
    
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

    def insert_time_features(self):
        # Добавляет минуты, дни, часы 
        # Убрать может раз вообще незначимо
        self.df['hours'] = self.df['utc'].dt.hour
        self.df['day'] = self.df['utc'].dt.day_of_year
        self.df['minute'] = (self.df['utc'].dt.minute + 60 * self.df['hours'])

    def insert_rolling_means(self, windows_ma):
        """
        Добавляет скользящие средние для указанных окон.

        Параметры:
            windows_ma (list): Список размеров окон для скользящих средних.
        """
        # скользящие средние 
        for i in windows_ma:
            self.df[f'ma_{i}'] = self.df['close'].rolling(window = i, closed="left").mean()
            self.df[f'close_normed_ma_{i}'] = self.df['close']/self.df[f'ma_{i}']
            
    def insert_exp_rolling_means(self, windows_ema):
        """
        Добавляет экспоненциальные скользящие средние для указанных окон.

        Параметры:
            windows_ema (list): Список размеров окон для EMA.
        """
        # экспоненциальные скользящие средние
        for i in windows_ema:
            self.df[f'ema_{i}'] = (self.df['close']).ewm(span=i).mean()
            self.df[f'close_normed_ema_{i}'] = self.df['open']/self.df[f'ema_{i}']

    def get_columns(self):
        return self.df.columns



        
    




