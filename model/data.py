import pandas as pd
import polars as pl
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

# Здесь все признаки и все по датафрейму

# to do: визуализации свечей в visualize_time_frame 
# Добавить все признаки, которые можно сюда добавить 


class FinData():
    def __init__(self, df_path = 'datasets/T_yandex_10min.csv'):
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

        self.df = pl.from_pandas(self.df)
        self.df = self.df.filter(pl.col("utc") >= pl.datetime(year, month, day))
        self.df = self.df.to_pandas()

    def restrict_time_up(self, year=2024, month=9, day=11, date = None):
        # обрезает датасет по времени ДО
        if date is not None:
            self.df = self.df[self.df <= date]
            return

        self.df = pl.from_pandas(self.df)
        self.df = self.df.filter(pl.col("utc") <= pl.datetime(year, month, day))
        self.df = self.df.to_pandas()

    def restrict_time_up_stupidly(self, months=2, days=0):
        # берёт первую дату в датасете (пусть это 2024.09.11) и оберзает все даты большие чем 2024.09.11 + months + days

        last_day = self.df['utc'][0] + pd.DateOffset(months=months, days=days)
        self.restrict_time_up(date=last_day)

    def set_tardet(self, target):
        # если понадобиться 
        self.target = target
    
    def visualize_time_frame(self,
                             year_start, month_start, day_start, 
                             year_end, month_end, day_end, 
                             column, type="line"):
        
        
        vis_data = pl.from_pandas(self.df)
        vis_data = vis_data.filter((pl.col("utc") <= pl.datetime(year_end, month_end, day_end)) 
                                   & (pl.col("utc") >= pl.datetime(year_start, month_start, day_start)))
        vis_data = vis_data.to_pandas()
        plt.figure(figsize=(12, 6))
        plt.plot(vis_data['utc'], vis_data[column], label=f'Data {column}')
    
    # Добавление признаков 
    def insert_shifts_norms(self, windows_shifts_norms):
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

    def insert_rolling_means(self, windows_ma = [3, 6, 18]):
        # скользящие средние 
        for i in windows_ma:
            self.df[f'ma_{i}'] = self.df['close'].rolling(window = i, closed="left").mean()
            self.df[f'close_normed_ma_{i}'] = self.df['close']/self.df[f'ma_{i}']
            
    def insert_exp_rolling_means(self, windows_ema = [3, 6, 18]):
        # экспоненциальные скользящие средние
        for i in windows_ema:
            self.df[f'ema_{i}'] = (self.df['close']).ewm(span=i).mean()
            self.df[f'close_normed_ema_{i}'] = self.df['close']/self.df[f'ema_{i}']
    
    def insert_rsi(self, windows_rsa = [3, 6, 18]):
        for i in windows_rsa:
            self.df[f'rsi_{i}'] = self.df['close'] - self.df['close'].shift(i)
            self.df[f'close_normed_rsi_{i}'] = self.df['close']/self.df[f'rsi_{i}']

    def get_columns(self):
        return self.df.columns



        
    




