from typing import Iterable
from model import FinData, CatboostFinModel
import datetime as dt
import pandas as pd

import logging


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("testing.log"),
        logging.StreamHandler()
    ]
)

class Backtest():
    def __init__(self, strategies: list[str], args: dict, dfs: list[str], comissions : list[int], timedelta: str = '10min', target: str = 'direction_binary'):
        self._strategies = strategies  # ['long', 'short']
        self._args = args  # usual argumets for CatboostFinModel
        self._dfs = dfs  # list of paths to datasests. For example ['../../datasets/']
        self._comission = comissions  # list of comissions, usually [0.0004]
        self._timedelta = timedelta  # the period of one candle. For example 10min, 1H and etc
        self._target = target
        self.X_train, self.X_val, self.X_test = None,None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.cat, self.num = [], []
    
    def custom_datasets(self, df_path, start_dt, end_dt, train_size, val_size, test_size = None, features = None):
        data = FinData(df_path)
        data.restrict_time_down(start_dt)

        if self._timedelta != '10min':
            data.merge_candles(self._timedelta)
        
        # TODO добавить сюда возможность кастомно добавлять фичи
        data.insert_shifts_norms([3, 6, 18])
        data.insert_rolling_means()
        data.insert_exp_rolling_means()
        data.insert_stochastic_oscillator()
        data.insert_high_low_diff()

        self.cat = data.get_cat_features()
        self.num = data.get_numeric_features()

        if isinstance(train_size, dt.timedelta):  # Можно задавать границу сплита на train, val и test через timedelta: 10 days - train, 3 days - val
            train = data.df[data.df['utc'] <= start_dt + train_size]
            val = data.df[data.df['utc'] > start_dt + train_size][data.df['utc'] <= start_dt + train_size + val_size]

            if test_size is None:
                test = data.df[data.df['utc'] > start_dt + train_size + val_size][data.df['utc'] <= end_dt]
            else:
                test = data.df[data.df['utc'] > start_dt + train_size + val_size][data.df['utc'] <= start_dt + train_size + val_size + test_size]
            self.X_train, self.X_val, self.X_test = train.drop(columns=self._target), val.drop(columns=self._target), test.drop(columns=self._target)
            self.y_train, self.y_val, self.y_test = train[self._target], val[self._target], test[self._target]

            return

        X, y = data.df.drop(columns=self._target), data.df[self._target]
        self.X_train, self.X_val, self.X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size: train_size + val_size + test_size]
        self.y_train, self.y_val, self.y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size: train_size + val_size + test_size]
    
    def test_trading(self, budget, custom_datasets_arg):

        results = []

        for df_path in self._dfs:
            custom_datasets_arg['df_path'] = df_path
            self.custom_datasets(**custom_datasets_arg)

            stock = df_path.split('/')[-1][:-4]  # так обрежеться всё до названия файла из datasets и тажке .csv
            logging.info(f"Backtesting started for stock - {stock}")
            logging.info(f"Train dates: {self.X_train['utc'].iloc[0]} - {self.X_train['utc'].iloc[-1]} | Valid dates: {self.X_val['utc'].iloc[0]} - {self.X_val['utc'].iloc[-1]} | Test dates: {self.X_test['utc'].iloc[0]} - {self.X_test['utc'].iloc[-1]}")

            self.X_train, self.X_val = self.X_train[self.num + self.cat], self.X_val[self.num + self.cat]
            model = CatboostFinModel(self._args)
            model.set_datasets(self.X_train, self.X_val, self.y_train, self.y_val)
            model.set_features(self.num, self.cat)

            model.fit()

            history = pd.DataFrame(columns=["datetime", "budget"])
            history.loc[0] = [self.X_test['utc'].iloc[0], budget]
            money = budget


            for i in range(self.X_test.shape[0] - 1):
                y_pred = model.predict(self.X_test[self.num + self.cat].iloc[i])
                close_in_ten_min = self.X_test['close'].iloc[i + 1]
                open_now = self.X_test['close'].iloc[i]

                history.loc[i + 1] = [self.X_test['utc'].iloc[i + 1], money]

                if money >= open_now and y_pred == 1 and 'long' in self._strategies:
                    commission_now = ((open_now + close_in_ten_min) * self._comission[0]) * (money  // open_now)
                    money += (close_in_ten_min - open_now) * (money  // open_now) - commission_now

                    logging.info(f"LONG! - Date&Time: {self.X_test['utc'].iloc[i]} - I bought Yandex for {open_now} and sold for {close_in_ten_min} + commission {commission_now} -> budget: {money}")
                elif money >= close_in_ten_min and y_pred == 0 and 'short' in self._strategies:
                    commission_now = ((open_now + close_in_ten_min) * self._comission[0]) * (money // close_in_ten_min)
                    money += (open_now - close_in_ten_min) * (money  // open_now) - commission_now
                    logging.info(f"SHORT! - Date&Time: {self.X_test['utc'].iloc[i]} - I bought Yandex for {close_in_ten_min} and sold for {open_now} + commission {commission_now} -> budget: {money}")

            logging.info(f"\n\n\nMy budget before {budget} and after trading {money}\nMommy, are you prod of me?")

            results.append((money - budget,  model.score(self.X_test, self.y_test))) # ны выходе прибыль
        return results


    # @property
    # def strategies(self) -> list:
    #     return self._strategies

    # @strategies.setter
    # def strategies(self, new_strategies: Iterable) -> None:
    #     self._strategies = list(new_strategies)

    # @property
    # def models(self) -> list:
    #     return self._models

    # @models.setter
    # def models(self, new_models: Iterable) -> None:
    #     self._models = list(new_models)

    # @property
    # def dfs(self) -> list:
    #     return self._dfs

    # @dfs.setter
    # def dfs(self, new_dfs: Iterable) -> None:
    #     self._dfs = list(new_dfs)

    # @property
    # def commision(self) -> int:
    #     return self._commision

    # @commision.setter
    # def dfs(self, new_dfs: Iterable) -> None:
    #     self._dfs = list(new_dfs)

    

    # def test_trading(self, df = None, target = 'direction_binary', start_date = None, end_date = None, proportion = [3, 1, 1],
    #                 train_df = None, val_df = None, test_df = None, short = True,
    #                 initial_budget = 10000, cat = [], num = [], commision = 0.0001):
    #     '''
    #     Примитиваня стратегия, пусть мы просто пока покупаем акцию сейчас, если предполагаем, что через десять минут она вырастит в цене
    #     (через 10 минут в этом случае её продаём)
    #     В ином случае мы ничего не делаем (ждём следующий период)
    #     Но также у нас есть ограничение - это бюджет (он ограчен => не всегда сможем купить акцию, чтобы продать её через 10 минут)
    #     '''

    #     if train_df is None or val_df is None or test_df is None:
    #         df_copy = df.copy()
    #         train_size, val_size, test_size = 1000, 180, 180
    #         if end_date is not None:
    #             df_copy = df[df["utc"] <= end_date].reset_index().drop(columns=['index'])
    #         if start_date is not None:
    #                 df_copy = df_copy[df_copy["utc"] >= start_date].reset_index().drop(columns=['index'])
    #                 df_size = df_copy.shape[0]
    #                 train_size, val_size = int(df_size * (proportion[0] / sum(proportion))), int(df_size * (proportion[1] / sum(proportion)))
    #                 test_size = df_size - train_size - val_size

    #         X, y = df_copy.drop(columns=target), df_copy[target]

    #         X_train, X_val, X_test = X[-(train_size + val_size + test_size):-(val_size + test_size)], X[-(val_size + test_size): -test_size], X[-test_size:]
    #         y_train, y_val, y_test = y[-(train_size + val_size + test_size):-(val_size + test_size)], y[-(val_size + test_size): -test_size], y[-test_size:]
        
    #     else:
    #         X_train, X_val, X_test = train_df.drop(columns=target), val_df.drop(columns=target), test_df.drop(columns=target)
    #         y_train, y_val, y_test = train_df[target], val_df[target], test_df[target]
        
    #     logging.info("Backtesting started")
    #     logging.info(f"Train dates: {X_train['utc'].iloc[0]} - {X_train['utc'].iloc[-1]} | Valid dates: {X_val['utc'].iloc[0]} - {X_val['utc'].iloc[-1]} | Test dates: {X_test['utc'].iloc[0]} - {X_test['utc'].iloc[-1]}")

    #     X_train, X_val = X_train[num + cat], X_val[num + cat]
    #     self.set_datasets(X_train, X_val, y_train, y_val)
    #     self.set_features(num, cat)

    #     self.fit()

    #     history = pd.DataFrame(columns=["datetime", "budget"])
    #     history.loc[0] = [X_test['utc'].iloc[0], initial_budget]
    #     money = initial_budget


    #     for i in range(X_test.shape[0] - 1):
    #         y_pred = self.predict(X_test[num + cat].iloc[i])
    #         close_in_ten_min = X_test['close'].iloc[i + 1]
    #         open_now = X_test['close'].iloc[i]

    #         history.loc[i + 1] = [X_test['utc'].iloc[i + 1], money]

    #         if money >= open_now and y_pred == 1:
    #             commission_now = ((open_now + close_in_ten_min) * commision) * (money  // open_now)
    #             money += (close_in_ten_min - open_now) * (money  // open_now) - commission_now

    #             logging.info(f"LONG! - Date&Time: {X_test['utc'].iloc[i]} - I bought Yandex for {open_now} and sold for {close_in_ten_min} + commission {commission_now} -> budget: {money}")
    #         elif y_pred == 0 and short == True:
    #             commission_now = ((open_now + close_in_ten_min) * commision) * (money // close_in_ten_min)
    #             money += (open_now - close_in_ten_min) * (money  // open_now) - commission_now
    #             logging.info(f"SHORT! - Date&Time: {X_test['utc'].iloc[i]} - I bought Yandex for {close_in_ten_min} and sold for {open_now} + commission {commission_now} -> budget: {money}")

    #     logging.info(f"\n\n\nMy budget before {initial_budget} and after trading {money}\nMommy, are you prod of me?")

    #     return money - initial_budget,  self.model.score(X_test, y_test) # ны выходе прибыль