from typing import Iterable

class Backtest():
    def __init__(self, strategies: Iterable, models: Iterable, dfs: Iterable, comission : int):
        self._strategies = list(strategies)
        self._models = list(models)
        self._dfs = list(dfs)
        self._comission = comission

    @property
    def strategies(self) -> list:
        return self._strategies

    @strategies.setter
    def strategies(self, new_strategies: Iterable) -> None:
        self._strategies = list(new_strategies)

    @property
    def models(self) -> list:
        return self._models

    @models.setter
    def models(self, new_models: Iterable) -> None:
        self._models = list(new_models)

    @property
    def dfs(self) -> list:
        return self._dfs

    @dfs.setter
    def dfs(self, new_dfs: Iterable) -> None:
        self._dfs = list(new_dfs)

    @property
    def commision(self) -> int:
        return self._commision

    @commision.setter
    def dfs(self, new_dfs: Iterable) -> None:
        self._dfs = list(new_dfs)

    

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