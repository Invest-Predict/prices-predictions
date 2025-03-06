from typing import Iterable
from model import FinData, CatboostFinModel, mul_PCA
from catboost import CatBoostClassifier
import datetime as dt
import pandas as pd
import numpy as np
from multiprocessing import Pool
from joblib import Parallel, delayed
from sklearn import clone
from os import cpu_count
from sklearn.preprocessing import StandardScaler
from backtests.logger_config import setup_logger
import logging


# def resample_last_batch(df, batch_size):
#     # это просто мне нужно было чтобы добавить наблюдения еще раз
#     # типо продублировать в выборке последний день, она здесь просто потому что я тут реально намусорила
#     regular_part = df.iloc[:-batch_size]  
#     last_batch = df.iloc[-batch_size:]   
#     resampled_last_batch = pd.concat([last_batch] * 2, ignore_index=True)  # Дублируем их
#     df_resampled = pd.concat([regular_part, resampled_last_batch], ignore_index=True)
#     return df_resampled

# self.logger = setup_logger("baktests_logger", "../logs/backtests.log", level=logging.INFO)

class Backtest():
    def __init__(self, strategies: list[str], args: dict, dfs: dict[str, any], features: list[list[str]], comissions : list[int], timedelta: str = '10min', target: str = 'direction_binary'):
        self._strategies = strategies  # ['long', 'short']
        self._args = args  # usual argumets for CatboostFinModel
        self._dfs = dfs  # dict {stock_name1: datasest1, stock_name2: dataset2, ...}  (Already with needed features)
        self._comission = comissions  # list of comissions, usually [0.0004]
        self._timedelta = timedelta  # the period of one candle. For example 10min, 1H and etc
        self._target = target
        self.X_train, self.X_val, self.X_test = None,None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.cat, self.num = features[0], features[1]
        self.features = None
        self.logger = setup_logger("baktests_logger", "../logs/backtests.log", level=logging.INFO)

    def set_features(self, features : dict):
        self.features = features

    
    def custom_datasets(self, df_path, start_dt, end_dt, train_size, val_size, test_size = None, use_PCA=False, return_split = False, ind = 0):
        data = FinData(df_path, indifference=ind)
        # timedelta чтобы не было нанов в признаках, потом оно еще раз режется после генерации
        data.restrict_time_down(start_dt - dt.timedelta(weeks=4))

        if self._timedelta != '10min':
            data.merge_candles(self._timedelta)
            data.make_binary_class_target(target_name=self._target)
        
        # TODO добавить сюда возможность кастомно добавлять фичи
        stock = df_path.split('/')[-1][:-11]
        # добавила новую свою фичу
        small_df = "../../datasets/" + stock + "_1_min.csv"
        data.insert_all(features_settings=self.features)
        self.cat = data.get_cat_features()
        self.num = data.get_numeric_features()
        data.restrict_time_down(start_dt)
        
        data.df = data.df[(data.df['utc'].dt.time >= dt.time(7, 0)) & (data.df['utc'].dt.time <= dt.time(21, 0))]


        if isinstance(train_size, dt.timedelta):  # Можно задавать границу сплита на train, val и test через timedelta: 10 days - train, 3 days - val
            train = data.df[data.df['utc'] <= start_dt + train_size]
            val = data.df[data.df['utc'] > start_dt + train_size][data.df['utc'] <= start_dt + train_size + val_size]

            # batch_size = 100
            # # Увеличиваем выборки
            # train = resample_last_batch(train, batch_size)
            # val = resample_last_batch(train, batch_size)


            if test_size is None:
                test = data.df[data.df['utc'] > start_dt + train_size + val_size][data.df['utc'] <= end_dt]
            else:
                test = data.df[data.df['utc'] > start_dt + train_size + val_size][data.df['utc'] <= start_dt + train_size + val_size + test_size]

            self.X_train, self.X_val, self.X_test = train.drop(columns=self._target), val.drop(columns=self._target), test.drop(columns=self._target)
            self.y_train, self.y_val, self.y_test = train[self._target], val[self._target], test[self._target]

            if use_PCA:
                close = self.X_test.close
                X_train, X_val, X_test, self.num = mul_PCA(X_train, X_val, X_test, n_comp="mle")
                X_test["close"] = close

            if return_split:
                return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

            return
        

        X, y = data.df.drop(columns=self._target), data.df[self._target]
        if test_size is not None:
            self.X_train, self.X_val, self.X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size: train_size + val_size + test_size]
            self.y_train, self.y_val, self.y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size: train_size + val_size + test_size]
        if start_dt_test is not None:
            test = data.df[data.df['utc'] >= start_dt_test][data.df['utc'] <= end_dt]
            self.X_test, self.y_test = test.drop(columns=self._target), test[self._target]

            X, y = data.df.drop(columns=self._target), data.df[self._target]
            self.X_train, self.X_val = X[:train_size], X[train_size:train_size + val_size]
            self.y_train, self.y_val = y[:train_size], y[train_size:train_size + val_size]

        if use_PCA:
            close = self.X_test.close
            utc_test = self.X_test.utc
            utc_train = self.X_train.utc
            utc_val = self.X_val.utc
            features = self.num + self.cat
            self.X_train, self.X_val, self.X_test = self.X_train[features], self.X_val[features], self.X_test[features]
            self.X_train, self.X_val, self.X_test, self.num = mul_PCA(self.X_train, self.X_val, self.X_test, n_comp="mle")
            self.X_test["close"] = close
            self.X_test["utc"] = utc_test
            self.X_train["utc"] = utc_train
            self.X_val["utc"] = utc_val

        if return_split:
            return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.num
    
    def another_train_val_test_split(self, df, train_size, val_size, test_size, start_dt_test):

        if isinstance(train_size, dt.timedelta):
            df_train = df[df['utc'] >= start_dt_test - val_size - train_size][df['utc'] < start_dt_test + val_size]
            df_val = df[df['utc'] >= start_dt_test - val_size][df['utc'] < start_dt_test]
            df_test = df[df['utc'] >= start_dt_test][df['utc'] < start_dt_test + test_size]
        else: # please, don't use it, let's use timedeltas instead =)
            df_train = df[df['utc'] < start_dt_test].iloc[-(train_size + val_size):val_size]
            df_val = df[df['utc'] < start_dt_test].iloc[-val_size:]
            df_test = df[df['utc'] >= start_dt_test].iloc[:test_size]
        
        X_train, y_train = df_train.drop(columns = self._target), df_train[self._target]
        X_val, y_val = df_val.drop(columns = self._target), df_val[self._target]
        X_test, y_test = df_test.drop(columns = self._target), df_test[self._target]
        return X_train, X_val, X_test, y_train, y_val, y_test

    
    def test_trading(self, budget, train_size, val_size, test_size, start_dt_test, end_dt_test, proba_limit = 0.5, use_already_fitted_model = False):

        results = []

        for stock, df in self._dfs.items():
            rounds = 0
            history = pd.DataFrame(columns=["datetime", "budget"])
            history.loc[0] = [start_dt_test, budget]
            money = budget
            itr = 0  # for the history
            corner_dt = start_dt_test
            while corner_dt <= end_dt_test:
                rounds += 1
                corner_dt += test_size
                X_train, X_val, X_test, y_train, y_val, y_test = self.another_train_val_test_split(df, train_size, val_size, test_size, corner_dt)
                logging.info(f"Backtesting started for stock - {stock} | round - {rounds}")
                logging.info(f"Train dates: {X_train['utc'].iloc[0]} - {X_train['utc'].iloc[-1]} | Valid dates: {X_val['utc'].iloc[0]} - {X_val['utc'].iloc[-1]} | Test dates: {X_test['utc'].iloc[0]} - {X_test['utc'].iloc[-1]}")

                if round == 1 or use_already_fitted_model == False:
                    X_train, X_val = X_train[self.num + self.cat], X_val[self.num + self.cat]
                    model = CatboostFinModel(self._args)
                    model.set_datasets(X_train, X_val, y_train, y_val)
                    model.set_features(self.num, self.cat)

                    model.fit()
                    logging.info(f"{model.get_top_imp_features(20)}")

                for i in range(X_test.shape[0] - 1):
                    itr += 1
                    # test = pd.DataFrame(scaler.transform(self.X_test[self.num + self.cat]))
                    y_preds = model.predict_proba(X_test[self.num + self.cat].iloc[i])
                    y_pred_1, y_pred_0 = y_preds[1], y_preds[0]
                    close_in_ten_min = X_test['close'].iloc[i + 1]
                    open_now = X_test['close'].iloc[i]

                    history.loc[itr] = [X_test['utc'].iloc[i + 1], money]

                    if money >= open_now and y_pred_1 > proba_limit and 'long' in self._strategies:
                        commission_now = ((open_now + close_in_ten_min) * self._comission[0]) * (money  // open_now)
                        money += (close_in_ten_min - open_now) * (money  // open_now) - commission_now

                        logging.info(f"LONG! - {stock}, Date&Time: {X_test['utc'].iloc[i]}, proba: {y_pred_1} - I bought for {open_now} and sold for {close_in_ten_min} + commission {commission_now} -> budget: {money}")
                    elif money >= close_in_ten_min and y_pred_0 > proba_limit and 'short' in self._strategies:
                        commission_now = ((open_now + close_in_ten_min) * self._comission[0]) * (money // close_in_ten_min)
                        money += (open_now - close_in_ten_min) * (money  // open_now) - commission_now
                    self.logger.info(f"LONG! - {stock}, Date&Time: {self.X_test['utc'].iloc[i]}, proba: {y_pred_1} - I bought for {open_now} and sold for {close_in_ten_min} + commission {commission_now} -> budget: {money}")
                elif money >= close_in_ten_min and y_pred_0 > proba_limit and 'short' in self._strategies:
                    commission_now = ((open_now + close_in_ten_min) * self._comission[0]) * (money // close_in_ten_min)
                    money += (open_now - close_in_ten_min) * (money  // open_now) - commission_now

                        logging.info(f"SHORT! - {stock}, Date&Time: {X_test['utc'].iloc[i]}, proba: {y_pred_0} - I bought for {close_in_ten_min} and sold for {open_now} + commission {commission_now} -> budget: {money}")
                logging.info(f"My budget on round - {rounds} before {budget} and after trading {money}\n")
                    self.logger.info(f"SHORT! - {stock}, Date&Time: {self.X_test['utc'].iloc[i]}, proba: {y_pred_0} - I bought for {close_in_ten_min} and sold for {open_now} + commission {commission_now} -> budget: {money}")

            logging.info(f"\n\n\nMy budget before {budget} and after trading {money}\nMommy, are you prod of me?")
            logging.info(model.score(X_test[self.num + self.cat], y_test))
            self.logger.info(f"\n\n\nMy budget before {budget} and after trading {money}\nMommy, are you prod of me?")
            self.logger.info(model.score(self.X_test[self.num + self.cat], self.y_test))

                # results.append((money - budget,  model.score(self.X_test, self.y_test))) # ны выходе прибыль (точнее список прибыли и accuracy)
            results.append(history)
        return results, money
    
    def process_company(self, custom_datasets_args, use_PCA, df_path):
        custom_datasets_args['df_path'] = df_path
        custom_datasets_args['return_split'] = True

        custom_datasets_args['ind'] = 0
        X_train, X_val, X_test, y_train, y_val, y_test, num = self.custom_datasets(**custom_datasets_args, use_PCA=use_PCA)
        stock = df_path.split('/')[-1][:-11]  
        self.logger.info(f"Training long model starterd for stock - {stock}")
        self.logger.info(f"Train dates: {X_train['utc'].iloc[0]} - {X_train['utc'].iloc[-1]} | Valid dates: {X_val['utc'].iloc[0]} - {X_val['utc'].iloc[-1]} | Test dates: {X_test['utc'].iloc[0]} - {X_test['utc'].iloc[-1]}")
        X_train, X_val = X_train[num], X_val[num]
        model_long = CatboostFinModel(self._args[0])
        model_long.set_datasets(X_train, X_val, y_train, y_val)
        model_long.set_features(num, [])
        model_long.fit()
        self.logger.info(f"Training long model ended with loss {model_long.model.best_score_}")

        custom_datasets_args['ind'] = 1
        X_train, X_val, X_test, y_train, y_val, y_test, num = self.custom_datasets(**custom_datasets_args, use_PCA=use_PCA)
        stock = df_path.split('/')[-1][:-11]  
        self.logger.info(f"Training short model starterd for stock - {stock}")
        self.logger.info(f"Train dates: {X_train['utc'].iloc[0]} - {X_train['utc'].iloc[-1]} | Valid dates: {X_val['utc'].iloc[0]} - {X_val['utc'].iloc[-1]} | Test dates: {X_test['utc'].iloc[0]} - {X_test['utc'].iloc[-1]}")
        X_train, X_val = X_train[num], X_val[num]

        model_short = CatboostFinModel(self._args[1])
        model_short.set_datasets(X_train, X_val, y_train, y_val)
        model_short.set_features(num, [])
        model_short.fit()
        self.logger.info(f"Training short model ended with loss {model_long.model.best_score_}")
        # models_short.append(model_short)
        return (X_test, y_test, stock, model_long, model_short, num)

    
    def test_multistock(self, budget, custom_datasets_args, proba_limit = 0.5, use_PCA = False, add_model : dict | None = None):
        models = []
        models_long = []
        models_short = []
        stocks = []
        X_tests, y_tests = [], []
        nums= []

        # if "long-short" in self._strategies:
        #     n_jobs = 2  # Использовать все доступные ядра
        #     results = Parallel(n_jobs=n_jobs)(
        #         delayed(self.process_company)(custom_datasets_args, use_PCA, data) 
        #         for data in self._dfs
        #     )


        # else:
        for df_path in self._dfs:  # здесь происходит сборка общего датасета. TODO подумать, а может обучаться надо на каком-то одном и передавать его как stock_train
            custom_datasets_args['df_path'] = df_path
            custom_datasets_args['return_split'] = True

            if "long-short" in self._strategies:
                custom_datasets_args['ind'] = 0
                X_train, X_val, X_test, y_train, y_val, y_test, num = self.custom_datasets(**custom_datasets_args, use_PCA=use_PCA)
                stock = df_path.split('/')[-1][:-11]  # так обрежеться всё до названия файла из datasets и тажке _10_min.csv
                self.logger.info(f"Training long model starterd for stock - {stock}")
                self.logger.info(f"Train dates: {self.X_train['utc'].iloc[0]} - {self.X_train['utc'].iloc[-1]} | Valid dates: {self.X_val['utc'].iloc[0]} - {self.X_val['utc'].iloc[-1]} | Test dates: {self.X_test['utc'].iloc[0]} - {self.X_test['utc'].iloc[-1]}")
                X_train, X_val = X_train[num + self.cat], X_val[num + self.cat]
                stocks.append(stock)
                X_tests.append(X_test)
                y_tests.append(y_test)
                nums.append(self.num)
                model_long = CatboostFinModel(self._args[0])
                model_long.set_datasets(X_train, X_val, y_train, y_val)
                model_long.set_features(num, self.cat)
                model_long.fit()
                self.logger.info(f"Training long model ended with loss {model_long.model.best_score_}")
                models_long.append(model_long)

                custom_datasets_args['ind'] = 1
                X_train, X_val, X_test, y_train, y_val, y_test, num = self.custom_datasets(**custom_datasets_args, use_PCA=use_PCA)
                stock = df_path.split('/')[-1][:-11]  # так обрежеться всё до названия файла из datasets и тажке _10_min.csv
                self.logger.info(f"Training short model starterd for stock - {stock}")
                self.logger.info(f"Train dates: {self.X_train['utc'].iloc[0]} - {self.X_train['utc'].iloc[-1]} | Valid dates: {self.X_val['utc'].iloc[0]} - {self.X_val['utc'].iloc[-1]} | Test dates: {self.X_test['utc'].iloc[0]} - {self.X_test['utc'].iloc[-1]}")
                X_train, X_val = X_train[num + self.cat], X_val[num + self.cat]

                model_short = CatboostFinModel(self._args[1])
                model_short.set_datasets(X_train, X_val, y_train, y_val)
                model_short.set_features(num, self.cat)
                model_short.fit()
                self.logger.info(f"Training short model ended with loss {model_short.model.best_score_}")
                models_short.append(model_short)

            else:
                self.custom_datasets(**custom_datasets_args, use_PCA=use_PCA)

                X_train, X_val, X_test, y_train, y_val, y_test = self.custom_datasets(**custom_datasets_args, use_PCA=use_PCA)
                stock = df_path.split('/')[-1][:-11]  # так обрежеться всё до названия файла из datasets и тажке _10_min.csv

                self.logger.info(f"Training model starterd for stock - {stock}")
                self.logger.info(f"Train dates: {self.X_train['utc'].iloc[0]} - {self.X_train['utc'].iloc[-1]} | Valid dates: {self.X_val['utc'].iloc[0]} - {self.X_val['utc'].iloc[-1]} | Test dates: {self.X_test['utc'].iloc[0]} - {self.X_test['utc'].iloc[-1]}")
                X_train, X_val = X_train[self.num + self.cat], X_val[self.num + self.cat]
                nums.append(self.num)

                stocks.append(stock)

                X_tests.append(X_test)
                y_tests.append(y_test)

                model = CatboostFinModel(self._args)
                model.set_datasets(X_train, X_val, y_train, y_val)
                model.set_features(self.num, self.cat)

                model.fit()

                models.append(model)
                

        history = []

        money = budget
        history.append(money)
        n = X_tests[0][0].shape[0] - 1
        for i in range(n):
            y_probs_0, y_probs_1 = [], []
            info = []
            if "long-short" in self._strategies:
                for X_test, y_test, stock, model_long, model_short, num in zip(X_tests, y_tests, stocks, model_long, models_short, nums):
                    y_pred_1 = model_long.predict_proba(X_test[num].iloc[i])[1]
                    y_pred_0 = model_short.predict_proba(X_test[num].iloc[i])[0]
                    close_in_ten_min = X_test['close'].iloc[i + 1]
                    open_now = X_test['close'].iloc[i]

                    y_probs_0.append(y_pred_0)
                    y_probs_1.append(y_pred_1)
                    info.append((close_in_ten_min, open_now, stock, X_test['utc'].iloc[i]))

                ind_0, ind_1 = np.argsort(y_probs_0)[::-1], np.argsort(y_probs_1)[::-1]
                
                # проходимся по топ-1

                if y_probs_1[ind_1[0]] >= proba_limit and y_probs_0[ind_1[0]] < 0.5:
                    # если лонг предсказывает повышение и шорт не предсказывает на этой компании понижения
                    if y_probs_0[ind_0[0]] <= proba_limit:
                        # если шорта нет 
                        # лонг на все деньги 
                        open_now = info[ind_1[0]][1]
                        close_in_ten_min = info[ind_1[0]][0]
                        stock = info[ind_1[0]][2]
                        commission_now = ((open_now + close_in_ten_min) * self._comission[0]) * (money  // open_now)
                        money += (close_in_ten_min - open_now) * (money  // open_now) - commission_now
                        self.logger.info(f"LONG! - stock: {stock} with proba 1 : {y_probs_1[ind_1[0]]} - Date&Time: {info[ind_1[0]][3]} - "
                                     f"I bought for {open_now} and sold for {close_in_ten_min} + commission {commission_now} -> budget: {money}")

                    elif y_probs_0[ind_0[0]] > proba_limit and y_probs_1[ind_0[0]] < 0.5:
                        # есть норм шорт, делим деньги между шортом и лонгом 
                        money_for_both = money / 2
                        # лонг 
                        open_now = info[ind_1[0]][1]
                        close_in_ten_min = info[ind_1[0]][0]
                        stock = info[ind_1[0]][2]
                        commission_now = ((open_now + close_in_ten_min) * self._comission[0]) * (money_for_both  // open_now)
                        money += (close_in_ten_min - open_now) * (money_for_both  // open_now) - commission_now
                        self.logger.info(f"LONG! - stock: {stock} with proba 1 : {y_probs_1[ind_1[0]]} - Date&Time: {info[ind_1[0]][3]} - "
                                     f"I bought for {open_now} and sold for {close_in_ten_min} + commission {commission_now} -> budget: {money}")
                        # шорт 
                        open_now = info[ind_0[0]][1]
                        close_in_ten_min = info[ind_0[0]][0]
                        stock = info[ind_0[0]][2]
                        commission_now = ((open_now + close_in_ten_min) * self._comission[0]) * (money_for_both // close_in_ten_min)
                        money += (open_now - close_in_ten_min) * (money_for_both  // open_now) - commission_now
                        self.logger.info(f"SHORT! - stock: {stock} with proba 0 : {y_probs_0[ind_0[0]]} - Date&Time: {info[ind_0[0]][3]} - "
                                     f"I bought for {close_in_ten_min} and sold for {open_now} + commission {commission_now} -> budget: {money}")
                        
                elif y_probs_0[ind_0[0]] > proba_limit and y_probs_1[ind_0[0]] < 0.5:
                    # только шорт 
                    open_now = info[ind_0[0]][1]
                    close_in_ten_min = info[ind_0[0]][0]
                    stock = info[ind_0[0]][2]
                    commission_now = ((open_now + close_in_ten_min) * self._comission[0]) * (money // close_in_ten_min)
                    money += (open_now - close_in_ten_min) * (money  // open_now) - commission_now
                    self.logger.info(f"SHORT! - stock: {stock} with proba 0 : {y_probs_0[ind_0[0]]} - Date&Time: {info[ind_0[0]][3]} - "
                                 f"I bought for {close_in_ten_min} and sold for {open_now} + commission {commission_now} -> budget: {money}")
                    
                # проходимся по топ-2 тут не if а elif, это важно, то есть я пока что просто страхуюсь на случай если мы не вошли в первый
                elif y_probs_1[ind_1[1]] >= proba_limit and y_probs_0[ind_1[1]] < 0.5:
                    # если лонг предсказывает повышение и шорт не предсказывает на этой компании понижения
                    if y_probs_0[ind_0[1]] <= proba_limit:
                        # если шорта нет 
                        # лонг на все деньги 
                        open_now = info[ind_1[1]][1]
                        close_in_ten_min = info[ind_1[1]][0]
                        stock = info[ind_1[1]][2]
                        commission_now = ((open_now + close_in_ten_min) * self._comission[0]) * (money  // open_now)
                        money += (close_in_ten_min - open_now) * (money  // open_now) - commission_now
                        self.logger.info(f"LONG! - stock: {stock} with proba 1 : {y_probs_1[ind_1[1]]} - Date&Time: {info[ind_1[1]][3]} - "
                                     f"I bought for {open_now} and sold for {close_in_ten_min} + commission {commission_now} -> budget: {money}")

                    elif y_probs_0[ind_0[1]] > proba_limit and y_probs_1[ind_0[1]] < 0.5:
                        # есть норм шорт, делим деньги между шортом и лонгом 
                        money_for_both = money / 2
                        # лонг 
                        open_now = info[[1]][1]
                        close_in_ten_min = info[ind_1[1]][0]
                        stock = info[ind_1[1]][2]
                        commission_now = ((open_now + close_in_ten_min) * self._comission[0]) * (money_for_both  // open_now)
                        money += (close_in_ten_min - open_now) * (money_for_both  // open_now) - commission_now
                        self.logger.info(f"LONG! - stock: {stock} with proba 1 : {y_probs_1[ind_1[1]]} - Date&Time: {info[ind_1[1]][3]} - "
                                     f"I bought for {open_now} and sold for {close_in_ten_min} + commission {commission_now} -> budget: {money}")
                        # шорт 
                        open_now = info[ind_0[1]][1]
                        close_in_ten_min = info[ind_0[1]][0]
                        stock = info[ind_0[1]][2]
                        commission_now = ((open_now + close_in_ten_min) * self._comission[0]) * (money_for_both // close_in_ten_min)
                        money += (open_now - close_in_ten_min) * (money_for_both  // open_now) - commission_now
                        self.logger.info(f"SHORT! - stock: {stock} with proba 0 : {y_probs_0[ind_0[1]]} - Date&Time: {info[ind_0[1]][3]} - "
                                     f"I bought for {close_in_ten_min} and sold for {open_now} + commission {commission_now} -> budget: {money}")
                        
                elif y_probs_0[ind_0[1]] > proba_limit and y_probs_1[ind_0[1]] < 0.5:
                    # только шорт 
                    open_now = info[ind_0[1]][1]
                    close_in_ten_min = info[ind_0[1]][0]
                    stock = info[ind_0[1]][2]
                    commission_now = ((open_now + close_in_ten_min) * self._comission[0]) * (money // close_in_ten_min)
                    money += (open_now - close_in_ten_min) * (money  // open_now) - commission_now
                    self.logger.info(f"SHORT! - stock: {stock} with proba 0 : {y_probs_0[ind_0[1]]} - Date&Time: {info[ind_0[1]][3]} - "
                                 f"I bought for {close_in_ten_min} and sold for {open_now} + commission {commission_now} -> budget: {money}")

            else:
                for X_test, y_test, stock, model, num in zip(X_tests, y_tests, stocks, models, nums):  # TODO - научиться проверять, что по времени (utc) все совпадают
                    if X_test.shape[0] > i + 1:
                        y_pred = model.predict_proba(X_test[num + self.cat].iloc[i])
                        close_in_ten_min = X_test['close'].iloc[i + 1]
                        open_now = X_test['close'].iloc[i]
                        y_probs_0.append((y_pred[0], close_in_ten_min, open_now, stock, X_test['utc'].iloc[i]))
                        y_probs_1.append((y_pred[1], close_in_ten_min, open_now, stock, X_test['utc'].iloc[i]))

                        # model.model.refit(X_test.iloc[i], np.array([y_test.iloc[i]]))
                                    
                y_probs_0.sort(reverse=True)
                y_probs_1.sort(reverse=True)

                # self.logger.info(f"All probs 0: {[e[0] for e in y_probs_0]} and stocks : {[e[3] for e in y_probs_0]}")
                # self.logger.info(f"All probs 1: {[e[0] for e in y_probs_1]} and stocks : {[e[3] for e in y_probs_1]}")

                if 'long' in self._strategies and y_probs_1[0][0] >= proba_limit:  #TODO - 1) Сделать возможность покупать сразу несколько акций 2) Поставить заглушку на min prob
                    open_now = y_probs_1[0][2]
                    close_in_ten_min = y_probs_1[0][1]
                    stock = y_probs_1[0][3]
                    commission_now = ((open_now + close_in_ten_min) * self._comission[0]) * (money  // open_now)
                    money += (close_in_ten_min - open_now) * (money  // open_now) - commission_now

                    self.logger.info(f"LONG! - stock: {stock} with proba 1 : {y_probs_1[0][0]} - Date&Time: {y_probs_1[0][4]} - I bought for {open_now} and sold for {close_in_ten_min} + commission {commission_now} -> budget: {money}")

                if 'short' in self._strategies and y_probs_0[0][0] >= proba_limit:
                    open_now = y_probs_0[0][2]
                    close_in_ten_min = y_probs_0[0][1]
                    stock = y_probs_0[0][3]
                    commission_now = ((open_now + close_in_ten_min) * self._comission[0]) * (money // close_in_ten_min)
                    money += (open_now - close_in_ten_min) * (money  // open_now) - commission_now
                    self.logger.info(f"SHORT! - stock: {stock} with proba 0 : {y_probs_0[0][0]} - Date&Time: {y_probs_0[0][4]} - I bought for {close_in_ten_min} and sold for {open_now} + commission {commission_now} -> budget: {money}")
                
            history.append(money)
        
        logging.info(f"\n\n\nMy budget before {budget} and after trading {money}\nMommy, are you prod of me?")