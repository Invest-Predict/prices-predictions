from .preprocessing import train_valid_test_split
# from utils import make_features, calculate_avret, append_tests_data #!!!!
import datetime as dt
import numpy as np

import logging
import pandas as pd


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("testing.log"),
        logging.StreamHandler()
    ]
)

# ------------------------------
# Test for model

def test_trading(finmodel, df, target = 'direction_binary', start_date = None, end_date = None, proportion = [1, 1, 1],
                    test_st_dt = None, test_end_dt = None, 
                    initial_budget = 10000, cat = [], num = [], print_actions = False, commision = 0.0001):
    '''
    Примитиваня стратегия, пусть мы просто пока покупаем акцию сейчас, если предполагаем, что через десять минут она вырастит в цене
    (через 10 минут в этом случае её продаём)
    В ином случае мы ничего не делаем (ждём следующий период)
    Но также у нас есть ограничение - это бюджет (он ограчен => не всегда сможем купить акцию, чтобы продать её через 10 минут)
    '''

    train_size, val_size, test_size = 1000, 180, 180
    if end_date is not None:
        df = df[df["utc"] <= end_date].reset_index().drop(columns=['index'])
    if start_date is not None:
            df = df[df["utc"] >= start_date].reset_index().drop(columns=['index'])
            df_size = df.shape[0]
            train_size, val_size = int(df_size * (proportion[0] / sum(proportion))), int(df_size * (proportion[1] / sum(proportion)))
            test_size = df_size - train_size - val_size

    X, y = df.drop(columns=target), df[target]

    X_train, X_val, X_test = X[-(train_size + val_size + test_size):-(val_size + test_size)], X[-(val_size + test_size): -test_size], X[-test_size:]
    X_train, X_val = X_train[num + cat], X_val[num + cat]
    y_train, y_val, y_test = y[-(train_size + val_size + test_size):-(val_size + test_size)], y[-(val_size + test_size): -test_size], y[-test_size:]

    if test_st_dt is not None:
        test_df = df = df[df["utc"] <= test_end_dt][df["utc"] >= test_st_dt].reset_index().drop(columns=['index'])
        X_test, y_test = test_df.drop(columns=target), test_df[target]

    finmodel.set_datasets(X_train, X_val, y_train, y_val)
    finmodel.set_features(num, cat)

    finmodel.fit()

    history = pd.DataFrame(columns=["datetime", "budget"])
    history.loc[0] = [X_test['utc'].iloc[0], initial_budget]
    money = initial_budget

    logging.info("Backtesting started")
    logging.info(f"Train dates: {X_train['utc'].iloc[0]} - {X_train['utc'].iloc[-1]} | Valid dates: {X_val['utc'].iloc[0]} - {X_val['utc'].iloc[-1]} | Test dates: {X_test['utc'].iloc[0]} - {X_test['utc'].iloc[-1]}")

    for i in range(X_test.shape[0] - 1):
        y_pred = finmodel.predict(X_test[num + cat].iloc[i])
        close_in_ten_min = X_test['close'].iloc[i + 1]
        open_now = X_test['close'].iloc[i]

        history.loc[i + 1] = [X_test['utc'].iloc[i + 1], money]

        if money >= open_now and y_pred == 1:
            money += (close_in_ten_min - open_now - (open_now + close_in_ten_min) * commision) * (money  // open_now) # продали за цену open_now и купили через 10 мин за close_in_ten_min

            logging.info(f"LONG! - Date&Time: {X_test['utc'].iloc[i]} - I bought Yandex for {open_now} and sold for {close_in_ten_min} -> budget: {money}")

            # if print_actions:
            #     s_add = ""
            #     if close_in_ten_min < open_now:
            #         s_add = " Daaaaaaaaaamn I was wrong"
            #     print(f"Date&Time: {X_test['utc'].iloc[i]} - I bought Yandex for {open_now} and sold for {close_in_ten_min} -> budget: {money}" + s_add)
        
        elif y_pred == 0:
            money += (open_now - close_in_ten_min - (open_now + close_in_ten_min) * commision) * (money  // open_now)  # купили сейчас за текущую цену open_now и продали через 10 мин за close_in_ten_min

            logging.info(f"SHORT! - Date&Time: {X_test['utc'].iloc[i]} - I bought Yandex for {open_now} and sold for {close_in_ten_min} -> budget: {money}")


            # if print_actions:
            #     s_add = ""
            #     if close_in_ten_min < open_now:
            #         s_add = " Daaaaaaaaaamn I was wrong"
            #     print(f"Date&Time: {X_test['utc'].iloc[i]} - I bought Yandex for {open_now} and sold for {close_in_ten_min} -> budget: {money}" + s_add)

        if y_pred == 0:
            money += (open_now - close_in_ten_min) * (money // open_now) # продали за цену open_now и купили через 10 минут за close_in_ten_min
                    
    print(f"My budget before {initial_budget} and after trading {money}\nMommy, are you prod of me?")
    logging.info(f"\n\n\nMy budget before {initial_budget} and after trading {money}\nMommy, are you prod of me?")

    return history


def test_weekly(finmodel, df, start_dt = dt.datetime(2024, 1, 1), end_dt=dt.datetime(2024, 12, 31), proportion = [15, 2, 3], target = 'direction_binary', cat = [], num = []):
    columns = df.columns
    # train, val, test = pd.DataFrame(columns=columns), pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)
    df_copy = df[df['utc'] >= start_dt][df['utc'] <= end_dt].copy()

    now_dt = df_copy['utc'].iloc[0]
    ind = 0

    train_ind, val_ind, test_ind = [], [], []
    while now_dt + dt.timedelta(days=sum(proportion)) < end_dt:
        next_dt = now_dt + dt.timedelta(days=proportion[0])  # for train
        while now_dt < next_dt:
            train_ind.append(ind)
            ind += 1
            now_dt = df_copy['utc'].iloc[ind]
        next_dt = now_dt + dt.timedelta(days=proportion[1])  # for val
        while now_dt < next_dt:
            val_ind.append(ind)
            ind += 1
            now_dt = df_copy['utc'].iloc[ind]
        
        next_dt = now_dt + dt.timedelta(days=proportion[2])  # for test
        while now_dt < next_dt:
            test_ind.append(ind)
            ind += 1
            now_dt = df_copy['utc'].iloc[ind]
    train = df_copy.iloc[train_ind]
    val = df_copy.iloc[val_ind]
    test = df_copy.iloc[test_ind]
    
    X_train, y_train = train.drop(columns=target), train[target]
    X_val, y_val = val.drop(columns=target), val[target]
    X_test, y_test = test.drop(columns=target), test[target]

    finmodel.set_datasets(X_train, X_val, y_train, y_val)
    finmodel.set_features(num, cat)

    finmodel.fit()

    finmodel.get_top_imp_features(20)

    # self.predict(X_test)

    return finmodel.model.score(X_test, y_test)


def test_intersect(finmodel, df, start_dt = dt.datetime(2024, 12, 1), end_dt=dt.datetime(2024, 12, 27), proportion = [15, 2, 3], target = 'direction_binary', cat = [], num = []):
    # train, val, test = pd.DataFrame(columns=columns), pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)
    df_copy = df[df['utc'] >= start_dt][df['utc'] <= end_dt].copy()

    # now_dt = df_copy['utc'].iloc[0]
    prev_dt = df_copy['utc'].iloc[0]
    prev_ind = 0
    ind = 0

    accuracy_sum = 0
    cnt = 0

    train_ind, val_ind, test_ind = [], [], []
    while prev_dt + dt.timedelta(days=sum(proportion)) <= end_dt:
        now_dt = prev_dt
        ind = prev_ind
        prev_dt += dt.timedelta(days=1)
        next_dt = now_dt + dt.timedelta(days=proportion[0])  # for train
        while now_dt < next_dt:
            train_ind.append(ind)
            ind += 1
            now_dt = df_copy['utc'].iloc[ind]
            if now_dt == prev_dt:
                prev_ind = ind
        next_dt = now_dt + dt.timedelta(days=proportion[1])  # for val
        while now_dt < next_dt:
            val_ind.append(ind)
            ind += 1
            now_dt = df_copy['utc'].iloc[ind]
        
        next_dt = now_dt + dt.timedelta(days=proportion[2])  # for test
        while now_dt < next_dt:
            test_ind.append(ind)
            ind += 1
            now_dt = df_copy['utc'].iloc[ind]

        train = df_copy.iloc[train_ind]
        val = df_copy.iloc[val_ind]
        test = df_copy.iloc[test_ind]
        
        X_train, y_train = train.drop(columns=target), train[target]
        X_val, y_val = val.drop(columns=target), val[target]
        X_test, y_test = test.drop(columns=target), test[target]

        finmodel.set_datasets(X_train, X_val, y_train, y_val)
        finmodel.set_features(num, cat)

        finmodel.fit()
        accuracy_sum += model.model.score(X_test, y_test)
        cnt += 1
    
    return accuracy_sum / cnt
