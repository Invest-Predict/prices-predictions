import sklearn
from sklearn.preprocessing import StandardScaler, PowerTransformer
from enum import Enum
import datetime as dt
from multiprocessing import Pool
from os import cpu_count
from statsmodels.tsa.filters.hp_filter import hpfilter
from functools import partial
import pandas as pd

# Фильтрация, разделение на тренд и цикл 

def process_row_hp(i, data, lamb):
    trend_open, cycle_open = hpfilter(data["open"][:i + 2], lamb=lamb)
    print(i)
    return list(trend_open)[-2], list(cycle_open)[-2]


def train_data_hp_filter(data, l=10, num_pr=None):
    data_copy = data.copy()
    if num_pr is None:
        num_pr = cpu_count()  # Количество процессов по умолчанию

    length = data_copy.shape[0]

    # Создаем частично применённую функцию с фиксированными аргументами data и l
    process_row = partial(process_row_hp, data=data_copy, lamb=l)

    # Создаем пул процессов
    with Pool(num_pr) as pool:
        results = pool.map(process_row, range(length))

    # Разворачиваем список кортежей в два отдельных списка
    trend, cycle = zip(*results)

    # Добавляем рассчитанные значения в DataFrame
    data_copy["trend_open"] = trend
    data_copy["cycle_open"] = cycle

    return data_copy

class Scaler(Enum):
    """
    Перечисление для выбора метода скейлинга данных.
    """
    Standard = StandardScaler
    Power = PowerTransformer

def scale_num_data(fit_data, tranform_data, numeric, scaler : Scaler):
    scale_model = scaler.value()
    fitted_scaler = scale_model.fit(fit_data[numeric])
    res = []
    for df in tranform_data:
        transformed_data = df.copy()
        transformed_data[numeric] = fitted_scaler.transform(transformed_data[numeric])
        res.append(transformed_data)
    return res

# def merged_split(data, 
#                  start_data, 
#                  num_train_candles, 
#                  num_valid_candles, 
#                  target,
#                  numeric = [], cat = [],
#                  n_periods = 5, 
#                  num_test_candles=0):
    
#     train_indexes = list(range(num_train_candles))
#     valid_indexes = list(range(num_train_candles, num_train_candles + num_valid_candles))
#     train_ind = num_train_candles
#     # valid_ind = num_valid_candles
#     for i in range(1, n_periods):
#         train_ind += num_valid_candles
#         train_indexes += list(range(train_ind, train_ind + num_train_candles))
#         train_ind += num_train_candles
#         valid_indexes += list(range(train_ind, train_ind + num_valid_candles))

#     restr_data = (data[data['utc'] >= start_data]).reset_index()
#     X_train = restr_data[numeric + cat].iloc[train_indexes]
#     y_train = restr_data[target].iloc[train_indexes]
#     X_val = restr_data[numeric + cat].iloc[valid_indexes]
#     y_val = restr_data[target].iloc[valid_indexes]
#     return X_train, X_val, y_train, y_val 

def merged_split(data, 
                 start_data, 
                 num_train_candles, 
                 num_valid_candles, 
                 target,
                 numeric = [], cat = [],
                 n_periods = 5, 
                 num_test_candles=0):
    
    train_indexes = list(range(num_train_candles))
    train_ind = num_train_candles
    valid_indexes = list(range(train_ind, train_ind + num_valid_candles))
    train_ind += num_valid_candles
    test_indexes = list(range(train_ind, train_ind + num_test_candles))
    train_ind += num_test_candles
    for _ in range(1, n_periods):
        train_indexes += list(range(train_ind, train_ind + num_train_candles))
        train_ind += num_train_candles
        valid_indexes += list(range(train_ind, train_ind + num_valid_candles))
        train_ind += num_valid_candles
        test_indexes += list(range(train_ind, train_ind + num_test_candles))
        train_ind += num_test_candles

    restr_data = (data[data['utc'] >= start_data]).reset_index()
    # print(restr_data)
    X_train = restr_data[numeric + cat].iloc[train_indexes]
    y_train = restr_data[target].iloc[train_indexes]
    X_val = restr_data[numeric + cat].iloc[valid_indexes]
    y_val = restr_data[target].iloc[valid_indexes]
    if num_train_candles == 0:
        return X_train, X_val, y_train, y_val 
    else: 
        X_test = restr_data[numeric + cat].iloc[test_indexes]
        y_test = restr_data[target].iloc[test_indexes]
        return X_train, X_val, X_test, y_train, y_val, y_test

# Трейн-валид разделение 
def train_valid_split(data, 
                      year, month, day, 
                      numeric, cat, target, 
                      utc = []): # utc здесь добавлено для optuna 
    # возвращает тестовую и валидационную выборки в завимости от заданного времени
    """
    Делит данные на обучающую и валидационную выборки в зависимости от указанной даты.

    Параметры:
        data (pd.DataFrame): Исходные данные.
        year (int): Год разделения.
        month (int): Месяц разделения.
        day (int): День разделения.
        numeric (list): Список числовых признаков.
        cat (list): Список категориальных признаков.
        target (str): Целевой столбец.
        utc (list, optional): Дополнительные временные признаки.

    Возвращает:
        tuple: Обучающие и валидационные признаки и целевые значения.
    """    
    train_df = data[data["utc"] < dt.datetime(year, month, day)]

    X_train = train_df[numeric + cat + utc]
    y_train = train_df[target]

    test_df = data[data["utc"] >=  dt.datetime(year, month, day)]

    X_val = test_df[numeric + cat + utc]
    y_val = test_df[target]

    return X_train, X_val, y_train, y_val

def train_valid_split_stupidly(data,  
                               target, last_days = 2): # utc здесь добавлено для optuna 
        
    # возвращает тестовую и валидационную выборки в завимости от заданного времени

    split_date = data["utc"][-1] - pd.DateOffset(days=last_days)
    
    train_df = data[data["utc"] < split_date]

    X_train = train_df.drop(columns=target)
    y_train = train_df[target]

    test_df = data[data["utc"] >= split_date]

    X_val = test_df.drop(columns=target)
    y_val = test_df[target]

    return X_train, X_val, y_train, y_val


def train_valid_test_split(data, 
                      test_start_data : dt.datetime, # с точностью до минут указываем начало тестового периода 
                      numeric, cat, target, 
                      utc = [], test_ticks = 10, val_ticks = 200): # utc здесь добавлено для optuna 
    
    """
    Делит данные на обучающую, валидационную и тестовую выборки.

    Параметры:
        data (pd.DataFrame): Исходные данные.
        test_start_data (datetime): Начало тестового периода.
        numeric (list): Список числовых признаков.
        cat (list): Список категориальных признаков.
        target (str): Целевой столбец.
        utc (list, optional): Дополнительные временные признаки.
        test_ticks (int, optional): Количество записей для тестовой выборки.
        val_ticks (int, optional): Количество записей для валидационной выборки.

    Возвращает:
        tuple: Обучающие, валидационные и тестовые признаки и целевые значения.
    """
    train_valid_df = data[data["utc"] < test_start_data]
    train_df = train_valid_df[:train_valid_df.shape[0] - val_ticks]
    valid_df = train_valid_df[train_valid_df.shape[0] - val_ticks : ]
    X_train = train_df[numeric + cat + utc]
    y_train = train_df[target]
    X_val = valid_df[numeric + cat + utc]
    y_val = valid_df[target]

    test_df = data[data["utc"] >=  test_start_data][:test_ticks]
    X_test = test_df[numeric + cat + utc]
    y_test = test_df[target]

    return X_train, X_val, X_test, y_train, y_val, y_test