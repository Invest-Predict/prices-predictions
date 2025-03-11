import sklearn
from sklearn.preprocessing import StandardScaler
from enum import Enum
import datetime as dt
from multiprocessing import Pool
from os import cpu_count
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.decomposition import PCA
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

def merged_split(data, 
                 start_data, 
                 num_train_candles, 
                 num_valid_candles, 
                 target,
                 numeric = [], cat = [],
                 n_periods = 5, 
                 num_test_candles=0):
    
    train_indexes = list(range(num_train_candles))
    ptr_ind = num_train_candles
    valid_indexes = list(range(ptr_ind, ptr_ind + num_valid_candles))
    ptr_ind += num_valid_candles
    test_indexes = list(range(ptr_ind, ptr_ind + num_test_candles))
    ptr_ind += num_test_candles

    for _ in range(1, n_periods):
        train_indexes += list(range(ptr_ind, ptr_ind + num_train_candles))
        ptr_ind += num_train_candles
        valid_indexes += list(range(ptr_ind, ptr_ind + num_valid_candles))
        ptr_ind += num_valid_candles
        test_indexes += list(range(ptr_ind, ptr_ind + num_test_candles))
        ptr_ind += num_test_candles

    restr_data = (data[data['utc'] >= start_data]).reset_index()
    X_train = restr_data[numeric + cat].iloc[train_indexes]
    y_train = restr_data[target].iloc[train_indexes]
    X_val = restr_data[numeric + cat].iloc[valid_indexes]
    y_val = restr_data[target].iloc[valid_indexes]

    if num_test_candles == 0:
        return X_train, X_val, y_train, y_val 
    
    else: 
        X_test = restr_data[numeric + cat].iloc[test_indexes]
        y_test = restr_data[target].iloc[test_indexes]
        return X_train, X_val, X_test, y_train, y_val, y_test
    
def scale_data(X_train, X_val, X_test, num_features, scaler = StandardScaler()):
    """
    Масштабирует числовые признаки с помощью переданного скейлера.
    
    :param X_train: Данные для обучения (DataFrame или массив)
    :param X_val: Данные для валидации (DataFrame или массив)
    :param X_test: Данные для тестирования (DataFrame или массив)
    :param num_features: Список имен числовых признаков
    :param scaler: Объект скейлера (например, StandardScaler, MinMaxScaler и т. д.)
    :return: Кортеж (X_train_scaled, X_val_scaled, X_test_scaled, fitted_scaler)
    """
    scaler = scaler.fit(X_train[num_features])

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[num_features] = scaler.transform(X_train[num_features])
    X_val_scaled[num_features] = scaler.transform(X_val[num_features])
    X_test_scaled[num_features] = scaler.transform(X_test[num_features])

    return X_train_scaled, X_val_scaled, X_test_scaled


def mul_PCA(X_train, X_val, X_test, n_comp = "mle"):
    # Нужно подавать центрированные данные по хорошему)
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_pca, X_val_pca, X_test_pca = pca.fit_transform(X_train), pca.transform(X_val), pca.transform(X_test)
    numeric_features = [f'PC{i+1}' for i in range(pca.n_components_)]
    
    X_train_pca_df = pd.DataFrame(X_train_pca, index=X_train.index, columns=numeric_features)
    X_val_pca_df = pd.DataFrame(X_val_pca, index=X_val.index, columns=numeric_features)
    X_test_pca_df = pd.DataFrame(X_test_pca, index=X_test.index, columns=numeric_features)
    
    return X_train_pca_df, X_val_pca_df, X_test_pca_df, numeric_features

    
def train_valid_split_candles(data, train_size, val_size, numeric, cat, target, silenced=True):
    # ДЛЯ ТРЕЙДИНГА
    train_df = data[-(train_size + val_size) : -val_size]
    val_df = data[-val_size : ]

    if not silenced:
        train_sd, val_sd = train_df["utc"].iloc[0], val_df["utc"].iloc[0]
        train_ed, val_ed = train_df["utc"].iloc[-1], val_df["utc"].iloc[-1]
        print(f"Начало тренировочного периода: {train_sd}. Конец тренировочного периода: {train_ed} \n \
                Начало валидационного периода: {val_sd}. Конец валидационного периода: {val_ed}")
    
    X_train, y_train = train_df[numeric + cat], train_df[target]
    X_val, y_val = val_df[numeric + cat], val_df[target]

    return X_train, X_val, y_train, y_val


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



def train_valid_test_split(data, start_period : dt.datetime, train_size, val_size, test_size, numeric, cat, target, silenced = True, 
                           need_test = False):
        
        cutted_df = data[data["utc"] >= start_period]
        train_df = cutted_df[:train_size]
        val_df = cutted_df[train_size : train_size+val_size]
        test_df = cutted_df[train_size+val_size : train_size+val_size+test_size]

        if not silenced:
            train_sd, val_sd, test_sd = train_df["utc"].iloc[0], val_df["utc"].iloc[0], test_df["utc"].iloc[0]
            train_ed, val_ed, test_ed = train_df["utc"].iloc[-1], val_df["utc"].iloc[-1], test_df["utc"].iloc[-1]
            print(f"Начало тренировочного периода: {train_sd}. Конец тренировочного периода: {train_ed} \n \
                    Начало валидационного периода: {val_sd}. Конец валидационного периода: {val_ed} \n \
                    Начало тестового периода: {test_sd}. Конец тестового периода: {test_ed} \n ")
        
        X_train, y_train = train_df[numeric + cat], train_df[target]
        X_val, y_val = val_df[numeric + cat], val_df[target]
        X_test, y_test = test_df[numeric + cat], test_df[target]

        return (X_train, X_val, X_test, y_train, y_val, y_test) if not need_test else (X_train, X_val, X_test, y_train, y_val, y_test, test_df)
