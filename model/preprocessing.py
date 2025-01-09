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

# Функция для обработки одной строки
# Лерино я тут поменяю
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


# Скейлинг, по идее нельзя особо юзать даже 
class Scaler(Enum):
    """
    Перечисление для выбора метода скейлинга данных.
    """
    Standart = StandardScaler()
    Power = PowerTransformer()

def scale_num_data(data, numeric, scaler : Scaler):
    """
    Масштабирует числовые признаки данных с использованием заданного скейлера.

    Параметры:
        data (pd.DataFrame): Исходные данные.
        numeric (list): Список числовых признаков для масштабирования.
        scaler (Scaler): Выбранный метод скейлинга из перечисления Scaler.

    Возвращает:
        pd.DataFrame: Данные с масштабированными числовыми признаками.
    """
    scale_model = scaler.value
    transformed_data = data.copy()
    transformed_data[numeric] = scale_model.fit(transformed_data[numeric]).transform(transformed_data[numeric])
    return transformed_data


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
                               target, last_days = 2,
                               utc = []): # utc здесь добавлено для optuna 
        
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