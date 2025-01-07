import sklearn
from sklearn.preprocessing import StandardScaler, PowerTransformer
from enum import Enum
import datetime as dt
from multiprocessing import Pool
from os import cpu_count
from statsmodels.tsa.filters.hp_filter import hpfilter
from functools import partial

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
    Standart = StandardScaler()
    Power = PowerTransformer()

def scale_num_data(data, numeric, scaler : Scaler):
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
        
        train_df = data[data["utc"] < dt.datetime(year, month, day)]

        X_train = train_df[numeric + cat + utc]
        y_train = train_df[target]

        test_df = data[data["utc"] >=  dt.datetime(year, month, day)]

        X_val = test_df[numeric + cat + utc]
        y_val = test_df[target]

        return X_train, X_val, y_train, y_val
