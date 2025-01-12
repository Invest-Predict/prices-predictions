from catboost import CatBoostClassifier, Pool
import polars as pl
import pandas as pd
import numpy as np
import shap
import optuna 
import datetime as dt
from dateutil.relativedelta import relativedelta
from optuna.pruners import MedianPruner
from optuna.storages import RDBStorage
from optuna.integration import CatBoostPruningCallback
from .data import FinData
# from preprocessing import train_valid_split_stupidly
from sklearn.metrics import accuracy_score

# визуализация предсказаний модели и матрицы 
# подбирать время для предсказаний - optuna chose time

# кросс-валидация 

# куча признаков и выбираются самые важные 

# чуть-чуть Лера переделает 
def restrict_time_down(df, year=2024, month=9, day=11, date=None):
        # обрезает датасет по времени ОТ
        df_copy = df.copy()
        if date is not None:
            df_copy = df_copy[df_copy["utc"] >= date].reset_index().drop(columns=['index'])
            return df_copy
        df_copy = pl.from_pandas(df_copy.reset_index()).rename({"index" : "old_indexes"})
        df_copy = df_copy.filter(pl.col("utc") >= pl.datetime(year, month, day))
        df_copy = df_copy.to_pandas()
        return df_copy.drop(columns=["old_indexes", "utc"]), df_copy.old_indexes 

def restrict_time_up(df, year=2024, month=9, day=11, date = None):
    # обрезает датасет по времени ДО
    df_copy = df.copy()

    if date is not None:
        df_copy = df_copy[df_copy["utc"] <= date].reset_index().drop(columns=['index'])
        return df_copy

    df_copy = pl.from_pandas(df_copy.reset_index()).rename({"index" : "old_indexes"})
    df_copy = df_copy.filter(pl.col("utc") >= pl.datetime(year, month, day))
    df_copy = df_copy.to_pandas()
    return df_copy.drop(columns=["old_indexes", "utc"]), df_copy.old_indexes

def restrict_time_up_stupidly(df, months=2, days=0):
    # берёт первую дату в датасете (пусть это 2024.09.11) и оберзает все даты большие чем 2024.09.11 + months + days

    last_day = df['utc'][0] + pd.DateOffset(months=months, days=days)
    return restrict_time_up(df, date=last_day)

def get_constant_accuracy(y_val):
        # возвращает точность контантного предсказания на валидационной выборке
        val_const = pl.from_pandas(y_val.reset_index())
        consts = val_const.group_by(pl.col("direction_binary")).agg(pl.col("index").count())
        zeroes = consts.filter(pl.col("direction_binary") == 0)['index'].item()
        ones = consts.filter(pl.col("direction_binary") == 1)['index'].item()

        return zeroes/(ones + zeroes)
 
def train_valid_split_stupidly(data,  
                               target, last_days = 2,
                               utc = []): # utc здесь добавлено для optuna 
        
    # возвращает тестовую и валидационную выборки в завимости от заданного времени

    split_date = data["utc"].iloc[-1] - pd.DateOffset(days=last_days)
    
    train_df = data[data["utc"] < split_date]

    X_train = train_df.drop(columns=target)
    y_train = train_df[target]

    test_df = data[data["utc"] >= split_date]

    X_val = test_df.drop(columns=target)
    y_val = test_df[target]

    return X_train, X_val, y_train, y_val


class CatboostFinModel():
    """
    Класс для работы с моделью CatBoostClassifier для финансовых данных.
    Поддерживает настройку датасетов, признаков, обучение, оценку, визуализацию
    важности признаков и использование библиотеки Optuna для оптимизации.
    """
    def __init__(self, args):
        """
        Инициализирует объект CatboostFinModel с заданными параметрами модели.

        Параметры:
            args (dict): Параметры для CatBoostClassifier.
        """
        self.model = CatBoostClassifier(**args)
        self.args = args
        # запоминает наилучший результат на валидационной выборке 
        self.best_accuracy = 0
        self.features_best_accuracy = []
        self.X_train : pd.DataFrame
        self.X_val : pd.DataFrame
        self.y_train : pd.DataFrame
        self.y_val : pd.DataFrame
        self.cat : list
        self.numeric : list

    def set_datasets(self, X_train, X_val, y_train, y_val):
        """
        Устанавливает обучающие и валидационные выборки.

        Параметры:
            X_train (pd.DataFrame): Признаки для обучения.
            X_val (pd.DataFrame): Признаки для валидации.
            y_train (pd.Series): Таргеты для обучения.
            y_val (pd.Series): Таргеты для валидации.
        """
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

    def set_features(self, numeric_features, cat_features):
        """
        Устанавливает числовые и категориальные признаки.

        Параметры:
            numeric_features (list): Список числовых признаков.
            cat_features (list): Список категориальных признаков.
        """
        self.cat, self.numeric = cat_features, numeric_features
    

    def fit(self):
        """
        Обучает модель CatBoostClassifier на обучающей выборке с использованием валидационной выборки.
        """
        self.model.fit(self.X_train, self.y_train, eval_set=Pool(self.X_val, self.y_val, cat_features = self.cat), cat_features = self.cat)
        # self.print_feature_importances()
        # self.visualise_shap_values()
        return self

    def predict(self, X_test):
        """
        Выполняет предсказание на тестовых данных.

        Параметры:
            X_test (pd.DataFrame): Данные для предсказания.

        Возвращает:
            np.ndarray: Предсказания модели.
        """
        return self.model.predict(X_test)
    
    def score(self, X_test, y_test):
        """
        Вычисляет точность (Accuracy) модели на тестовых данных.

        Параметры:
            X_test (pd.DataFrame): Признаки тестовой выборки.
            y_test (pd.Series): Реальные значения тестовой выборки.

        Возвращает:
            float: Точность модели.
        """
        return self.model.score(X_test, y_test)

    def print_model_best_features(self):
        # хз что это я подумаю может убрать
        print(f"Точность: {self.best_accuracy}")
        print("Набор признаков: ")
        for feature in self.features_best_accuracy:
            print(feature)
    
    def get_model_best_features(self):
        # это с верхним вместе идет 
        return self.best_accuracy, self.features_best_accuracy

    def print_constant_accuracy(self, y_test):
        """
        Выводит точность константного предсказания (конкретно, всегда "0").

        Параметры:
            y_test (pd.Series): Реальные значения для валидационной выборки.
        """
        val_const = pl.from_pandas(y_test.reset_index())
        consts = val_const.group_by(pl.col("direction_binary")).agg(pl.col("index").count())
        zeroes = consts.filter(pl.col("direction_binary") == 0)['index'].item()
        ones = consts.filter(pl.col("direction_binary") == 1)['index'].item()

        print(f"Точность константного предсказания {max(zeroes, ones)/(ones + zeroes)}")

    def print_feature_importances(self):
        """
        Выводит важности признаков, отсортированные по их влиянию на модель.
        """
        indexes = np.argsort(self.model.feature_importances_)
        sorted_importances = self.model.feature_importances_[indexes]
        sorted_names = np.array(self.model.feature_names_)[indexes]
        for i, imp in enumerate(sorted_importances):
            print(imp, sorted_names[i])

    def get_shap_values(self):
        """
        Вычисляет значения SHAP для объяснения модели.

        Возвращает:
            np.ndarray: SHAP значения для обучающих данных.
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_train)
        return shap_values
    
    def visualise_shap_values(self):
        """
        Визуализирует значения SHAP в виде графика важности признаков.
        """
        shap_values = self.get_shap_values()
        shap.summary_plot(shap_values, self.X_train)

    def optuna_choose_time(self, valid_date : dt.datetime, days_num = 15, interval_num = 10, metric = "Accuracy"):
        """
        Выполняет оптимизацию времени валидации с использованием Optuna.

        Параметры:
            valid_date (datetime): Дата начала валидации.
            days_num (int): Количество дней для анализа.
            interval_num (int): Интервалы для разбиения.
            metric (str): Метрика для оптимизации (по умолчанию "Accuracy").
        """

        storage = RDBStorage(url="sqlite:///optuna_trials.db")
        pruner = MedianPruner(n_min_trials=3)
        current_date = valid_date
        self.X_val = self.X_val.drop(columns=["utc"])

        def objective(trial):
            nonlocal current_date
            days_from_date = trial.suggest_int("days_from_date", 
                                         low=days_num, 
                                         high=days_num*interval_num, 
                                         step=interval_num)
            
            current_date = current_date - dt.timedelta(days=days_from_date)

            current_X_train, indexes = restrict_time_down(self.X_train, 
                                                 current_date.year, 
                                                 current_date.month, 
                                                 current_date.day)
            
            current_y_train = self.y_train[indexes]

            model = CatBoostClassifier(
                **self.args, verbose=0 # выводим в консоль 0 информации о процессе обученя
            )

            pruning_callback = CatBoostPruningCallback(trial, metric)

            model.fit(
                current_X_train, 
                current_y_train, 
                eval_set=Pool(self.X_val, self.y_val, cat_features=self.cat), 
                cat_features=self.cat, 
                verbose=0,  
                callbacks=[pruning_callback]
            )

            pruning_callback.check_pruned()

            accuracy = model.score(self.X_val, self.y_val)
            return accuracy
    
        study = optuna.create_study(direction="maximize", pruner=pruner, storage=storage, load_if_exists=True)
        study.optimize(objective, n_trials=interval_num)

    def cross_validation(self, df, cat, n_samples = 3):
        '''
        данная функция рандомно берёт два месяца начиная с 2024.01.01 и на них обучает, потом тестит на последующих двух дня
        если точность константного предсказания лежит в (0.49, 0.52), то добавляем полученную accuracy к итоговому списку
        в конце считаем среднее ариф.  из списка
        '''

        #подумать, как учитывать тот факт, что тестовые выборки не всегда одного размера (пока что надо как-то в среднем арифметическом веса учитывать)
        
        trials_sum, trials_cnt = 0, 0
        trial_on_const_accuracy = 3

        while n_samples > 0:
            # np.random.seed(100)
            month = np.random.randint(low=7, high=9)
            day = np.random.randint(31)

            first_date = dt.datetime(2024, 1, 1) + relativedelta(months=month, days=day)
            df_restricted = restrict_time_down(df, date=first_date)
            df_restricted = restrict_time_up_stupidly(df_restricted)

            print('fist_date:', df_restricted['utc'].iloc[0], '- last_date:', df_restricted['utc'].iloc[-1])


            X_train, X_val, y_train, y_val = train_valid_split_stupidly(df_restricted, target = "direction_binary", last_days=2)
            const_acc = get_constant_accuracy(y_val)
            print('const_acc:', const_acc)
            
            if trial_on_const_accuracy > 0 and (const_acc > 0.52 or const_acc < 0.49):
                trial_on_const_accuracy -= 1
                continue

            print(y_val.shape)
            new_model = self.model
            new_model.fit(X_train, y_train, eval_set=Pool(X_val, y_val, cat_features = cat), cat_features = cat, verbose=False)

            # const_acc = get_constant_accuracy(y_val)
            # print('const_acc:', const_acc)
            
            # if trial_on_const_accuracy > 0 and const_acc > 0.52 or const_acc < 0.49:
            #     trial_on_const_accuracy -= 1
            #     continue

            trial_on_const_accuracy = 3
            n_samples -= 1
            y_pred = new_model.predict(X_val)
            acc = accuracy_score(y_pred, y_val)
            trials_sum += acc
            trials_cnt += 1
            print(f"On trial {n_samples} with date {first_date} got accuracy {acc}")
        
        return trials_sum / trials_cnt