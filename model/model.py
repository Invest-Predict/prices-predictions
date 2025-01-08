from catboost import CatBoostClassifier, Pool
import polars as pl
import pandas as pd
import numpy as np
import shap
import optuna 
import datetime as dt
from optuna.pruners import MedianPruner
from optuna.storages import RDBStorage
from optuna.integration import CatBoostPruningCallback

# визуализация предсказаний модели и матрицы 
# подбирать время для предсказаний - optuna chose time

# кросс-валидация 

# куча признаков и выбираются самые важные 



# чуть-чуть Лера переделает 
def restrict_time_down(df, year, month, day):
        # обрезает датасет по времени ОТ
        df_copy = df.copy()
        df_copy = pl.from_pandas(df_copy.reset_index()).rename({"index" : "old_indexes"})
        df_copy = df_copy.filter(pl.col("utc") >= pl.datetime(year, month, day))
        df_copy = df_copy.to_pandas()
        return df_copy.drop(columns=["old_indexes", "utc"]), df_copy.old_indexes 

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
        self.features_best_acuuracy = []
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

        print(f"Точность константного предсказания {zeroes/(ones + zeroes)}")

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


    def optuna_chose_time(self, 
                          valid_date : dt.datetime, days_num = 15, interval_num = 10, metric = "Accuracy"):
        """
        Выполняет оптимизацию времени валидации с использованием Optuna.

        Параметры:
            valid_date (datetime): Дата начала валидации.
            days_num (int): Количество дней для анализа.
            interval_num (int): Интервалы для разбиения.
            metric (str): Метрика для оптимизации (по умолчанию "Accuracy").
        """

        storage = RDBStorage(url="sqlite:///optuna_trials.db")
        pruner = MedianPruner(n_min_trials=5)
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
                verbose=300,  
                callbacks=[pruning_callback]
            )

            accuracy = model.score(self.X_val, self.y_val)
            return accuracy
    
        study = optuna.create_study(direction="maximize", pruner=pruner, storage=storage, load_if_exists=True)
        study.optimize(objective, n_trials=20)

        best_params = study.best_params
        print(f"Лучшее количество дней: {best_params['days_from_date']}")
        print(f"Лучший результат: {study.best_value}")

        return best_params
        








    
        


        
    