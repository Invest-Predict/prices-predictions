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
from .data import FinData # это надо нам или не? 
from sklearn.metrics import accuracy_score

# визцализация предсказаний по матрице 
# куча признаков и выбираются самые важные 
# бутстрап 


def get_constant_accuracy(y_val):
        # возвращает точность контантного предсказания на валидационной выборке
        val_const = pl.from_pandas(y_val.reset_index())
        consts = val_const.group_by(pl.col("direction_binary")).agg(pl.col("index").count())
        zeroes = consts.filter(pl.col("direction_binary") == 0)['index'].item()
        ones = consts.filter(pl.col("direction_binary") == 1)['index'].item()

        return zeroes/(ones + zeroes)



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
        # чтобы не нужно было постоянно заново разделять выборку перед fit, достаточно просто установить numeric и сat в модели 
        # и передать можно со всеми признаками X_val и X_train
        self.X_val = self.X_val[self.numeric + self.cat]
        self.X_train = self.X_train[self.numeric + self.cat]
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
    
    def __call__(self, X_test):
        return self.predict(X_test)
    
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

    def optuna_choose_time(self, candles_interval_low = 3000,
                                 candles_interval_high = 30000,  
                                 number_of_trials = 10,
                                 min_trials_for_pruner = 3):

        storage = RDBStorage(url="sqlite:///optuna_time_trials.db")
        pruner = MedianPruner(n_min_trials=min_trials_for_pruner)
        len_X_train = self.X_train.shape[0] 

        if len_X_train < candles_interval_high:
                raise ValueError(
                    f"Максимальное количество свечей для обучения: {candles_interval_high}. "
                    f"Количество свечей в датасете для обучения: {len_X_train}."
                )
        
        def objective(trial):
            nonlocal len_X_train
            num_train_candles = trial.suggest_int("num_train_candles", 
                                         low=candles_interval_low, 
                                         high=candles_interval_high)
            
            current_X_train = self.X_train[len_X_train - num_train_candles : ]
            current_y_train = self.y_train[len_X_train - num_train_candles : ]

            model = CatBoostClassifier(
                **self.args
            )

            pruning_callback = CatBoostPruningCallback(trial, metric=self.args["eval_metric"])

            model.fit(
                current_X_train, 
                current_y_train, 
                eval_set=Pool(self.X_val, self.y_val, cat_features=self.cat), 
                cat_features=self.cat, 
                verbose=1000,  
                callbacks=[pruning_callback]
            )

            pruning_callback.check_pruned()

            accuracy = model.score(self.X_val, self.y_val)
            return accuracy
    
        study = optuna.create_study(direction="maximize", pruner=pruner, storage=storage, load_if_exists=True)
        study.optimize(objective, n_trials=number_of_trials)

    def optuna_choose_params(self, changing_params : dict = {"learning_rate" : {"low" : 0.001, "high" : 0.03},
                                                             "l2_leaf_reg" : {"low" : 3, "high" : 500}, 
                                                             "depth" : {"low" : 3, "high" : 6}},
                                                     min_trials_for_pruner = 3, 
                                                     number_of_trials = 10):
        
        storage = RDBStorage(url="sqlite:///optuna_params_trials.db")
        pruner = MedianPruner(n_min_trials=min_trials_for_pruner)

        args = self.args

        def objective(trial):
            nonlocal args
            suggested_learning_rate = trial.suggest_float("learning_rate", 
                                                          low=changing_params["learning_rate"]["low"], 
                                                          high=changing_params["learning_rate"]["high"])
            
            suggested_l2_leaf_reg = trial.suggest_int("l2_leaf_reg", 
                                                      low=changing_params["l2_leaf_reg"]["low"], 
                                                      high=changing_params["l2_leaf_reg"]["high"])
            suggested_depth = trial.suggest_int("depth", 
                                                low=changing_params["depth"]["low"], 
                                                high=changing_params["depth"]["high"])

            args["learning_rate"] = suggested_learning_rate
            args["l2_leaf_reg"] = suggested_l2_leaf_reg
            args["depth"] = suggested_depth

            model = CatBoostClassifier(**args)

            pruning_callback = CatBoostPruningCallback(trial, metric=args["eval_metric"])

            model.fit(
                self.X_train, 
                self.y_train, 
                eval_set=Pool(self.X_val, self.y_val, cat_features=self.cat), 
                cat_features=self.cat, 
                verbose=1000,  
                callbacks=[pruning_callback]
            )

            pruning_callback.check_pruned()

            accuracy = model.score(self.X_val, self.y_val)
            return accuracy
    
        study = optuna.create_study(direction="maximize", pruner=pruner, storage=storage, load_if_exists=True)
        study.optimize(objective, n_trials=number_of_trials)

    def cross_validation(self, X, y, n_samples = 5):
        '''
        кросс валиадция, в которой выборка бьётся на (n_samples + 2)
        каждый раз (всего n_folds) выборка train увеличивается на один fold, а test и val остаются таким же по размеру
        на тестовой выборке считается accuracy и добавляется в массив
        на выходе средняя accuracy
        подробнее здесь в секции "Кросс-валидация на временных рядах" - https://education.yandex.ru/handbook/ml/article/kross-validaciya
        '''

        #подумать, как учитывать тот факт, что тестовые выборки не всегда одного размера (пока что надо как-то в среднем арифметическом веса учитывать)
        
        fold_size = X.shape[0] // (n_samples + 2)
        scores = []
        
        # Цикл по всем частям
        for i in range(n_samples):
            # Выборка тренировочной и тестовой частей
            train_idx = range(0, (i + 1) * fold_size)
            val_idx = range((i + 1) * fold_size, (i + 2) * fold_size)
            test_idx = range((i + 2) * fold_size, (i + 3) * fold_size)
            
            # Обучение модели на тренировочной части
            model = self.model
            model.fit(
                X.iloc[train_idx, :], 
                y.iloc[train_idx], 
                eval_set=Pool(X.iloc[val_idx, :], y.iloc[val_idx], cat_features=self.cat), 
                cat_features=self.cat, 
                verbose=1000
            )

            # Прогнозирование на тестовой части и сохранение результата
            predictions = model.predict(X.iloc[test_idx, :])
            scores.append(accuracy_score(y.iloc[test_idx], predictions))
        
        print(f"Array of scores: {scores}")
            
        return sum(scores) / n_samples