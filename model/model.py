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
from sklearn.metrics import accuracy_score, classification_report
import logging


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("testing.log"),
        logging.StreamHandler()
    ]
)



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
    

    def fit(self, mod = True):
        """
        Обучает модель CatBoostClassifier на обучающей выборке с использованием валидационной выборки.
        """
        if mod:
            self.X_val = self.X_val[self.numeric + self.cat]
            self.X_train = self.X_train[self.numeric + self.cat]
        self.model.fit(self.X_train, self.y_train, eval_set=Pool(self.X_val, self.y_val, cat_features = self.cat), cat_features = self.cat)
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
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
    
    def get_constant_accuracy(self, y_test, unknown = False, target_name = "direction_binary"):
        val_const = pl.from_pandas(y_test.reset_index())
        consts = val_const.group_by(pl.col(target_name)).agg(pl.col("index").count())
        zeroes = consts.filter(pl.col(target_name) == 0)['index'].item()
        ones = consts.filter(pl.col(target_name) == 1)['index'].item()
        return max(zeroes, ones)/(ones + zeroes) if not unknown else zeroes/(ones + zeroes)

    def score(self, X_scored, y_scored, output_dict=False):
        y_pred = self.model.predict(X_scored)
        return(classification_report(y_scored, y_pred, output_dict=output_dict))

    def _get_sorted_feature_importances(self):
        indexes = np.argsort(self.model.feature_importances_)
        sorted_importances = self.model.feature_importances_[indexes]
        sorted_names = np.array(self.model.feature_names_)[indexes]
        return zip(sorted_importances, sorted_names)


    def print_feature_importances(self):
        """
        Выводит важности признаков, отсортированные по их влиянию на модель.
        """
        indexes = np.argsort(self.model.feature_importances_)
        sorted_importances = self.model.feature_importances_[indexes]
        sorted_names = np.array(self.model.feature_names_)[indexes]
        for i, imp in enumerate(sorted_importances):
            print(imp, sorted_names[i])

    def get_top_imp_features(self, n : int):
        indexes = np.argsort(self.model.feature_importances_)
        sorted_importances = self.model.feature_importances_[indexes][-n:]
        sorted_names = np.array(self.model.feature_names_)[indexes][-n:]
        for i, imp in enumerate(sorted_importances):
            print(imp, sorted_names[i])
        return sorted_names

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
        
        """
        Оптимизирует количество свечей для обучения модели с использованием Optuna.

        Параметры:
            candles_interval_low (int): Минимальное количество свечей для обучения.
            candles_interval_high (int): Максимальное количество свечей для обучения.
            number_of_trials (int): Количество попыток для Optuna.
            min_trials_for_pruner (int): Минимальное количество попыток для активации прунера.

        Исключения:
            ValueError: Выбрасывается, если размер обучающего датасета меньше candles_interval_high.

        Оптимизация проводится с использованием MedianPruner, чтобы исключить малообещающие гипотезы.
        """

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
                                                     number_of_trials = 10, n_warmup_steps=0, verbose=1000):
        """
        Оптимизирует гиперпараметры модели CatBoostClassifier с использованием Optuna.

        Параметры:
            changing_params (dict): Словарь параметров для оптимизации. Ключи - названия параметров,
                значения - словари с "low" и "high" границами значений (по умолчанию содержит learning_rate, l2_leaf_reg и depth).
            min_trials_for_pruner (int): Минимальное количество попыток для активации прунера.
            number_of_trials (int): Количество попыток для Optuna.
            n_warmup_steps (int): Количество деревьев до того как прунер начнет прунить.
            verbose (int): Отчет катбуста каждые verbose итераций.

        Пример changing_params:
            {
                "learning_rate": {"low": 0.001, "high": 0.03},
                "l2_leaf_reg": {"low": 3, "high": 500},
                "depth": {"low": 3, "high": 6}
            }

        Оптимизация проводится с использованием MedianPruner, чтобы исключить малообещающие гипотезы.
        """
        
        storage = RDBStorage(url="sqlite:///optuna_params_trials.db")
        pruner = MedianPruner(n_warmup_steps=n_warmup_steps, n_min_trials=min_trials_for_pruner)

        args = self.args

        def objective(trial):
            nonlocal args

            # learning rate
            if changing_params["learning_rate"] is not None:
                suggested_learning_rate = trial.suggest_float("learning_rate", 
                                                            low=changing_params["learning_rate"]["low"], 
                                                            high=changing_params["learning_rate"]["high"])
                args["learning_rate"] = suggested_learning_rate
            
            # regularization
            suggested_l2_leaf_reg = trial.suggest_int("l2_leaf_reg", 
                                                      low=changing_params["l2_leaf_reg"]["low"], 
                                                      high=changing_params["l2_leaf_reg"]["high"])
            args["l2_leaf_reg"] = suggested_l2_leaf_reg

            # depth
            suggested_depth = trial.suggest_int("depth", 
                                                low=changing_params["depth"]["low"], 
                                                high=changing_params["depth"]["high"])
            args["depth"] = suggested_depth

            model = CatBoostClassifier(**args)

            pruning_callback = CatBoostPruningCallback(trial, metric=args["eval_metric"])

            model.fit(
                self.X_train, 
                self.y_train, 
                eval_set=Pool(self.X_val, self.y_val, cat_features=self.cat), 
                cat_features=self.cat, 
                verbose=verbose,  
                callbacks=[pruning_callback]
            )

            pruning_callback.check_pruned()

            accuracy = model.score(self.X_val, self.y_val)
            return accuracy
    
        study = optuna.create_study(direction="maximize", pruner=pruner, storage=storage, load_if_exists=True)
        study.optimize(objective, n_trials=number_of_trials)

    def cross_validation(self, X, y, n_samples = 5):
        '''
        Кросс валиадция, в которой выборка бьётся на (n_samples + 2)
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
    

    def test_trading(self, df = None, target = 'direction_binary', start_date = None, end_date = None, proportion = [3, 1, 1],
                    train_df = None, val_df = None, test_df = None, short = True,
                    initial_budget = 10000, cat = [], num = [], commision = 0.0001):
        '''
        Примитиваня стратегия, пусть мы просто пока покупаем акцию сейчас, если предполагаем, что через десять минут она вырастит в цене
        (через 10 минут в этом случае её продаём)
        В ином случае мы ничего не делаем (ждём следующий период)
        Но также у нас есть ограничение - это бюджет (он ограчен => не всегда сможем купить акцию, чтобы продать её через 10 минут)
        '''

        if train_df is None or val_df is None or test_df is None:
            df_copy = df.copy()
            train_size, val_size, test_size = 1000, 180, 180
            if end_date is not None:
                df_copy = df[df["utc"] <= end_date].reset_index().drop(columns=['index'])
            if start_date is not None:
                    df_copy = df_copy[df_copy["utc"] >= start_date].reset_index().drop(columns=['index'])
                    df_size = df_copy.shape[0]
                    train_size, val_size = int(df_size * (proportion[0] / sum(proportion))), int(df_size * (proportion[1] / sum(proportion)))
                    test_size = df_size - train_size - val_size

            X, y = df_copy.drop(columns=target), df_copy[target]

            X_train, X_val, X_test = X[-(train_size + val_size + test_size):-(val_size + test_size)], X[-(val_size + test_size): -test_size], X[-test_size:]
            y_train, y_val, y_test = y[-(train_size + val_size + test_size):-(val_size + test_size)], y[-(val_size + test_size): -test_size], y[-test_size:]
        
        else:
            X_train, X_val, X_test = train_df.drop(columns=target), val_df.drop(columns=target), test_df.drop(columns=target)
            y_train, y_val, y_test = train_df[target], val_df[target], test_df[target]
        
        logging.info("Backtesting started")
        logging.info(f"Train dates: {X_train['utc'].iloc[0]} - {X_train['utc'].iloc[-1]} | Valid dates: {X_val['utc'].iloc[0]} - {X_val['utc'].iloc[-1]} | Test dates: {X_test['utc'].iloc[0]} - {X_test['utc'].iloc[-1]}")

        X_train, X_val = X_train[num + cat], X_val[num + cat]
        self.set_datasets(X_train, X_val, y_train, y_val)
        self.set_features(num, cat)

        self.fit()

        history = pd.DataFrame(columns=["datetime", "budget"])
        history.loc[0] = [X_test['utc'].iloc[0], initial_budget]
        money = initial_budget


        for i in range(X_test.shape[0] - 1):
            y_pred = self.predict(X_test[num + cat].iloc[i])
            close_in_ten_min = X_test['close'].iloc[i + 1]
            open_now = X_test['close'].iloc[i]

            history.loc[i + 1] = [X_test['utc'].iloc[i + 1], money]

            if money >= open_now and y_pred == 1:
                commission_now = ((open_now + close_in_ten_min) * commision) * (money  // open_now)
                money += (close_in_ten_min - open_now) * (money  // open_now) - commission_now

                logging.info(f"LONG! - Date&Time: {X_test['utc'].iloc[i]} - I bought Yandex for {open_now} and sold for {close_in_ten_min} + commission {commission_now} -> budget: {money}")
            elif y_pred == 0 and short == True:
                commission_now = ((open_now + close_in_ten_min) * commision) * (money // close_in_ten_min)
                money += (open_now - close_in_ten_min) * (money  // open_now) - commission_now
                logging.info(f"SHORT! - Date&Time: {X_test['utc'].iloc[i]} - I bought Yandex for {close_in_ten_min} and sold for {open_now} + commission {commission_now} -> budget: {money}")

        logging.info(f"\n\n\nMy budget before {initial_budget} and after trading {money}\nMommy, are you prod of me?")

        return money - initial_budget,  self.model.score(X_test, y_test) # ны выходе прибыль


    def test_weekly(self, df, start_dt = dt.datetime(2024, 1, 1), end_dt=dt.datetime(2024, 12, 31), proportion = [15, 2, 3], target = 'direction_binary', cat = [], num = []):
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

        self.set_datasets(X_train, X_val, y_train, y_val)
        self.set_features(num, cat)

        self.fit()

        self.get_top_imp_features(20)

        # self.predict(X_test)

        return self.model.score(X_test, y_test)


    def test_intersect(self, df, start_dt = dt.datetime(2024, 12, 1), end_dt=dt.datetime(2024, 12, 27), proportion = [15, 2, 3], target = 'direction_binary', cat = [], num = []):
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

            self.set_datasets(X_train, X_val, y_train, y_val)
            self.set_features(num, cat)

            self.fit()
            accuracy_sum += self.model.score(X_test, y_test)
            cnt += 1
        
        return accuracy_sum / cnt
    
    def test_within_category(self, category, start_dt = dt.datetime(2024, 1, 1), end_dt=dt.datetime(2024, 12, 31), proportion = [15, 2, 3]):
        """
        Обучает несколько моделей, последовательно выбирая в качестве таргета каждую акцию из категории и заполняя признаками общий датасет.
        Для target-акции заполняются признаки shifts_norms, high_low_diff, rolling_means и exp_rolling_means с параметрами по умолчанию.
        Для остальных акций добавляются признаки angle_{other_name} и angle_ln_{other_name}.
        Тестирует при помощи test_weekly с заданным промежутком.

        Параметры:
        category (str) - название категории из datasets/categories.py
        start_dt (datetime) - начало периода
        end_dt (datetime) - конец периода
        proportion (list) - деление на train val test

        Возвращает: список названий акций, для которых обучалась модель и проводилось предсказание и список Accuracy на тесте для каждой модели.
        """
        target_names = []
        accuracies = []

        # Каждая акция из категории побывает таргетом
        for target_name in category:
            target_names.append(target_name)
            dfs = []
            numerics = []
            cats = []
            
            for name in category:
                data = FinData(f"../../datasets/{name}_10_min.csv")
                data.restrict_time_down(start_dt)

                if name == target_name:
                    # windows_ma = [2, 3, 5, 7, 9, 18, 21, 28, 30, 50, 500]
                    # data.insert_time_features()
                    data.insert_rolling_means(windows_ma=[3,6,18])
                    data.insert_shifts_norms()
                    data.insert_exp_rolling_means()
                    data.insert_high_low_diff()
                    data.make_binary_class_target(target_name="direction_binary")

                data.df.set_index('utc', inplace=True)

                if name != target_name:
                    data.df.rename({feature: feature + '_' + name for feature in data.df.columns}, axis=1, inplace=True)
                    data.numeric_features = [feature + '_' + name for feature in data.numeric_features]
                    data.cat_features = [feature + '_' + name for feature in data.cat_features]

                dfs.append(data.df)
                numerics += data.numeric_features
                cats += data.cat_features

            joint_data = FinData(pd.concat(dfs, axis=1).reset_index())

            # добавляем углы 
            for name in category:
                if name != target_name:
                    joint_data.insert_angle(name)
                    joint_data.insert_angle_ln(name)
                    numerics += [f'angle_{name}', f'angle_ln_{name}']

            joint_data.numeric_features = numerics
            joint_data.cat_features = cats

            numeric = joint_data.get_numeric_features()
            cat = joint_data.get_cat_features()

            # Аргументы для моделей
            args = {"iterations" : 5000, 
            "depth" : 6, 
            "use_best_model" : True, 
            "verbose" : False,
            "l2_leaf_reg" : 350,
            "loss_function" : 'Logloss', 
            "eval_metric" : 'Accuracy', 
            "cat_features" : cat, 
            "random_state" : 42,
            "early_stopping_rounds" : 1000}
                            
            model = CatboostFinModel(args=args)

            test_accuracy = model.test_weekly(df=joint_data.df, start_dt=start_dt, end_dt=end_dt, proportion=proportion, cat=cat, num=numeric)
            accuracies.append(test_accuracy)
        return target_names, accuracies