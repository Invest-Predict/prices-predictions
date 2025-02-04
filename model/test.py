from .model import CatboostFinModel
from .data import FinData
from .preprocessing import train_valid_test_split
# from utils import make_features, calculate_avret, append_tests_data #!!!!
import datetime as dt
import numpy as np

def make_features(data : FinData, features_settings : dict):
    # Attention: inplace modification
    # TODO Move to FinData class
    features = list(features_settings.keys())
    if "shifts_norms" in features:
        data.insert_shifts_norms(features_settings["shifts_norms"])
    if "ma" in features:
        data.insert_rolling_means(features_settings["ma"])
    if "ema" in features:
        data.insert_exp_rolling_means(features_settings["ema"])
    if "boll" in features:
        data.insert_bollinger()
    if "rsi" in features:
        data.insert_rsi()
    if "hl_diff" in features:
        data.insert_high_low_diff()
    if "stoch_osc" in features:
        data.insert_stochastic_oscillator()
    if "rand_pred" in features:
        data.insert_random_prediction()

def calculate_avret(matrix_probs : np.array, matrix_tests : np.array, norm_value):
    idx_max = np.argpartition(matrix_probs, -norm_value, axis=0)[-norm_value:]
    idx_min = np.argpartition(matrix_probs, norm_value, axis=0)[:norm_value]
    values_max_curr, values_max_next = np.take_along_axis(matrix_tests, idx_max, axis=0), np.take_along_axis(matrix_tests, idx_max + 1, axis=0)
    values_min_curr, values_min_next = np.take_along_axis(matrix_tests, idx_min, axis=0), np.take_along_axis(matrix_tests, idx_min + 1, axis=0)
    return (np.sum(values_max_next/values_max_curr))/norm_value - (np.sum(values_min_next/values_min_curr))/norm_value

def test_average_return(dfs_paths, start_period : dt.datetime, feature_settings : dict, args : dict,
                        train_size = 1000, 
                        val_size = 500, 
                        test_size = 500,
                        percentage = 0.2):
    
    n = len(dfs_paths)
    x = int(np.ceil(n * percentage))
    tests = []
    probs = []

    for path in dfs_paths:
        data = FinData(path)
        make_features(data, feature_settings)
        num = data.get_numeric_features()
        cat = data.get_cat_features()
        modified_args = args
        modified_args["cat_features"] = cat
        target = data.target
        X_train, X_val, X_test, y_train, y_val, y_test = train_valid_test_split(data.df, start_period, train_size, val_size, test_size, num, cat, target)
        model = CatboostFinModel(args = args)
        model.set_datasets(X_train, X_val, y_train, y_val)
        model.set_features(num, cat)
        model.fit()
        tests.append(y_test)
        # вероятности для последних периодов не добавляем, так как для них не будет будущего 
        probs.append(model.predict_proba(X_test)[:,1][:-1]) 

    matrix_tests = np.array(tests)
    matrix_probs = np.array(probs)

    return calculate_avret(matrix_probs, matrix_tests, norm_value=x)


    





