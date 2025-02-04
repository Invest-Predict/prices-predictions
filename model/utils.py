from model import CatboostFinModel
from data import FinData
from preprocessing import train_valid_test_split
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
    idx_min = np.argpartition(matrix_probs, norm_value, axis=0)[norm_value:]
    values_max_curr, values_max_next = np.take_along_axis(matrix_tests, idx_max, axis=0), np.take_along_axis(matrix_tests, idx_max + 1, axis=0)
    values_min_curr, values_min_next = np.take_along_axis(matrix_tests, idx_min, axis=0), np.take_along_axis(matrix_tests, idx_min + 1, axis=0)
    return (np.sum(values_max_next/values_max_curr))/norm_value - (np.sum(values_min_next/values_min_curr))/norm_value

def append_tests_data(dfs_paths, start_period, feature_settings, args, train_size, val_size, test_size):
    pass

def pupa():
    pass

def lupa():
    pass

