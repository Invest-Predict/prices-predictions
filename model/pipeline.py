import datetime as dt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from .data import FinData

def initialize_findata(df_path, fill_skips):
    def _initialize_findata(X):
        return FinData(df_path, fill_skips=fill_skips)
    return FunctionTransformer(_initialize_findata, validate=False)

def restrict_time_down(weeks):
    def _restrict_time_down(data):
        data.restrict_time_down(data.df['utc'].min() - dt.timedelta(weeks=weeks))
        return data
    return FunctionTransformer(_restrict_time_down, validate=False)

def restrict_time_up(end_dt):
    def _restrict_time_up(data):
        data.restrict_time_up(end_dt)
        return data
    return FunctionTransformer(_restrict_time_up, validate=False)

def insert_features(features):
    def _insert_features(data):
        data.insert_all(features_settings=features)
        return data
    return FunctionTransformer(_insert_features, validate=False)

def restrict_time_down_to_start(start_dt):
    def _restrict_time_down_to_start(data):
        data.restrict_time_down(start_dt)
        return data
    return FunctionTransformer(_restrict_time_down_to_start, validate=False)

def filter_trading_hours():
    def _filter_trading_hours(data):
        data.df = data.df[(data.df['utc'].dt.time >= dt.time(7, 0)) & (data.df['utc'].dt.time <= dt.time(21, 0))]
        return data
    return FunctionTransformer(_filter_trading_hours, validate=False)

def create_pipeline(df_path, start_dt, end_dt, features=None, fill_skips=True):
    pipeline = Pipeline([
        ('initialize_findata', initialize_findata(df_path, fill_skips)),
        ('restrict_time_down', restrict_time_down(weeks=4)),
        ('restrict_time_up', restrict_time_up(end_dt)),
        ('insert_features', insert_features(features)),
        ('restrict_time_down_to_start', restrict_time_down_to_start(start_dt)),
        ('filter_trading_hours', filter_trading_hours()),
    ])
    return pipeline

# Example usage
# df_path = 'path/to/your/data.csv'
# start_dt = dt.datetime(2024, 1, 1)
# end_dt = dt.datetime(2024, 12, 31)
# features = {
#     "shifts_norms": [1, 2, 3],
#     "ma": [5, 10],
#     "ema": [5, 10],
#     "boll": True,
#     "rsi": [14],
#     "hl_diff": [5, 10],
#     "stoch_osc": [14],
#     "rand_pred": True,
#     "mini_features": [1, 2, 3]
# }
# fill_skips = True

# pipeline = create_pipeline(df_path, start_dt, end_dt, features, fill_skips)
# findata = pipeline.fit_transform(None)