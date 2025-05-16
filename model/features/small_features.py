import pandas as pd
import numpy as np
import datetime as dt

class SmallFeaturesMixin:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.cat_features: list = []
        self.numeric_features: list = []

    def insert_small_close_shifts(self, small_df, candles_num = 10, step = 1):
        if isinstance(small_df, pd.DataFrame):
            df = small_df.copy()
        else:
            df = pd.read_csv(small_df)

        new_num = []
        start_utc = self.df.iloc[0].utc - dt.timedelta(weeks=1)
        df.utc = pd.to_datetime(df.utc).dt.tz_localize(None)
        df = df[df.utc >= start_utc]
        for i in range(candles_num, 0, -step):
            df[f"close_t-{i}"] = df["close"].shift(i) / df["close"].shift(i - 1)
            new_num.append(f"close_t-{i}")
            
        df = df.reset_index()
        self.df.set_index("utc", inplace=True)
        self.df = pd.merge_asof(self.df.reset_index(), df, on="utc", direction="backward", suffixes=("", "_min"))
        
        self.numeric_features += new_num 





    