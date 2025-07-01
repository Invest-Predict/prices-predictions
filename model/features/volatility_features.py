from ..newsdata import NewsFinData
import pandas as pd
import numpy as np


class VolatilityFeaturesMixin:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.cat_features: list = []
        self.numeric_features: list = []

    def insert_corwin_and_schultz_features(self):
        log_hl = np.log(self.df['high'] / self.df['low'])
        beta = (log_hl.rolling(window=2).apply(lambda x: (x**2).sum(), raw=True)).shift(-1)
        gamma = (np.log(self.df['high'].rolling(window=2).max() / self.df['low'].rolling(window=2).min()) ** 2).shift(-1)

        cs_vol = np.sqrt( (gamma - 0.5 * beta) / (np.log(2) - 0.5) )

        self.df['corwin_and_schultz'] = cs_vol
        if 'corwin_and_schultz' not in self.numeric_features:
            self.numeric_features += ['corwin_and_schultz']

    def insert_parkinson_features(self):

        self.df['sigma2_parkinson'] = (1 / (4 * np.log(2))) * (np.log(self.df['high'] / self.df['low']))**2
        if 'sigma2_parkinson' not in self.numeric_features:
            self.numeric_features += ['sigma2_parkinson']
