import requests
import json

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


class MoexRequest:
    HEAD_URL = 'https://iss.moex.com/iss/'
    markets = []
    engines = []
    boards = []
    security_types = []
    securities = []

    def get(self, url, params={}):
        return requests.get(url=self.HEAD_URL + url, params=params).json
    
    def general_info(self):
        url = 'index.json'
        j = self.get(url)

        self.markets = [x[4] for x in j['markets']['data']]
        self.engines = [x[1] for x in j['engines']['data']]
        self.boards = [x[4] for x in j['boards']['data']]
        self.security_types = [x[4] for x in j['security_types']['data']]

        url = '/securities.json'
        j = self.get(url)
        self.securities = [x[1] for x in j['securities']['data']]

        return
    
    def get_ofz_bonds(self):
        """
        тут только гос облигации

        возвращает 
        securities: list -> тикеры ценных бумаг
        """

        url = 'engines/stock/markets/bonds/securities.json'
        params = {'sectypes': '3'}

        j = self.get(url, params).json()
        return [x[0] for x in j['securities']['data']]
    
    def candles_no_board(self, market="shares", engine="stocks", security="SU26207RMFS9", interval=10):
        url = f'engines/{engine}/markets/{market}/securities/{security}/candles.json'
        params = {'interval': interval}

        return self.get(url, params)
    
    def get_age_of_secid(self, market="shares", engine="stocks", security="SU26207RMFS9"):
        url = f'engines/{engine}/markets/{market}/securities/{security}/candleborders.json'
        j = self.get(url)
        return j['border']['data'][0][0] # TODO: добавить сюда именно разность в минутах (?) с текущим временем 

    def make_df(self, j, name):
        return pd.DataFrame([{k : r[i] for i, k in enumerate(j[name]['columns'])} for r in j[name]['data']])
    
