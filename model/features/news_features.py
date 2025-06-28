from ..newsdata import NewsFinData
import pandas as pd


class NewsFeaturesMixin:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.cat_features: list = []
        self.numeric_features: list = []

    def insert_news_features(self, news_df, 
                              step = 60, am_news = 50, threshold=0.6):
        # создаем объект класса новости
        news = NewsFinData(news_df)
        # подстраиваем новостное время к времени свечей 
        # на параметрах ПО УМОЛЧАНИЮ (1 минута, начало 7:00, конец 20:00)
        news.adjust_time_to_candles()
        # фильтруем новости по датам аналогично датафрейму со свечами
        # нужна колонка utc
        start_date = self.df.utc.iloc[0]
        end_date = self.df.utc.iloc[-1]
        # здесь нужна колонка Date
        news.filter_by_datetime_range('Date', start_date, end_date)
        # соединяем новости со свечами, по умолчанию смотрим поминутно
        news.merge_news_with_df(self.df, 'utc')
        news.compute_price_changes_all([step]) # для цены закрытия
        hashtag_impact = news.compute_hashtag_influence(change_col=f'pct_change_{step}')
        # выбираем топ-10 хэштегов по влиянию 
        hashtags = hashtag_impact[hashtag_impact.counts > am_news].head(10)['hashtag'].tolist()
        # выделяем значимые новости в датасете с новостями
        news.add_imp_tags(imp_tags=hashtags)
        # теперь проставляем метки -1, 0, 1 и проставляем временные рамки - сколько минут прошло с новости
        news.add_news_labels(threshold, [step])
        news.propagate_news_labels()
        self.df['new_label'] = news.cand_news_df['news_label']
        self.df['time_from_new'] = news.cand_news_df['time_from_new']
        # добавляем в фичи 
        self.cat_features += ['new_label']
        self.numeric_features += ['time_from_new']