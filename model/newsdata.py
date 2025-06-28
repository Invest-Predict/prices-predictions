from collections import defaultdict
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date, time, timedelta
import ast


def contains_keywords(hashtags_nested, keywords_upper):
    if not isinstance(hashtags_nested, list):
        return False
    for sublist in hashtags_nested:
        for tag in sublist:
            tag_upper = str(tag).upper()
            if tag_upper in keywords_upper or any(kw in tag_upper for kw in keywords_upper):
                return True
    return False


class NewsFinData():
    """
    Класс для обработки и анализа финансовых новостей. 
    """
    def __init__(self, news_df, column_names=None):
        if isinstance(news_df, pd.DataFrame):
            self.news_df = news_df.copy()
        else:
            self.news_df = pd.read_csv(news_df)

        # Переименование колонок, если указано
        if column_names:
            self.news_df.rename(columns=column_names, inplace=True)

        # убираем информацию о часовом поясе 
        self.news_df['Date'] = self.news_df['Date'].str.replace(r'\+00:00', '', regex=True)
        self.news_df['Date'] = pd.to_datetime(self.news_df['Date'], errors='coerce')
        self.news_df = self.news_df.dropna(subset=['Text', 'list_of_hashtags']).reset_index(drop=True)
        # из списка строк в список списков
        self.news_df['list_of_hashtags'] = self.news_df['list_of_hashtags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        self.cand_news_df = None

    def _round_time_up(self, dt: dt.datetime, 
                       interval_minutes: int) -> dt.datetime:
        """
        Округляет datetime вверх до ближайшего интервала.
        """
        seconds = (dt - dt.replace(second=0, microsecond=0)).seconds
        remainder = seconds % (interval_minutes * 60)
        if remainder != 0:
            dt += timedelta(seconds=(interval_minutes * 60 - remainder))
        dt = dt.replace(second=0, microsecond=0)
        return dt

    def adjust_time_to_candles(self, 
                            candle_int_min: int = 1,
                            start_cd_time: time = time(7, 0),
                            end_cd_time: time = time(20, 0)):
        """
        Корректирует время новостей по заданному интервалу свечей.
        Новости вне дневного окна (между end_cd_time и start_cd_time) 
        сдвигаются на ближайшее дневное время.
        
        Parameters:
        - candle_int_min: интервал свечи в минутах
        - start_cd_time: начало дневного времени (по умолчанию 07:00)
        - end_cd_time: конец дневного времени (по умолчанию 20:00)
        """
        def adjust(row):
            datte = row['Date']
            datte = self._round_time_up(datte, candle_int_min)

            # Если время >= вечернего порога — перенос на утро следующего дня
            if datte.time() >= end_cd_time:
                datte = (datte + timedelta(days=1)).replace(
                    hour=start_cd_time.hour, minute=start_cd_time.minute, second=0, microsecond=0
                )
            # Если время < утреннего порога — перенос на утро текущего дня
            elif datte.time() < start_cd_time:
                datte = datte.replace(
                    hour=start_cd_time.hour, minute=start_cd_time.minute, second=0, microsecond=0
                )
            return datte

        self.news_df['AdjustedTime'] = self.news_df.apply(adjust, axis=1)
        return self.news_df
    
    def filter_by_datetime_range(self,
                                column: str = 'AdjustedTime',
                                start_datetime: dt.datetime = None,
                                end_datetime: dt.datetime = None):
        """
        Фильтрует строки датафрейма по диапазону дат и времени на основе указанной колонки.

        Parameters:
        - column: имя колонки с datetime (по умолчанию 'AdjustedTime')
        - start_datetime: нижняя граница включительно (если None — без нижней границы)
        - end_datetime: верхняя граница включительно (если None — без верхней границы)
        """
        if column not in self.news_df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")

        if start_datetime is not None:
            self.news_df = self.news_df[self.news_df[column] >= start_datetime]
        
        if end_datetime is not None:
            self.news_df = self.news_df[self.news_df[column] <= end_datetime]

        return self.news_df
    
    def merge_news_with_df(self, candles_df, merge_row_df : str, 
                           merge_row_news = 'AdjustedTime', 
                           timedel = pd.Timedelta(minutes=1)):
        df = candles_df.copy()
        n = len(df)
        df['news'] = [[] for _ in range(n)]
        df['dates'] = [[] for _ in range(n)]
        df['hashtags'] = [[] for _ in range(n)]

        # Устанавливаем временной индекс один раз
        df.set_index(merge_row_df, inplace=True)
        df = df.sort_index()

        for idx, news_row in self.news_df.iterrows():
            news_time = news_row[merge_row_news]

            # Ищем ближайшую свечу выше по времени
            while news_time not in df.index:
                news_time += timedel
                if news_time > df.index.max():
                    break

            if news_time in df.index:
                df.at[news_time, 'news'].append(news_row['Text'])
                df.at[news_time, 'dates'].append(news_row['Date'])
                df.at[news_time, 'hashtags'].append(news_row['list_of_hashtags'])

        df.reset_index(inplace=True) 
        self.cand_news_df = df
        return self.cand_news_df
    
    def compute_price_changes_all(self, steps=[30, 60, 120], col='close'):
        """
        Считает процентное изменение по столбцу col через steps свечей для всех строк.
        И обновляет self.cand_news_df.

        Parameters:
        - steps : список целых чисел — окна (в свечах), через которые считать изменение
        - col : имя колонки с ценой (по умолчанию 'close')

        Returns:
        - self.cand_news_df с новыми колонками: pct_change_{step}
        """
        if self.cand_news_df is None:
            raise ValueError("Атрибут self.cand_news_df не инициализирован. "
                            "Сначала объедините новости и свечи.")

        df = self.cand_news_df.copy()

        for step in steps:
            future_price = df[col].shift(-step)
            df[f'pct_change_{step}'] = (future_price - df[col]) / df[col] * 100

        self.cand_news_df = df
        return self.cand_news_df


    def compute_hashtag_influence(self, change_col='pct_change_10'):
        """
        Считает среднее абсолютное влияние на цену закрытия для каждого уникального хэштега.

        Params:
        - change_col : имя колонки с процентным изменением цены (по умолчанию 'pct_change_10')

        Returns: 
        - DataFrame с колонками: hashtag, mean_abs_influence, count
        """
        if self.cand_news_df is None:
            raise ValueError("Атрибут self.cand_news_df не инициализирован. "
                            "Сначала объедините новости и свечи.")
        

        hashtag_to_changes = defaultdict(list)

        # Только строки, где значение изменения определено
        filtered_df = self.cand_news_df[self.cand_news_df[change_col].notna()]

        for _, row in filtered_df.iterrows():
            pct_change = abs(row[change_col])
            all_hashtags = set(tag for sublist in row["hashtags"] for tag in sublist)

            for tag in all_hashtags:
                hashtag_to_changes[tag].append(pct_change)

        # Преобразуем словарь в таблицу
        stats = pd.DataFrame([
            {
                "hashtag": tag,
                "mean_abs_influence": np.mean(changes),
                "counts": len(changes)
            }
            for tag, changes in hashtag_to_changes.items()
        ])

        # Сортируем по убыванию влияния
        stats = stats.sort_values("mean_abs_influence", ascending=False).reset_index(drop=True)
        return stats
    
    def add_imp_tags(self, imp_tags : list):
        keywords_upper = [kw.upper() for kw in imp_tags]
        if self.cand_news_df is None:
            raise ValueError("Атрибут self.cand_news_df не инициализирован. "
                            "Сначала объедините новости и свечи.")
        self.cand_news_df['has_tags'] = self.cand_news_df['hashtags'].apply(lambda tags: contains_keywords(tags, keywords_upper))

    def add_news_labels(self, threshold=1.0, steps=[30, 60], label_col="news_label"):
        """
        Добавляет колонку с метками изменений:
        -1 — сильное падение
        0 — нейтральное изменение
        1 — сильный рост

        Parameters:
        - threshold : порог в процентах
        - steps : список шагов в свечах
        - label_col : имя создаваемой колонки
        """
        def classify(row):
            for step in steps:
                val = row.get(f"pct_change_{step}", None)
                if pd.isna(val):
                    continue
                if val > threshold:
                    return 1   # рост
                elif val < -threshold:
                    return -1  # падение
            return 0           # нейтрально

        self.cand_news_df[label_col] = self.cand_news_df.apply(classify, axis=1)
        return self.cand_news_df



    def propagate_news_labels(self, label_col="news_label", forward_col="forward_label", time_col="time_from_new"):
        """
        Распространяет метку новости на следующие строки, пока не встретится следующая новость.
        
        Parameters:
        - label_col : колонка с меткой новости
        - forward_col : новая колонка для размеченных вперёд меток
        - time_col : колонка расстояния от последней новости
        """
        df = self.cand_news_df.copy()
        df[forward_col] = None
        df[time_col] = None

        n = len(df)
        i = 0

        while i < n:
            if df.iloc[i]["has_tags"]:
                label = df.iloc[i][label_col]
                df.at[df.index[i], forward_col] = label
                df.at[df.index[i], time_col] = 0
                offset = 1
                j = i + 1
                while j < n and not df.iloc[j]["has_tags"]:
                    df.at[df.index[j], forward_col] = label
                    df.at[df.index[j], time_col] = offset
                    offset += 1
                    j += 1
                i = j
            else:
                i += 1

        self.cand_news_df = df
        return self.cand_news_df


    
    







    




