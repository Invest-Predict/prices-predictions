from datetime import timedelta, time
import matplotlib.pyplot as plt

import plotly.express as px
import pandas as pd
import pandas as pd
import numpy as np

def adjust_news_time(row):
    # округляем время вверх
    dt = row['Date']
    # Округляем секунды вверх до ближайшей минуты
    if dt.second > 0 or dt.microsecond > 0:
        dt = dt + timedelta(minutes=1)
    dt = dt.replace(second=0, microsecond=0)
    # Если время после 20:00, переносим на следующий день 7:00
    if dt.time() >= time(20, 0):
        dt = (dt + timedelta(days=1)).replace(hour=7, minute=0)
    # Если время между 00:00 и 07:00, ставим 7:00 текущего дня
    elif time(0, 0) <= dt.time() < time(7, 0):
        dt = dt.replace(hour=7, minute=0)
    return dt

def add_news_to_df(df, news_df, merge_row_df : str, merge_row_news : str):
    df = df.copy()
    df['news'] = [[] for _ in range(len(df))]
    df['dates'] = [[] for _ in range(len(df))]
    df['hashtags'] = [[] for _ in range(len(df))]

    # Устанавливаем временной индекс один раз
    df.set_index(merge_row_df, inplace=True)
    df = df.sort_index()

    for idx, news_row in news_df.iterrows():
        news_time = news_row[merge_row_news]

        # Ищем ближайшую свечу выше по времени
        while news_time not in df.index:
            news_time += pd.Timedelta(minutes=1)
            if news_time > df.index.max():
                break

        if news_time in df.index:
            df.at[news_time, 'news'].append(news_row['Text'])
            df.at[news_time, 'dates'].append(news_row['Date'])
            df.at[news_time, 'hashtags'].append(news_row['list_of_hashtags'])

    df.reset_index(inplace=True) 
    return df

def contains_keywords(hashtags_nested, keywords_upper):
    if not isinstance(hashtags_nested, list):
        return False
    for sublist in hashtags_nested:
        for tag in sublist:
            tag_upper = str(tag).upper()
            if tag_upper in keywords_upper or any(kw in tag_upper for kw in keywords_upper):
                return True
    return False

def compute_price_changes(df, steps=[1, 5, 10], col='close'):
    """
    Для строк с has_tags=True считает процентное изменение close через steps свечей.

    Параметры:
    df : DataFrame с колонками 'has_tags' и 'close'
    steps : список целых чисел — через сколько свечей считать изменение
    col : название колонки с ценой (по умолчанию 'close')

    Возвращает:
    df — DataFrame с новыми колонками: pct_change_{step}
    """
    df = df.copy()
    
    # Инициализируем колонки значениями NaN
    for step in steps:
        df[f'pct_change_{step}'] = np.nan

    # Получаем индексы строк, где есть новости
    tagged_indices = df.index[df['has_tags']].tolist()

    # Считаем изменение цены через нужное количество шагов
    for idx in tagged_indices:
        current_price = df.at[idx, col]
        for step in steps:
            future_idx = idx + step
            if future_idx < len(df):
                future_price = df.at[future_idx, col]
                df.at[idx, f'pct_change_{step}'] = (future_price - current_price) / current_price * 100

    return df


def compute_price_changes_all(df, steps=[1, 5, 10], col='close'):
    """
    Считает процентное изменение цены 'close' через steps свечей для всех строк.

    Параметры:
    - df : pd.DataFrame — должен содержать колонку 'close'
    - steps : список целых чисел — окна (в свечах), через которые считать изменение
    - col : имя колонки с ценой (по умолчанию 'close')

    Возвращает:
    - копию df с новыми колонками: pct_change_{step}
    """
    df = df.copy()
    
    for step in steps:
        # Сдвигаем цену на -step вперёд, чтобы получить future_price
        future_price = df[col].shift(-step)
        df[f'pct_change_{step}'] = (future_price - df[col]) / df[col] * 100

    return df



def get_significant_news(df, threshold=1.0, steps=[1, 5, 10]):
    """
    Возвращает строки с has_tags == True и изменением цены выше порога по любому из указанных окон.

    Параметры:
    - df: DataFrame с колонками pct_change_{step}
    - threshold: порог (в процентах), выше которого изменение считается значимым
    - steps: список шагов (например, [1, 5, 10])

    Возвращает:
    - DataFrame только с новостями и превышением порога
    """
    df_filtered = df[df["has_tags"]].copy()
    mask = np.zeros(len(df_filtered), dtype=bool)
    
    for step in steps:
        mask |= df_filtered[f"pct_change_{step}"].abs() > threshold
    
    return df_filtered[mask]



def plot_news_points(df, threshold=1.0, steps=[1, 5, 10]):
    """
    Визуализирует закрытие и новости цветом:
    - красный: сильное падение
    - зелёный: сильный рост
    - серый: нейтральное изменение
    """
    df_all = df.copy()  # для линии закрытия
    df_news = df[df["has_tags"]].copy()  # только строки с новостями

    # Классификация
    def classify(row):
        for step in steps:
            val = row.get(f"pct_change_{step}", None)
            if pd.isna(val):
                continue
            if val > threshold:
                return "green"
            elif val < -threshold:
                return "red"
        return "gray"

    df_news["color"] = df_news.apply(classify, axis=1)

    # Построим график
    plt.figure(figsize=(15, 6))
    plt.plot(df_all["utc"], df_all["close"], label="Close Price", color='lightblue')

    # Наложим точки
    for color in ["red", "gray", "green"]:
        subset = df_news[df_news["color"] == color]
        plt.scatter(subset["utc"], subset["close"], label=color.capitalize(), color=color)

    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.title(f"News Points with Threshold {threshold}%")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_interactive_news(df, threshold=1.0, steps=[1, 5, 10]):
    """
    Создаёт интерактивный график с новостями, где при наведении видна дата и текст новости.
    Цвета точек:
    - зелёный: рост
    - красный: падение
    - серый: в пределах порога
    """
    df_plot = df[df["has_tags"]].copy()

    # Классификация по порогу
    def classify(row):
        for step in steps:
            val = row[f"pct_change_{step}"]
            if pd.isna(val):
                continue
            if val > threshold:
                return "green"
            elif val < -threshold:
                return "red"
        return "gray"

    df_plot["color"] = df_plot.apply(classify, axis=1)
    df_plot["hover_text"] = df_plot.apply(
        lambda row: "<br>".join([
            f"<b>Date:</b> {row['dates'][0] if row['dates'] else row['utc']}",
            f"<b>News:</b> {row['news'][0][:300]}..." if row['news'] else "No news"
        ]), axis=1
    )

    fig = px.scatter(
        df_plot,
        x="utc",
        y="close",
        color="color",
        hover_name="hover_text",
        title=f"Significant News Events (Threshold = {threshold}%)",
        color_discrete_map={"red": "red", "green": "green", "gray": "gray"}
    )

    fig.add_scatter(
        x=df["utc"],
        y=df["close"],
        mode='lines',
        name='Close Price',
        line=dict(color='lightblue')
    )

    fig.update_layout(
        hoverlabel=dict(bgcolor="white", font_size=12),
        xaxis_title="Time",
        yaxis_title="Close Price",
        legend_title="Signal"
    )

    fig.show()
