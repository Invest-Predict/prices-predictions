from tinkoff.invest.sandbox.async_client import AsyncSandboxClient
from tinkoff.invest.schemas import OrderDirection, OrderType, MoneyValue, CandleSource
from tinkoff.invest.utils import now
from tinkoff.invest import CandleInterval
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import time
import logging
from uuid import uuid4
import time
import pandas as pd
import os

import sys
sys.path.append('..')

from model import FinData
from model import CatboostFinModel
from model import train_valid_split_candles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sand_trading.log"),
        logging.StreamHandler()
    ]
)

load_dotenv()
ttoken = os.getenv("T_TOKEN")

TOKEN = ttoken

# вспомогательные функции для работы торгов 
def convert_price(value : dict | MoneyValue):
    return value['units'] + value['nano'] / 1e9 if type(value) == dict else value.units + value.nano / 1e9

def wait_until_next_x1_multiple(x = 10):
    """Ждет до ближайшей минуты, которая кратна x."""
    now = datetime.now()
    next_minute = (now.minute // x + 1) * x  # Следующая X1-кратная минута
    next_time = now.replace(minute=next_minute % 60, second=0, microsecond=0)

    # Если перешли через час
    if next_minute >= 60:
        next_time += timedelta(hours=1)

    wait_seconds = (next_time - now).total_seconds()
    logging.info(f"Ждем {wait_seconds:.2f} секунд до {next_time.strftime('%H:%M:%S')}...")
    time.sleep(wait_seconds)

def make_features(data : FinData, features_settings : dict):
    # Attention: inplace modification
    # Эту нужно будет доделать...
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

async def get_accounts():
# возвращает все активные аккаунты 
    async with AsyncSandboxClient(TOKEN) as client:
        return (await client.users.get_accounts()).accounts
    
async def get_money_on_acc(acc_id):
    async with AsyncSandboxClient(TOKEN) as client:
        portfolio = await client.operations.get_portfolio(account_id=acc_id)
    return convert_price(portfolio.total_amount_currencies)
    

    
async def open_account(name : str, money_value = MoneyValue(currency="643", units=10000, nano=0)):
# создает аккаунт и пополняет его на указанную сумму денег
# торги в песочнице происходят только в рублях, код рубля "643"
# количество денег если хотите поменять, надо указывать как по дефолту
# units - целая часть суммы, nano - неинтутивно, нецелая часть суммы, которая считается как nano/1e9
# то есть итоговая сумма, которую вы добавите будет типо units + nano/1e9 = units,(nano/1e9)
    async with AsyncSandboxClient(TOKEN) as client:
        acc_id = (await client.sandbox.open_sandbox_account(name=name)).account_id
        response = await client.sandbox.sandbox_pay_in(account_id=acc_id, amount=money_value)
    return acc_id, response

async def close_all_accs():
# закрыть все активные аккаунты 
    async with AsyncSandboxClient(TOKEN) as client:
        accs = await get_accounts()
        for acc in accs:
            await client.sandbox.close_sandbox_account(account_id=acc.id)

async def find_instrument(query : str):
# найти инструменты по любому запросу, я советую сюда тикер пихать
    async with AsyncSandboxClient(TOKEN) as client:
        response = await client.instruments.find_instrument(query=query)
    return response.instruments

async def get_candles(figi : str, start_date : datetime | None = None, last_candle : bool = False):
    async with AsyncSandboxClient(TOKEN) as client:
        td = datetime.now() - start_date if not last_candle else timedelta(minutes=10)

        candles = [candle async for candle in client.get_all_candles(
                                                                        instrument_id=figi,
                                                                        from_=now() - td,
                                                                        interval=CandleInterval.CANDLE_INTERVAL_10_MIN,
                                                                        candle_source_type=CandleSource.CANDLE_SOURCE_UNSPECIFIED
                                                                    )]

    candles = pd.DataFrame(candles)
    candles = candles if not last_candle else candles.iloc[0].to_frame().T
    candles = candles.rename(columns={'time': 'utc'})
    for col in ['open', 'high', 'low', 'close']:
                candles[col] = candles[col].apply(convert_price)

    return candles 

async def make_order(order_dir, figi, lot, acc_id, order_type = OrderType.ORDER_TYPE_MARKET):
    async with AsyncSandboxClient(TOKEN) as client:
        response = await client.orders.post_order(order_type=order_type,
                                                  direction=order_dir,
                                                  instrument_id=figi,
                                                  quantity=lot,
                                                  account_id=acc_id,
                                                  order_id=str(uuid4())
                                                  )
    return response



async def make_data_for_trading(share_id, start_date, features_sets):
    logging.info("Начало загрузки данных для обучения модели")
    candles = await get_candles(figi=share_id, start_date=start_date)
    clear_data_df = candles[:-1]
    data = FinData(clear_data_df)
    make_features(data, features_sets)
    logging.info(f"Собрано {data.df.shape[0]} свечей")
    logging.info("Данные успешно загружены и обработаны")
    return clear_data_df, data, data.get_numeric_features(), data.get_cat_features(), data.target

async def model_train(data, num, cat, target, args, train_size, val_size):
    logging.info("Начало обучения модели")
    X_train, X_val, y_train, y_val = train_valid_split_candles(data.df, 
                                                               train_size=train_size, 
                                                               val_size=val_size, 
                                                               numeric=num, 
                                                               cat=cat, 
                                                               target=target)
    model = CatboostFinModel(args=args)
    model.set_datasets(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
    model.set_features(numeric_features=num, cat_features=cat)
    model.fit()
    logging.info("Обучение модели завершено")
    return model


async def make_trading(model, clear_data_df : pd.DataFrame, 
                        end_datetime : datetime, 
                        lot, 
                        figi,
                        features_sets, 
                        acc_name = "lerochka",
                        money = MoneyValue(currency="643", units=10000, nano=0)):
    
    logging.info("Запуск торгов в песочнице")
    acc_id, resp = await open_account(name=acc_name, money_value=money)
    logging.info(f"Создан аккаунт {acc_id}, пополнен на {convert_price(money)}")    
    portfolio_shares = []
    logging.info(f"Торги начинаются: {datetime.now()} / Завершение: {end_datetime}")
    while datetime.now() < end_datetime:
        wait_until_next_x1_multiple(10)
        logging.info("Начинаем новый торговый цикл")
        for share in portfolio_shares:
            if share["dur"] == "buy":
                response = await make_order(order_dir=OrderDirection.ORDER_DIRECTION_SELL, 
                                            figi = share["inst_id"], 
                                            lot = share["quant"], 
                                            acc_id = acc_id)
                
                logging.info(f"Продана акция: {response}")

            else:
                response = await make_order(order_dir=OrderDirection.ORDER_DIRECTION_BUY, 
                                            figi = share["inst_id"], 
                                            lot = share["quant"], 
                                            acc_id = acc_id)
                
                logging.info(f"Куплена акция: {response}")

            
        logging.info(await get_money_on_acc(acc_id))
        portfolio_shares.clear()
        time.sleep(0.1) # вот тут вопросик по задержкам поступления информации про свечу 
        curr_candle = await get_candles(figi = figi, last_candle=True)
                
        clear_data_df = pd.concat([clear_data_df, curr_candle], ignore_index=True)
        data = FinData(clear_data_df)
        make_features(data, features_sets)
        num = data.get_numeric_features()
        cat = data.get_cat_features()
        x_curr = data.df.tail(1)[num + cat]
        y_pred = model.predict(x_curr)  
        logging.info(f"Предсказание модели: {y_pred}") 
        money_am = await get_money_on_acc(acc_id) 
        inst_am = int(money_am // data.df.tail(1).close.item())
        if y_pred == 1:
            
            response = await make_order(order_dir=OrderDirection.ORDER_DIRECTION_BUY, 
                                        figi = figi, 
                                        lot = inst_am, 
                                        acc_id = acc_id)
            portfolio_shares.append({"inst_id" : figi, "quant" : inst_am, "dur" : "buy"})
            logging.info(f"Куплена акция: {response}")

        if y_pred == 0:
            response = await make_order(order_dir=OrderDirection.ORDER_DIRECTION_SELL, 
                                        figi = figi, 
                                        lot = inst_am, 
                                        acc_id = acc_id)
            portfolio_shares.append({"inst_id" : figi, "quant" : inst_am, "dur" : "sell"})
            logging.info(f"Продана акция: {response}")
    return acc_id

async def trade(figi, lot,  args, start_date, end_date, features_sets, train_size, val_size, close_acc = False, 
                acc_name = "lerochka", money = MoneyValue(currency="643", units=10000, nano=0)):
    
    clear_data_df, data, num, cat, target = await make_data_for_trading(figi, start_date, features_sets)

    model = await model_train(data, num, cat, target, args, train_size, val_size)

    acc_id = await make_trading(model, clear_data_df, end_date, lot, figi, features_sets, acc_name, money)  

    async with AsyncSandboxClient(TOKEN) as client:
        portfolio = await client.operations.get_portfolio(account_id=acc_id)
        if close_acc:
            await client.sandbox.close_sandbox_account(account_id=acc_id)

    logging.info(f"Торги завершены с окончательной суммой {portfolio.total_amount_currencies}")
    if close_acc:
        logging.info("Аккаунт закрыт")

    return portfolio
     



                
    
    


    

    
