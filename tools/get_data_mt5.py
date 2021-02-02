import json
import os
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd
from tqdm import tqdm

with open("../.env_conf.json") as json_file:
    config_dict = json.loads(json_file.read())["get_data_params"]

os.makedirs("../data", exist_ok=True)
if not mt5.initialize(login=config_dict["user"], server="ModalMais-DMA4 - Beta", password=config_dict["password"]):
    print("initialize() failed, error code =", mt5.last_error())
    quit(1)

with open("tickers.json") as json_file:
    json_dict = json.loads(json_file.read())

string_to_timeframe = {"m1": mt5.TIMEFRAME_M1, "m2": mt5.TIMEFRAME_M2, "m3": mt5.TIMEFRAME_M3, "m4": mt5.TIMEFRAME_M4,
                       "m5": mt5.TIMEFRAME_M5, "m6": mt5.TIMEFRAME_M6, "m10": mt5.TIMEFRAME_M10,
                       "m12": mt5.TIMEFRAME_M12, "m15": mt5.TIMEFRAME_M15, "m20": mt5.TIMEFRAME_M20,
                       "m30": mt5.TIMEFRAME_M30, "h1": mt5.TIMEFRAME_H1, "h2": mt5.TIMEFRAME_H2, "h3": mt5.TIMEFRAME_H3,
                       "h4": mt5.TIMEFRAME_H4, "h6": mt5.TIMEFRAME_H6, "h8": mt5.TIMEFRAME_H8, "h12": mt5.TIMEFRAME_H12,
                       "d1": mt5.TIMEFRAME_D1, "w1": mt5.TIMEFRAME_W1, "mn1": mt5.TIMEFRAME_MN1}
timeframe = json_dict["timeframe"]
utc_from = datetime.now()
for ticker in tqdm(json_dict["tickers"]):
    try:
        rates = mt5.copy_rates_from(ticker, string_to_timeframe[timeframe], utc_from, 9999999)

        rates_frame = pd.DataFrame(rates)
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
        rates_frame = rates_frame.set_index("time")
        rates_frame.to_csv(f"../data/{ticker}_{timeframe}.csv")
    except KeyError as e:
        print(f"Error type: {type(e)}")
        print("Try to set unlimited bars on mt5 platform.")
    except Exception as e:
        print(f"Error on ticker: {ticker}; error: {type(e)}{e}")

mt5.shutdown()
