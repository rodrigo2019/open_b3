import sys

sys.path.append("..")  # noqa

import os

import numpy as np
import pandas as pd
from imutils import paths
from tqdm import tqdm

from b3_predictor import Broker

DATA_PATH_LIST = list(paths.list_files("../data/", ".csv"))
WINDOWS_SIZE = 40

# Just working for TP > SL
TAKE_PROFIT_FACTOR = 2
STOP_LOSS_FACTOR = 1

types = ["buy", "sell", "hold"]
os.makedirs("dataset/raw/buy", exist_ok=True)
os.makedirs("dataset/raw/sell", exist_ok=True)
os.makedirs("dataset/raw/hold", exist_ok=True)

history = []
position_history = []
for data_path in tqdm(DATA_PATH_LIST, position=0):
    broker = Broker(csv_file_path=data_path, window_size=WINDOWS_SIZE, ignore_spread=True)
    ticker = os.path.basename(data_path)
    ticker = os.path.splitext(ticker)[0]
    ticker = ticker.split("_")[0]

    input_data = None
    output_data = None
    get_new_data = True

    for data in broker:
        if get_new_data:
            input_data = data
            mean = abs(data["low"] - data["high"]).values.mean()
            tp = mean * TAKE_PROFIT_FACTOR
            sl = mean * STOP_LOSS_FACTOR
            broker.buy(0, broker.ask - sl, broker.ask + tp)
            broker.sell(0, broker.bid + sl, broker.bid - tp)

            get_new_data = False
        elif len(broker.opened_positions) == 0:
            p = broker.history_positions[-1]
            output_data = p.open_order.order_type
            if broker.history_positions[-1].profit < 0:
                output_data = 2
            input_data.insert(len(input_data.columns), "label", [output_data] * len(input_data))
            input_data.insert(len(input_data.columns), "symbol", [ticker] * len(input_data))
            input_data.insert(len(input_data.columns), "open_time", [p.open_time] * len(input_data))
            input_data.insert(len(input_data.columns), "close_time", [p.close_time] * len(input_data))
            timedelta = np.timedelta64(p.close_time - p.open_time, "D").astype("int")
            input_data.insert(len(input_data.columns), "timedelta_days", [timedelta] * len(input_data))
            input_data.insert(len(input_data.columns), "open_price", [p.opened_price] * len(input_data))
            input_data.insert(len(input_data.columns), "take_profit", [p.open_order.tp] * len(input_data))
            input_data.insert(len(input_data.columns), "stop_loss", [p.open_order.sl] * len(input_data))
            input_data.insert(len(input_data.columns), "profit", [f"{p.profit:.02f}"] * len(input_data))
            input_data.insert(len(input_data.columns), "profit_percent",
                              [f"{p.profit / p.opened_price:.02%}"] * len(input_data))
            history.append({"input": input_data, "output": output_data})
            broker.close_all_positions()

            get_new_data = True
    added_ticker = broker.history_positions
    for p in added_ticker:
        p.ticker = ticker
    position_history += added_ticker

samples_count = {}
for i, pack_data in enumerate(tqdm(history)):
    data = pack_data["input"]
    label = str(int(pack_data["output"]))
    if label not in samples_count:
        samples_count[label] = 1
    else:
        samples_count[label] += 1
    data.to_csv(f"dataset/raw/{types[int(label)]}/{i:04d}.csv")
print(samples_count)

position_df = {"open_time": [], "close_time": [], "timedelta": [], "opened_price": [], "closed_price": [],
               "take_profit": [], "gain_percentage": [], "loss_percentage": [], "stop_loss": [], "position_type": [],
               "profit": [], "ticker": []}
for p in position_history:
    timedelta = np.timedelta64(p.close_time - p.open_time, "D").astype("int")
    position_df["open_time"].append(p.open_time)
    position_df["close_time"].append(p.close_time)
    position_df["timedelta"].append(timedelta)
    position_df["opened_price"].append(f"{p.opened_price:.02f}")
    position_df["closed_price"].append(f"{p.last_price:.02f}")
    position_df["take_profit"].append(f"{p.open_order.tp:.02f}")
    if p.open_order.order_type == 0:  # buy
        position_df["gain_percentage"].append(f"{p.open_order.tp / p.opened_price - 1:.02%}")
        position_df["loss_percentage"].append(f"{1 - p.open_order.sl / p.opened_price:.02%}")
    else:
        position_df["gain_percentage"].append(f"{p.opened_price / p.open_order.tp - 1:.02%}")
        position_df["loss_percentage"].append(f"{1 - p.opened_price / p.open_order.sl:.02%}")
    position_df["stop_loss"].append(f"{p.open_order.sl:.02f}")
    position_df["position_type"].append("buy" if p.open_order.order_type == 0 else "sell")
    position_df["profit"].append(f"{p.profit:02f}")
    position_df["ticker"].append(p.ticker)

df = pd.DataFrame(position_df)
df.index.name = "index"
df.to_csv("debug.csv")
