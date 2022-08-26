import sys

import pandas as pd

sys.path.append("../..")  # noqa

import os
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, as_completed
from multiprocessing import cpu_count

import numpy as np
from imutils import paths
from tqdm import tqdm

from stock_market_simulator import Broker

TYPES = ["buy", "sell", "hold"]
VAL_YEARS = [2021]
TEST_YEARS = [2022]


def run_broker(path, window_size, tp_factor, sl_factor):
    """
    For each simulated day, a buy and sell position will be opened, this function will run e many stock symbols.
    This function will return the window with the data when the position was opened and also return all position
    history. With the position history it is possible to know the trade time, the profit, etc.

    :param path: [str] the csv path with the data
    :param window_size: [int] the past days that you wish to look.
    :param tp_factor: [float] the reference for this factor will be the mean difference between max and min from all
        candles inside of the window size.
    :param sl_factor: [float] the reference for this factor will be the mean difference between max and min from all
        candles inside of the window size.
    :return: [tuple[list[pandas.Dataframe], list[b3_predictor.Position]]]
    """
    broker = Broker(data_frame=path, window_size=window_size, ignore_spread=True)
    broker.add_indicator("close_21_sma")
    broker.add_indicator("macd")

    ticker = os.path.basename(path)
    ticker = os.path.splitext(ticker)[0]
    ticker = ticker.split("_")[0]

    history = []
    for _data in broker:
        mean = abs(_data[:, broker.columns["low"]] - _data[:, broker.columns["high"]]).mean()
        tp = mean * tp_factor
        sl = mean * sl_factor
        broker.buy(0, broker.ask - sl, broker.ask + tp)
        broker.sell(0, broker.bid + sl, broker.bid - tp)
        positions = broker.opened_positions[-2:]

        _data = pd.DataFrame(_data, columns=list(broker.columns.keys()))
        history.append({"data": _data.copy(),
                        "buy_id": positions[0].ticket,
                        "sell_id": positions[1].ticket,
                        "ticker": ticker})
    broker.close_all_positions()
    history_positions = broker.history_positions
    for _h in history_positions:
        _h.ticker = ticker
    return history, history_positions


def process_data(input_data, position_buy, position_sell, ticker, _i):
    """
    Label all data generate by the run_broker function.
    This function will label and save the data in folders "Train", "Val" and "Test".

    :param input_data: [pandas.Dataframe] the df generate by the WINDOW_SIZE
    :param position_buy: [Position] the buy position opened for this window
    :param position_sell: [Position] the sell position opened for this window
    :param ticker: [str] The ticker symbol name
    :param _i: [int] Index value that will be saved in the name of the csv.
    :return: None
    """
    # Check what will be the label for this window
    if position_buy.profit < 0 and position_sell.profit < 0:
        if position_buy.close_time > position_sell.close_time:
            selected_position = position_buy
        else:
            selected_position = position_sell
        output_data = 2
    elif position_buy.profit > 0:
        selected_position = position_buy
        output_data = 0
    else:
        selected_position = position_sell
        output_data = 1

    # Generate simple metrics that will be saved with the csv
    input_data.insert(len(input_data.columns), "label", output_data)
    input_data.insert(len(input_data.columns), "symbol", ticker)
    input_data.insert(len(input_data.columns), "open_time", selected_position.open_time)
    input_data.insert(len(input_data.columns), "close_time", selected_position.close_time)
    timedelta = np.timedelta64(selected_position.close_time - selected_position.open_time, "D").astype("int")
    input_data.insert(len(input_data.columns), "timedelta_days", timedelta)
    input_data.insert(len(input_data.columns), "open_price", selected_position.opened_price)
    input_data.insert(len(input_data.columns), "take_profit", selected_position.open_order.tp)
    input_data.insert(len(input_data.columns), "stop_loss", selected_position.open_order.sl)
    input_data.insert(len(input_data.columns), "profit", f"{selected_position.profit:.02f}")
    input_data.insert(len(input_data.columns), "profit_percent",
                      f"{selected_position.profit / selected_position.opened_price:.02%}")

    # apply some rule for filtering the generated data
    if 0 < float(input_data["profit_percent"].values[0][:-1]) < 0.5 or timedelta > 10:
        output_data = 2
        input_data["label"] = output_data

    # check if the data is train, val and test type
    if input_data["time"].loc[0].year in VAL_YEARS:
        sample_type = "val"
    elif input_data["time"].loc[0].year in TEST_YEARS:
        sample_type = "test"
    else:
        sample_type = "train"
    dst = f"dataset/{sample_type}/{TYPES[output_data]}/{_i:04d}.csv"
    input_data.to_csv(dst, index=False)


if __name__ == "__main__":
    import time

    start = time.time()
    DATA_PATH_LIST = list(paths.list_files("../../data/", ".csv"))
    WINDOW_SIZE = 40
    limit = cpu_count()

    executor = ProcessPoolExecutor(max_workers=limit)
    futures = set()
    all_futures = []

    # Just working for TP > SL
    TAKE_PROFIT_FACTOR = 2
    STOP_LOSS_FACTOR = 1

    # create the folder where the data will be saved.
    for folder_type in ["train", "val", "test"]:
        for folder_label in TYPES:
            os.makedirs(f"dataset/{folder_type}/{folder_label}", exist_ok=True)

    # Execute the broker using subprocess for parallelizing it
    _history = []
    _position_history = []
    for data_path in DATA_PATH_LIST:
        all_futures.append(executor.submit(run_broker, data_path, WINDOW_SIZE, TAKE_PROFIT_FACTOR, STOP_LOSS_FACTOR))

    for future in tqdm(as_completed(all_futures), total=len(all_futures)):
        h, ph = future.result()
        _history += h
        _position_history += ph

    # process all data produced using subprocess again for getting speed
    for i, data in enumerate(tqdm(_history)):
        buy_id = data["buy_id"]
        sell_id = data["sell_id"]
        _input_data = data["data"]
        _ticker = data["ticker"]
        p_buy = next(p for p in _position_history if p.ticket == buy_id and p.ticker == _ticker)
        p_sell = next(p for p in _position_history if p.ticket == sell_id and p.ticker == _ticker)
        _position_history.remove(p_buy)
        _position_history.remove(p_sell)

        if len(futures) >= limit:
            completed, futures = wait(futures, return_when=FIRST_COMPLETED)
        futures.add(executor.submit(process_data, _input_data, p_buy, p_sell, _ticker, i))
    print(time.time() - start)
