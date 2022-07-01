from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go

AMPLITUDE = 10
OFFSET = 10
FREQUENCY = 2
NOISE_AMPLITUDE = 2
HIGH_LOW_NOISE_AMPLITUDE = 0.5
GAP_NOISE = 0.5
START_DATE = datetime(1970, 1, 1)
TIME_LENGTH_DAYS = 10_000
RANDOM_SEED = 1

df = {"time": [], "open": [], "high": [], "low": [], "close": [], "tick_volume": [], "spread": [], "real_volume": []}
time = np.arange(0, (TIME_LENGTH_DAYS + 1) / 10, 0.1)

np.random.seed(RANDOM_SEED)
open_price = np.sin(FREQUENCY * time) * AMPLITUDE / 2 + OFFSET
open_price -= np.random.rand(len(open_price)) * NOISE_AMPLITUDE - NOISE_AMPLITUDE / 2
close_price = open_price[1:].copy()
close_price += (np.random.rand(TIME_LENGTH_DAYS) * GAP_NOISE - GAP_NOISE / 2)
open_price = open_price[:-1]

high_price = np.maximum(open_price, close_price) + np.random.rand(TIME_LENGTH_DAYS) * HIGH_LOW_NOISE_AMPLITUDE
low_price = np.minimum(open_price, close_price) - np.random.rand(TIME_LENGTH_DAYS) * HIGH_LOW_NOISE_AMPLITUDE

time = np.array([START_DATE] * TIME_LENGTH_DAYS) + np.array([timedelta(days=i) for i in range(TIME_LENGTH_DAYS)])

df["time"] = time
df["open"] = np.round(open_price, 2)
df["high"] = np.round(high_price, 2)
df["low"] = np.round(low_price, 2)
df["close"] = np.round(close_price, 2)
df["tick_volume"] = [0] * TIME_LENGTH_DAYS
df["spread"] = [0] * TIME_LENGTH_DAYS
df["real_volume"] = [0] * TIME_LENGTH_DAYS
df = pd.DataFrame(df)
df.to_csv("../data/sin_wave_sample.csv", index=False)

fig = go.Figure(data=[go.Candlestick(x=time,
                                     open=open_price,
                                     high=high_price,
                                     low=low_price,
                                     close=close_price)])

fig.show()
