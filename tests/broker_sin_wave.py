import numpy as np
import pandas as pd
import quantstats as qs
from tqdm import tqdm

from stock_market_simulator.market import Broker

SELL_ABOVE = 14
BUY_BELOW = 6
CSV_PATH = "../data/sin_wave_sample.csv"

qs.extend_pandas()
broker = Broker(csv_file_path=CSV_PATH, window_size=5)
broker.points2currency = 1
print(broker.columns)

close_index = broker.columns["close"]
high_low = False
for data in tqdm(broker):
    close_price = data[close_index, -1]
    if close_price > SELL_ABOVE and not high_low:
        broker.close_all_positions()
        broker.sell(volume=1)
        high_low = True
    elif close_price < BUY_BELOW and high_low:
        broker.close_all_positions()
        broker.buy(volume=1)
        high_low = False
broker.close_all_positions()

print("finish", broker.balance, len(broker.history_positions))

returns = []
time = []
for position in broker.history_positions:
    returns.append(position.profit)
    time.append(position.close_time)

returns = np.asarray(returns).cumsum()
returns = pd.Series(returns, index=time)
qs.reports.html(returns, download_filename="broker_sin_wave_report.html", output="")
