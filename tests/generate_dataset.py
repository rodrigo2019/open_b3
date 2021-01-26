from tqdm import tqdm

import import_workaround  # noqa
from b3_predictor import Broker

CSV_DATA = "../datasets/WIN%N_m5.csv"
WINDOWS_SIZE = 20
TAKE_PROFIT = 200
STOP_LOSS = 100

broker = Broker(csv_file_path=CSV_DATA, window_size=WINDOWS_SIZE)

history = []
get_new_data = True
started_price = -1
for data in tqdm(broker):
    if get_new_data:
        temp_data = data
        started_price = broker.ask
        get_new_data = False
    else:
        if STOP_LOSS < abs(broker.ask - started_price) > TAKE_PROFIT:
            # bla bla bla
