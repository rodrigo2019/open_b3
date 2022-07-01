import os
import random

import pandas as pd
import plotly.graph_objects as go
from imutils import paths

csv_files_path = list(paths.list_files("./dataset", ".csv"))
random.shuffle(csv_files_path)
idx_to_label = ["buy", "sell", "hold"]
data = []
for fname in csv_files_path[:30]:
    df = pd.read_csv(fname)
    symbol = df["symbol"].values[0]
    label = df["label"].values[0]
    basename = os.path.basename(fname)
    basename = os.path.splitext(basename)[0]
    name = f"{symbol}_{basename}_{idx_to_label[label]}"
    data.append(go.Candlestick(x=df['time'],
                               open=df['open'],
                               high=df['high'],
                               low=df['low'],
                               close=df['close'], name=name))

fig = go.Figure(data=data)
fig.update_layout(title="datasets", yaxis_title="BRL")
fig.show()
