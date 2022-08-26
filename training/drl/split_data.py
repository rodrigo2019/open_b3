import os

import pandas as pd
from tqdm import tqdm

from utils import list_files

DATA_PATH = "../../data/"
DST_TRAIN = "./train_files"
DST_VAL = "./val_files"
TRAIN_SIZE = 0.8


def split_data(fname, fname_train, fname_val):
    df = pd.read_csv(fname)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    date = df.index[int(len(df) * TRAIN_SIZE)]

    condition = df.index < date
    df_train = df[condition]
    df_val = df[~condition]

    df_train.to_csv(fname_train)
    df_val.to_csv(fname_val)


if os.path.isdir(DATA_PATH):
    data_files = list_files(DATA_PATH, valid_exts=".csv")
else:
    data_files = [DATA_PATH]

os.makedirs(DST_TRAIN, exist_ok=True)
os.makedirs(DST_VAL, exist_ok=True)

for _fname in tqdm(list(data_files)):
    split_data(_fname,
               os.path.join(DST_TRAIN, os.path.basename(_fname)),
               os.path.join(DST_VAL, os.path.basename(_fname)))
