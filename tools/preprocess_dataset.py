import os
import shutil

import pandas as pd
from imutils import paths
from tqdm import tqdm

ALL_CSV_PATH = list(paths.list_files("./dataset/raw", ".csv"))
VAL_YEARS = [2019]
TEST_YEARS = [2020]

for data_type in ["train", "val", "test"]:
    for data_class in ["buy", "sell", "hold"]:
        os.makedirs(f"./dataset/splited/{data_type}/{data_class}", exist_ok=True)

discarded = 0
for csv_path in tqdm(ALL_CSV_PATH):
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    if df["timedelta_days"].values[0] > 15:
        discarded += 1
        continue
    if 0 < float(df["profit_percent"].values[0][:-1]) < 0.5:
        discarded += 1
        continue
    label = os.path.normpath(csv_path)
    label = label.split(os.path.sep)[-2]
    sample_year = df.index[-1].year
    if sample_year in VAL_YEARS:
        dst = os.path.join(f"./dataset/splited/val/{label}", os.path.basename(csv_path))
    elif sample_year in TEST_YEARS:
        dst = os.path.join(f"./dataset/splited/test/{label}", os.path.basename(csv_path))
    else:
        dst = os.path.join(f"./dataset/splited/train/{label}", os.path.basename(csv_path))
    shutil.copy(csv_path, dst)

print(f"{discarded} discarded files.")
