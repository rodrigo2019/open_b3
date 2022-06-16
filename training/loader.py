import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from imutils import paths
from sklearn.preprocessing import StandardScaler as Scaler
from tqdm import tqdm


def load_process_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df["close"]

    input_data = df.values
    input_data = Scaler().fit_transform(input_data.reshape(-1, 1)).flatten()

    output_data = os.path.normpath(csv_path)
    output_data = output_data.split(os.path.sep)[-2]
    return input_data, output_data


def load_data(path, label_encoder):
    data = {"train_o": [], "train_i": [], "val_o": [], "val_i": [], "test_o": [], "test_i": []}
    pool = ProcessPoolExecutor(max_workers=os.cpu_count())
    label_weights = {}
    for data_type in ["train", "val", "test"]:
        csv_files_path = list(paths.list_files(os.path.join(path, data_type), ".csv"))
        all_futures = []
        for csv_path in tqdm(csv_files_path):
            all_futures.append(pool.submit(load_process_csv, csv_path))

        for future in tqdm(as_completed(all_futures), total=len(all_futures)):
            input_data, output_data = future.result()
            data[f"{data_type}_o"].append(label_encoder[output_data])
            data[f"{data_type}_i"].append(input_data)
            if data_type == "train":
                if label_encoder[output_data] not in label_weights:
                    label_weights[label_encoder[output_data]] = 1
                else:
                    label_weights[label_encoder[output_data]] += 1
        if data_type == "train":
            for k in label_weights:
                label_weights[k] = len(all_futures) / label_weights[k]
    for key in data:
        data[key] = np.asarray(data[key])

    return data, label_weights


if __name__ == "__main__":
    le = {"buy": 0, "sell": 1, "hold": 2}
    data_dict = load_data("../tools/dataset", le)
