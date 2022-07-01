import os
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd
from imutils import paths
from sklearn.preprocessing import StandardScaler as Scaler
from tqdm import tqdm


def load_process_csv(csv_path):
    df = pd.read_csv(csv_path)

    input_data = df.values[..., 1:-10]
    input_data = Scaler().fit_transform(input_data)

    output_data = os.path.normpath(csv_path)
    output_data = output_data.split(os.path.sep)[-2]
    return input_data.astype('float32'), output_data


def load_data(path, label_encoder):
    data = {"train_o": [], "train_i": [], "val_o": [], "val_i": [], "test_o": [], "test_i": []}
    limit = os.cpu_count()
    pool = ProcessPoolExecutor(max_workers=limit)
    label_weights = {}
    for data_type in ["train", "val", "test"]:
        csv_files_path = list(paths.list_files(os.path.join(path, data_type), ".csv"))
        all_futures = []
        for csv_path in tqdm(csv_files_path):
            all_futures.append(pool.submit(load_process_csv, csv_path))
            if len(all_futures) < limit:
                continue

            futures_done, _ = wait(all_futures, return_when=FIRST_COMPLETED)
            for future in futures_done:
                input_data, output_data = future.result()
                data[f"{data_type}_o"].append(label_encoder[output_data])
                data[f"{data_type}_i"].append(input_data)
                if data_type == "train":
                    if label_encoder[output_data] not in label_weights:
                        label_weights[label_encoder[output_data]] = 1
                    else:
                        label_weights[label_encoder[output_data]] += 1
                all_futures.remove(future)

        if data_type == "train":
            for k in label_weights:
                label_weights[k] = len(all_futures) / label_weights[k]
    for key in data:
        data[key] = np.asarray(data[key])

    return data, label_weights


if __name__ == "__main__":
    le = {"buy": 0, "sell": 1, "hold": 2}
    data_dict, weights = load_data("dataset", le)
