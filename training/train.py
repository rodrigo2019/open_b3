import os

import numpy as np
import pandas as pd
import tensorflow as tf
from imutils import paths
from sklearn.preprocessing import StandardScaler as Scaler
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
from tqdm import tqdm


def load_data(path, label_encoder):
    data = {"train_o": [], "train_i": [], "val_o": [], "val_i": [], "test_o": [], "test_i": []}
    for data_type in ["train", "val", "test"]:
        csv_files_path = list(paths.list_files(os.path.join(path, data_type), ".csv"))
        for csv_path in tqdm(csv_files_path):
            df = pd.read_csv(csv_path)
            df = df["close"]

            input_data = df.values
            input_data = Scaler().fit_transform(input_data.reshape(-1, 1)).flatten()

            output_data = os.path.normpath(csv_path)
            output_data = output_data.split(os.path.sep)[-2]

            data[f"{data_type}_o"].append(label_encoder[output_data])
            data[f"{data_type}_i"].append(input_data)
    for key in data:
        data[key] = np.asarray(data[key])
    return data


def create_model(input_shape, output_shape):
    model_input = Input(shape=input_shape, name='model_input')

    x = Dense(32, activation="relu")(model_input)
    x = BatchNormalization()(x)
    x = Dense(16, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(8, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(output_shape, activation="softmax")(x)

    return Model(inputs=model_input, outputs=x)


def set_memory_growth():
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception as e:
        print(e)
        pass


if __name__ == "__main__":
    le = {"buy": 0, "sell": 1, "hold": 2}
    data_dict = load_data("../tools/dataset/splited", le)

    set_memory_growth()
    model = create_model(data_dict["train_i"].shape[1], 3)
    model.summary()
    model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["acc"])
    model.fit(data_dict["train_i"], data_dict["train_o"], validation_data=(data_dict["val_i"], data_dict["val_o"]),
              epochs=10000, batch_size=8, shuffle=True)
