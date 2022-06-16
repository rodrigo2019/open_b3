import os

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from callbacks import ModelCheckpointLogged, ConfusionMatrixTB
from loader import load_data


def create_model(input_shape, output_shape):
    model_input = Input(shape=input_shape, name='model_input')

    x = Dense(1024, activation="selu")(model_input)
    x = BatchNormalization()(x)
    x = Dense(512, activation="selu")(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="selu")(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation="selu")(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation="selu")(x)
    x = BatchNormalization()(x)
    x = Dense(8, activation="selu")(x)
    x = BatchNormalization()(x)
    x = Dense(output_shape, activation="softmax")(x)

    return Model(inputs=model_input, outputs=x)


def create_callbacks(data_val):
    os.makedirs("./logs", exist_ok=True)
    folders = 0
    _, dirnames, _ = next(os.walk("./logs"))
    folders += len(dirnames)

    log_folder = os.path.join("./logs", str(folders))
    os.mkdir(log_folder)

    tb_cb = TensorBoard(log_folder)
    cm_acc_cb = ConfusionMatrixTB(data_val, ["buy", "sell", "hold"], log_folder,
                                  monitor="val_acc", mode="max")
    cm_loss_cb = ConfusionMatrixTB(data_val, ["buy", "sell", "hold"], log_folder,
                                   monitor="val_loss", mode="min")
    best_loss = ModelCheckpointLogged(os.path.join(log_folder, "best_loss.h5"), save_best_only=True, monitor="val_loss",
                                      mode="auto", period=1, verbose=1)
    best_acc = ModelCheckpointLogged(os.path.join(log_folder, "best_acc.h5"), save_best_only=True, monitor="val_acc",
                                     mode="auto", period=1, verbose=1)

    ckp = ModelCheckpointLogged(os.path.join(log_folder, "ckp.h5"), save_best_only=False, period=10, verbose=1)
    return [tb_cb, best_loss, best_acc, ckp, cm_acc_cb, cm_loss_cb]


def set_memory_growth():
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception as e:
        print(e)
        pass


if __name__ == "__main__":
    le = {"buy": 0, "sell": 1, "hold": 2}
    data_dict, class_weights = load_data("../tools/dataset", le)
    print(class_weights)
    set_memory_growth()
    model = create_model(data_dict["train_i"].shape[1], 3)
    model.summary()
    opt = Adam()
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["acc"])
    model.fit(data_dict["train_i"], data_dict["train_o"], validation_data=(data_dict["val_i"], data_dict["val_o"]),
              epochs=10000, batch_size=128, shuffle=True,
              callbacks=create_callbacks((data_dict["val_i"], data_dict["val_o"])),
              class_weight=class_weights)
