import logging
import os
import sys
from logging import handlers

import numpy as np
import optuna
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from utils import list_files

sys.path.append("../../")
from stock_market_simulator.envs import SingleStockTradingEnv  # noqa: E402

SERVER_URL = "mysql://root:q1w2e3r4@localhost/optuna"
STUDY_NAME = "optuna5"
TRAINING_FOLDER = f"./trains/{STUDY_NAME}"
TRAIN_PATH = "./train_files/"
VAL_PATH = "./val_files/"

WINDOW_SIZE = 1

tf.compat.v1.disable_eager_execution()
tf.config.set_visible_devices([], 'GPU')

os.makedirs(TRAINING_FOLDER, exist_ok=True)

study = optuna.create_study(study_name=STUDY_NAME, storage=SERVER_URL, load_if_exists=True, direction="maximize")

log_level = logging.DEBUG
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(log_level)
stream_handler.setFormatter(formatter)

logger_name = "train_env"
file_handler = logging.handlers.RotatingFileHandler(f"{TRAINING_FOLDER}/{logger_name}.log",
                                                    maxBytes=1024 * 1024, backupCount=5)  # 1mb
file_handler.setLevel(log_level)
file_handler.setFormatter(formatter)

train_files = list(list_files(TRAIN_PATH, valid_exts=".csv")) if os.path.isdir(TRAIN_PATH) else TRAIN_PATH
train_env = SingleStockTradingEnv(data_frame=train_files, window_size=WINDOW_SIZE, logger_name="train_env",
                                  one_trade_at_time=True,
                                  ignore_spread=True, random_start=False, max_iter=10_000)

# train_env.logger.addHandler(stream_handler)
train_env.logger.addHandler(file_handler)
train_env.logger.setLevel(log_level)

val_files = list(list_files(VAL_PATH, valid_exts=".csv")) if os.path.isdir(VAL_PATH) else VAL_PATH
eval_env = SingleStockTradingEnv(data_frame=val_files, window_size=WINDOW_SIZE, logger_name="val_env",
                                 ignore_spread=True, one_trade_at_time=True)

logger_name = "val_env"
file_handler = logging.handlers.RotatingFileHandler(f"{TRAINING_FOLDER}/{logger_name}.log",
                                                    maxBytes=1024 * 1024, backupCount=5)  # 1mb
file_handler.setLevel(log_level)
file_handler.setFormatter(formatter)
# eval_env.logger.addHandler(stream_handler)
eval_env.logger.addHandler(file_handler)
eval_env.logger.setLevel(log_level)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=train_env.observation_space.shape))
model.add(keras.layers.Dense(units=4, activation="tanh"))
model.add(keras.layers.Dense(units=4, activation="sigmoid"))

trainable_count = np.sum([keras.backend.count_params(w) for w in model.trainable_weights])


def model_weights_as_matrix(_model, weights_vector):
    weights_matrix = []

    start = 0
    for layer_idx, layer in enumerate(_model.layers):  # model.get_weights():
        # for w_matrix in model.get_weights():
        layer_weights = layer.get_weights()
        if layer.trainable:
            for l_weights in layer_weights:
                layer_weights_shape = l_weights.shape
                layer_weights_size = l_weights.size

                layer_weights_vector = weights_vector[start:start + layer_weights_size]
                layer_weights_matrix = np.reshape(layer_weights_vector, newshape=layer_weights_shape)
                weights_matrix.append(layer_weights_matrix)

                start = start + layer_weights_size
        else:
            for l_weights in layer_weights:
                weights_matrix.append(l_weights)

    return weights_matrix


def optimize_agent(trial):
    weights = [trial.suggest_float(f"w{i}", -4, 4) for i in range(trainable_count)]
    weights_matrix = model_weights_as_matrix(model, weights)
    model.set_weights(weights=weights_matrix)
    solution_fitness = 0
    for i in tqdm(range(len(train_env.data_frames_list))):
        observation = train_env.reset(idx=i)
        done = False
        while not done:
            observation = observation[np.newaxis]
            action = model.predict(observation)[0]
            observation, reward, done, info = train_env.step(action)
            if done:
                break
        solution_fitness += train_env.total_reward
    try:
        if solution_fitness > study.best_value:
            model.save(f"{TRAINING_FOLDER}/model_{solution_fitness}.h5")
    except ValueError:
        model.save(f"{TRAINING_FOLDER}/model_{solution_fitness}.h5")

    return solution_fitness


if __name__ == '__main__':
    study.optimize(optimize_agent, catch=(ValueError,))
