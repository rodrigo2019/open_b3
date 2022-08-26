import logging
import os
import sys
from logging import handlers

import numpy as np
import pygad.kerasga
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from utils import list_files

sys.path.append("../../")
from stock_market_simulator.envs import SingleStockTradingEnv  # noqa: E402

TRAINING_FOLDER = "./trains/"
TRAIN_PATH = "./train_files"
VAL_PATH = "./val_files"

WINDOW_SIZE = 1

tf.compat.v1.disable_eager_execution()
tf.config.set_visible_devices([], 'GPU')

os.makedirs(TRAINING_FOLDER, exist_ok=True)
folders = len(next(os.walk(TRAINING_FOLDER))[1])
current_training_path = os.path.join(TRAINING_FOLDER, f"ga_{folders}")
os.makedirs(current_training_path, exist_ok=True)

log_level = logging.DEBUG
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(log_level)
stream_handler.setFormatter(formatter)

logger_name = "train_env"
file_handler = logging.handlers.RotatingFileHandler(f"{current_training_path}/{logger_name}.log",
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
file_handler = logging.handlers.RotatingFileHandler(f"{current_training_path}/{logger_name}.log",
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

model.summary()
keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=10)
best_fit = -np.inf


def fitness_func(solution, sol_idx):
    global keras_ga, model, best_fit

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                 weights_vector=solution)

    model.set_weights(weights=model_weights_matrix)

    try:
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
    except KeyboardInterrupt as e:
        raise e
    except:
        solution_fitness = -np.inf

    if solution_fitness > best_fit:
        best_fit = solution_fitness
        model.save(f"{current_training_path}/model_{solution_fitness}.h5")
    return solution_fitness


def callback_generation(instance):
    print(f"Generation = {instance.generations_completed}")
    print(f"Fitness    = {instance.best_solution()[1]}")


num_generations = 2500
num_parents_mating = 5
initial_population = keras_ga.population_weights

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation)

ga_instance.run()
