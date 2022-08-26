import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

from utils import list_files

sys.path.append("../../")
from stock_market_simulator.envs import SingleStockTradingEnv  # noqa: E402

TRAINING_FOLDER = "./trains/"
DATA_PATH = "./val_files"
MODEL_PATH = "./trains/optuna9/model_2803.425745997559.h5"

tf.compat.v1.disable_eager_execution()
tf.config.set_visible_devices([], 'GPU')

model = load_model(MODEL_PATH)
window_size = model.input.shape[-1]

files = list(list_files(DATA_PATH, valid_exts=".csv")) if os.path.isdir(DATA_PATH) else DATA_PATH
env = SingleStockTradingEnv(data_frame=files, window_size=window_size, logger_name="train_env",
                            one_trade_at_time=True,
                            ignore_spread=True, random_start=False, max_iter=10_000)

log_level = logging.DEBUG
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(log_level)
stream_handler.setFormatter(formatter)

env.logger.addHandler(stream_handler)
env.logger.setLevel(log_level)

total_reward = 0
total_balance = 0
gross_profit = 0
gross_loss = 0
for i in tqdm(range(len(env.data_frames_list))):
    observation = env.reset(idx=i)
    done = False
    while not done:
        observation = observation[np.newaxis]
        action = model.predict(observation)[0]
        observation, reward, done, info = env.step(action)
        if done:
            break

    if env.balance > 0:
        gross_profit += env.balance
    else:
        gross_loss -= env.balance
    total_reward += env.total_reward
    total_balance += env.balance

print(f"balance: {total_balance}")
print(f"gross profit: {gross_profit}")
print(f"gross loss: {gross_loss}")
print(f"recovery factor: {gross_profit / gross_loss:.2f}")
print(f"reward: {total_reward}")

model.summary()