import logging
import os
import sys
from logging import handlers

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from utils import list_files

sys.path.append("../../")
from stock_market_simulator.envs import SingleStockTradingEnv  # noqa: E402

TRAINING_FOLDER = "./trains/"
TRAIN_PATH = "./train_files"
VAL_PATH = "./val_files"

MODEL = "PPO"
WINDOW_SIZE = 1

models = {
    "A2C": A2C,
    "DDPG": DDPG,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,

}
os.makedirs(TRAINING_FOLDER, exist_ok=True)
folders = len(next(os.walk(TRAINING_FOLDER))[1])
current_training_path = os.path.join(TRAINING_FOLDER, f"{MODEL}_{folders}")
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

train_env.logger.addHandler(stream_handler)
train_env.logger.addHandler(file_handler)
train_env.logger.setLevel(log_level)

val_files = list(list_files(VAL_PATH, valid_exts=".csv")) if os.path.isdir(VAL_PATH) else VAL_PATH
eval_env = SingleStockTradingEnv(data_frame=val_files, window_size=WINDOW_SIZE, logger_name="val_env",
                                 ignore_spread=True)

logger_name = "val_env"
file_handler = logging.handlers.RotatingFileHandler(f"{current_training_path}/{logger_name}.log",
                                                    maxBytes=1024 * 1024, backupCount=5)  # 1mb
file_handler.setLevel(log_level)
file_handler.setFormatter(formatter)
eval_env.logger.addHandler(stream_handler)
eval_env.logger.addHandler(file_handler)
eval_env.logger.setLevel(log_level)

eval_callback_stochastic = EvalCallback(Monitor(eval_env),
                                        best_model_save_path=os.path.join(current_training_path, "stochastic"),
                                        log_path=current_training_path, eval_freq=10_000, n_eval_episodes=10,
                                        deterministic=False, render=False)
eval_callback_deterministic = EvalCallback(Monitor(eval_env),
                                           best_model_save_path=os.path.join(current_training_path, "deterministic"),
                                           log_path=current_training_path, eval_freq=10_0000, n_eval_episodes=1,
                                           deterministic=True, render=False)
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=current_training_path)
callbacks = [eval_callback_stochastic, eval_callback_deterministic, checkpoint_callback]

agent = models[MODEL]("MlpPolicy", train_env, verbose=1, tensorboard_log=current_training_path)
agent.learn(total_timesteps=100_000_000, callback=callbacks)
