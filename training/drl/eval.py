import logging
import os
import sys

from matplotlib import pyplot as plt
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
import numpy as np

sys.path.append("../../")
from stock_market_simulator.envs import SingleStockTradingEnv  # noqa: E402
from stock_market_simulator.market import OrderTypes  # noqa: E402

TRAINING_FOLDER = "./trains/"
MODEL_FOLDER = "PPO_22"
WINDOW_SIZE = 120
model_type = MODEL_FOLDER.split("_")[0]
models = {
    "A2C": A2C,
    "DDPG": DDPG,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}

log_level = logging.DEBUG
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(log_level)
stream_handler.setFormatter(formatter)

eval_env = SingleStockTradingEnv(data_frame="./val.csv", window_size=WINDOW_SIZE, logger_name="train_env",
                                 ignore_spread=True)
eval_env.logger.addHandler(stream_handler)
eval_env.logger.setLevel(log_level)

agent = models[model_type].load(os.path.join(TRAINING_FOLDER, MODEL_FOLDER, "stochastic", "best_model.zip"),
                                print_system_info=True)

obs = eval_env.reset()
# obs = eval_env.goto_iteration(60 + 5)
done = False
while not done:
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)

x = []
y_price = []
y_percent = []
total_percent = 1
for position in eval_env.history_positions:
    x.append(position.close_time)
    y_price.append(position.profit)
    if position.open_order.order_type == OrderTypes.BUY:
        profit = position.closed_price / position.opened_price
        total_percent *= profit
        y_percent.append((total_percent - 1) * 100)
    else:
        profit = position.opened_price / position.closed_price
        total_percent *= profit
        y_percent.append((total_percent - 1) * 100)
    # print(position.close_time, position.open_time, position.opened_price, position.open_order.sl, position.open_order.tp)
y_price = np.cumsum(y_price)
plt.plot(x, y_price)
plt.figure()
plt.plot(x, y_percent)
plt.show()
