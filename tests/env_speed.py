import sys

from tqdm import tqdm

sys.path.append("../../")
from stock_market_simulator.envs import SingleStockTradingEnv  # noqa: E402

FILE_PATH = "../data/WIN$N_m1.csv"
WINDOW_SIZE = 120

env = SingleStockTradingEnv(data_frame=FILE_PATH, window_size=WINDOW_SIZE, logger_name="train_env",
                            ignore_spread=True)

action = env.action_space.sample()
env.reset()
done = False
pbar = tqdm()
while not done:
    obs, reward, done, info = env.step(action)
    pbar.update()
