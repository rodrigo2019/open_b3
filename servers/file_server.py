import os
import time
from abc import ABC, abstractmethod
from threading import Thread
import sys

import pandas as pd
from imutils import paths
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

sys.path.append("..")
from stock_market_simulator.envs import SingleStockTradingEnv  # noqa: E402

FILE_PATH = r"C:\Users\rodri\AppData\Roaming\MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075\Agent-127.0.0.1-3000\MQL5\Files"
MODEL = "PPO"
WINDOW_SIZE = 120
MODEL_PATH = "../training/drl/trains/PPO_24/stochastic/best_model.zip"

models = {
    "A2C": A2C,
    "DDPG": DDPG,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}


class WatchDog(ABC):
    def __init__(self, path):
        self._path = path

        self._running = False
        self._break_process = False

        self._process_thread = None

    @staticmethod
    def _wait_file_creation(path):
        file = None
        while True:
            time.sleep(0.05)
            try:
                file = open(path)
            except (OSError, PermissionError):
                continue
            break
        file.close()

    def _process_folder(self):
        self._running = True
        while not self._break_process:
            try:
                time.sleep(0.05)
                for fname in paths.list_files(self._path):
                    self._wait_file_creation(fname)
                    self.on_event(fname)
                if self._break_process:
                    break
            except Exception as e:
                raise

        self._running = False

    @abstractmethod
    def on_event(self, path):
        raise NotImplementedError("You must implement 'on_event' method")

    def run(self):
        self._break_process = False
        self._process_thread = Thread(target=self._process_folder)
        self._process_thread.daemon = True
        self._process_thread.start()

    def stop(self):
        self._break_process = True
        self._process_thread.join()


class CsvWatchDog(WatchDog):
    def __init__(self, path):
        super().__init__(path)
        self._agent = models[MODEL].load(MODEL_PATH)

    def on_event(self, path):
        print(f"File {path} was created")

        df = pd.read_csv(path, encoding="utf-16")
        df["time"] = pd.to_datetime(df["time"])
        env = SingleStockTradingEnv(data_frame=df, window_size=WINDOW_SIZE, ignore_spread=True)
        observation = env.goto_iteration(len(env) - 1)
        action, _ = self._agent.predict(observation, deterministic=True)
        buy_or_sell, take_profit, stop_loss = env.compute_action(action)
        print(len(df))
        action_path = os.path.join(FILE_PATH, "action.csv_")
        with open(action_path, "w", encoding="utf-16") as f:
            f.write("long_or_short,take_profit,stop_loss, price\n")
            f.write(f"{round(buy_or_sell)}, {take_profit}, {stop_loss}, {env.bid}\n")
        os.rename(action_path, action_path[:-1])
        os.remove(path)

        print(df.tail())
        # raise Exception("Stop")


if __name__ == "__main__":
    watchdg = CsvWatchDog(FILE_PATH)
    watchdg.run()
    while True:
        pass
