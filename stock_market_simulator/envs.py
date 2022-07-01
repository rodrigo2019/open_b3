import copy

import gym
import numpy as np

from .market import Broker


class SingleStockTradingEnv(gym.Env, Broker):
    """
    A stock trading environment
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._step_count = 0
        self._reward = 0

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(2,))  # buy or sell, take profit
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space, self._window_size))

        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []

    def _check_profit(self):
        broker = copy.deepcopy(self)
        balance = broker.balance
        for _ in broker:
            pass
        broker.close_all_positions()
        return balance - broker.balance

    def _get_next_observation(self):
        next_state = next(self)
        next_state = next_state[1:]  # exclude the timestamp
        for i in range(next_state.shape[0]):
            next_state[i, :] = next_state[i, :] - next_state[i, :].min()
            max_value = next_state[i, :].max()
            if max_value > 0:
                next_state[i, :] /= max_value

        return next_state

    def step(self, action):
        buy_or_sell, take_profit = action
        take_profit /= 15
        take_profit += 0.005
        stop_loss = take_profit * 1
        done = False

        if self.iter == len(self) - 2:
            done = True
        if not self.has_position:
            if buy_or_sell > 0.5:
                # TODO: the prices must be rounded for the minimum price step
                take_profit = self.ask + self.ask * take_profit
                stop_loss = self.ask - self.ask * stop_loss
                self.buy(0, stop_loss, take_profit)
            else:
                take_profit = self.bid - self.bid * take_profit
                stop_loss = self.bid + self.bid * stop_loss
                self.sell(0, stop_loss, take_profit)
            # reward = self._check_profit()
        reward = self.balance
        next_state = self._get_next_observation()
        reward = self.balance - reward

        self.rewards_memory.append(reward)
        self.state_memory.append(next_state)
        self.actions_memory.append(action)
        if done:
            with open("results.txt", "a") as f:
                acc = 1
                avg_profit = 0
                avg_loss = 0
                for position in self.history_positions:
                    if position.profit > 0:
                        acc += 1
                        avg_profit += position.profit
                    else:
                        avg_loss += position.profit
                avg_profit /= acc
                avg_loss /= (len(self.history_positions) - acc)
                acc /= (len(self.history_positions) + 1)

                f.write(f"balance: {self.balance}\n")
                f.write(f"trades: {len(self.history_positions)}\n")
                f.write(f"accuracy: {acc:.2f}\n")
                f.write(f"avg profit: {avg_profit:.2f}\n")
                f.write(f"avg loss: {avg_loss:.2f}\n")
                f.write(f"-----------------------------\n")

        with open("debug.txt", "a") as f:
            f.write(f"{action} {reward} {self.balance}\n")
        return next_state, reward, done, {}

    def reset(self):
        Broker.reset(self)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []

        return self._get_next_observation()

    def render(self, mode="human"):
        pass

    @property
    def state_space(self):
        return len(self.columns) - 1

    def add_indicator(self, indicator_name):
        super().add_indicator(indicator_name)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space, self._window_size))

    def drop_feature(self, feature):
        super().drop_feature(feature)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space, self._window_size))


if __name__ == "__main__":
    from tqdm import tqdm

    env = SingleStockTradingEnv(csv_file_path="./data/ABEV3_d1.csv", window_size=60)
    _done = False
    pbar = tqdm()
    while True:
        while not _done:
            _state, _reward, _done, _ = env.step([np.random.rand(), np.random.rand(), 0.005, 0.005])
            pbar.update()
        print(env.balance, len(env.history_positions), len(env))
        env.reset()

        _done = False
