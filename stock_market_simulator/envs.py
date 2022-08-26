import logging
import random
from typing import Union, List, Tuple

import gym
import numpy as np
import pandas as pd

from .market import Broker, OrderTypes
from .utils.metrics import sharpe_ratio


class SingleStockTradingEnv(gym.Env, Broker):
    """
    A stock trading environment
    """

    def __init__(self, logger_name: str = "SingleStockTradingEnv", random_start: bool = False, max_iter: int = None,
                 one_trade_at_time: bool = False, **kwargs) -> None:
        """

        :param logger_name: The name of logger, useful there are situation with many environments, it helps for tracking
            logs from each object.
        :param random_start: A random step to start the simulations, it helps to bring stochasticity to the env, as
            every episode has the same samples.
        :param max_iter: Max number of iterations to consider it done.
        :param one_trade_at_time: if true it will not start a new position ultil the last one is closed. its similar
            Hedge vs Netting.
        :param kwargs: Check the information in the base classes.
        """
        data_frame = kwargs.get("data_frame", None)

        self._data_frames_list = None
        if data_frame is not None:
            if not isinstance(data_frame, list):
                data_frame = [data_frame]
            self._data_frames_list = data_frame
            self._current_choice = random.choice(self._data_frames_list)
            kwargs["data_frame"] = self._current_choice
        super().__init__(**kwargs)
        self._use_raw_candle_as_feature = True
        self._set_indicators()

        self._hit_count = 0
        self._total_reward = 0
        self._next_raw_state = None

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(4,))  # buy or sell, take profit, stop loss, sit
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.state_space, self._window_size))

        self.logger = logging.getLogger(logger_name)

        self._one_trade_at_time = one_trade_at_time
        self._random_start = random_start
        self._max_iter = max_iter
        self._relative_max_iter = None

    def _set_indicators(self) -> None:
        """
        Add and drop some features that will be used to generate the environment observation.

        :return:
        """
        self._use_raw_candle_as_feature = False
        for drop_feature in ["tick_volume"]:
            self.drop_feature(drop_feature)
        for indicator_name in ["rsi", "macd", "boll_ub", "boll_lb"]:
            self.add_indicator(indicator_name)

    def _get_next_observation(self) -> np.ndarray:
        """
        Generate the next observation.

        Currently the next observation is based on the features set in the "self._set_indicators" method.
        All features will be scaled between 0 and 1, the normalization its make based on correlated features. E.G.: the
         correlated features for the raw candle is OHLC.

        The output will be an array with shape [N_features, Window size].
        :return:
        """
        next_state = next(self).copy()
        self._next_raw_state = next_state.copy()

        selected_columns = ()
        #  make the normalization of the state keep the correlation between the features
        # raw candle
        if self._use_raw_candle_as_feature:
            columns = (self.columns["open"], self.columns["high"], self.columns["low"], self.columns["close"])
            selected_columns += columns
            next_state[columns, :] = next_state[columns, :] - next_state[columns, :].min()
            max_value = next_state[columns, :].max()
            if max_value != 0:
                next_state[columns, :] = next_state[columns, :] / next_state[columns, :].max()

        # volume
        try:
            columns = (self.columns["volume"],)
            selected_columns += columns
            next_state[columns, :] = next_state[columns, :] - next_state[columns, :].min()
            max_value = next_state[columns, :].max()
            if max_value != 0:
                next_state[columns, :] = next_state[columns, :] / next_state[columns, :].max()
        except KeyError:
            pass

        # Bollinger bands
        try:
            columns = (self.columns["boll"], self.columns["boll_ub"], self.columns["boll_lb"])
            selected_columns += columns
            next_state[columns, :] = next_state[columns, :] - next_state[columns, :].min()
            max_value = next_state[columns, :].max()
            if max_value != 0:
                next_state[columns, :] = next_state[columns, :] / next_state[columns, :].max()
        except KeyError:
            pass

        # MACD
        try:
            columns = (self.columns["macdh"], self.columns["macd"], self.columns["macds"])
            selected_columns += columns
            next_state[columns, :] = next_state[columns, :] - next_state[columns, :].min()
            max_value = next_state[columns, :].max()
            if max_value != 0:
                next_state[columns, :] = next_state[columns, :] / next_state[columns, :].max()
        except KeyError:
            pass

        # RSI
        try:
            columns = (self.columns["rs_14"], self.columns["rsi"])
            selected_columns += columns
            next_state[columns, :] = next_state[columns, :] - next_state[columns, :].min()
            max_value = next_state[columns, :].max()
            if max_value != 0:
                next_state[columns, :] = next_state[columns, :] / next_state[columns, :].max()
        except KeyError:
            pass

        next_state = next_state[selected_columns, :]

        next_state = next_state.astype("float32")
        return next_state

    def _log_episode_end(self) -> None:
        """
        Generate a message displaying metrics from the current episode.

        :return:
        """
        acc = 1
        gross_profit = 0
        gross_loss = 0

        count_short = 0
        count_long = 0

        for position in self.history_positions:
            if position.open_order.order_type == OrderTypes.BUY:
                count_long += 1
            else:
                count_short += 1
            if position.profit > 0:
                acc += 1
                gross_profit += position.profit
            else:
                gross_loss += position.profit
        avg_profit = gross_profit / acc
        avg_loss = gross_loss / (len(self.history_positions) - acc + 1e-10)
        avg_loss = abs(avg_loss)
        acc /= (len(self.history_positions) + 1)
        stability = acc - (avg_loss / (avg_loss + avg_profit + 1e-10))  # calculate how much we are above 0x0 rate
        try:
            sharpe = sharpe_ratio([position.profit for position in self.history_positions])
        except ZeroDivisionError:
            sharpe = 0

        message = "\n#########################################\n"
        message += f"current choice: {self._current_choice}\n"
        message += f"balance: {self.balance:.2f}\n"
        message += f"total reward: {self.total_reward:.2f}\n"
        message += f"gross profit: {gross_profit:.2f}\n"
        message += f"gross loss: {gross_loss:.2f}\n"
        message += f"avg profit: {avg_profit:.2f}\n"
        message += f"avg loss: {avg_loss:.2f}\n"
        message += "#########################################\n"
        message += f"stability: {stability:.2%}\n"
        message += f"sharpe: {sharpe:.2f}\n"
        message += f"recovery factor: {gross_profit / (abs(gross_loss) + 1e-10):.02f}\n"
        message += "#########################################\n"
        message += f"trades: {len(self.history_positions)}\n"
        message += f"short trades: {count_short} ({count_short / (len(self.history_positions) + 1e-10):.2%})\n"
        message += f"long trades: {count_long} ({count_long / (len(self.history_positions) + 1e-10):.2%})\n"
        message += f"accuracy: {acc:.2%}\n"

        self.logger.info(message)

    def _get_reward(self, buy_or_sell: float, **kwargs) -> float:
        """
        Calculate the reward for the current trade. This method will iterate over a copy or itself and it will stop when
            the position is closed.
        It will iterate over a copy when onde_trade_at_time=False, otherwise it will iterate over itself.

        :param buy_or_sell: 0=sell, 1=buy.
        :param kwargs:
        :return:
        """
        if self._one_trade_at_time:
            broker = self
        else:
            broker = self.copy()

        if buy_or_sell > 0.5:
            broker.buy(**kwargs)
        else:
            broker.sell(**kwargs)
        if not broker.has_position:
            raise RuntimeError("'_get_reward' method was called without a open position.")

        while broker.has_position:
            if broker.iter < len(broker) - 2:
                next(broker)
            else:
                broker.close_all_positions()
                break

        if not self._one_trade_at_time:
            self._history_position_list.append(broker.history_positions[-1])

        if broker.history_positions[-1].open_order.order_type == OrderTypes.BUY:
            reward = broker.history_positions[-1].closed_price / broker.history_positions[-1].opened_price
        else:
            reward = broker.history_positions[-1].opened_price / broker.history_positions[-1].closed_price
        reward -= 1  # -1 to inf%
        if reward > 0:
            self._hit_count += 1

        reward *= 100  # -100% to inf%
        reward *= (self._hit_count / (self.iter - self._window_size + 2))
        return reward

    def compute_action(self, action: Union[List[float], np.ndarray]) -> Tuple[bool, float, float, bool]:
        """
        The raw action from the model will be an array with values between 0 and 1. We must process it to convert
            into the actual action.
            The currently action are:
                Buy or sell: Indicate the trade direction
                take profit: the take profit price
                stop loss: the stop loss price
                sit: if true no trade will be made

        :param action: The raw array from the model.
        :return:
        """
        buy_or_sell, take_profit, stop_loss, sit = action
        sit = bool(round(sit))

        if self._next_raw_state is None:
            raise ValueError("The next state is not computed yet")
        mean_high = self._next_raw_state[self.columns["high"]].mean()
        mean_low = self._next_raw_state[self.columns["low"]].mean()
        mean_candle_height = mean_high - mean_low

        take_profit *= 4  # Scale our max take profit
        take_profit += 1  # Add a small amount to avoid taking the exact price as take profit
        take_profit *= mean_candle_height  # Scale our take profit to the mean candle height

        stop_loss *= 4  # Scale our max stop loss
        stop_loss += 1  # Add a small amount to avoid taking the exact price as stop loss
        stop_loss *= mean_candle_height  # Scale our stop loss to the mean candle height

        if buy_or_sell > 0.5:
            # TODO: the prices must be rounded for the minimum price step
            take_profit = self.ask + take_profit
            stop_loss = self.ask - stop_loss
        else:
            take_profit = self.bid - take_profit
            stop_loss = self.bid + stop_loss

        take_profit = max(0, take_profit)
        stop_loss = max(0, stop_loss)

        take_profit = round(take_profit, 2)
        stop_loss = round(stop_loss, 2)
        return buy_or_sell, take_profit, stop_loss, sit

    def step(self, action: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute the action and return the observation, reward in the gym env style.

        :param action: The raw array action from the model.
        :return:
        """
        buy_or_sell, take_profit, stop_loss, sit = self.compute_action(action)

        if not sit:
            reward = self._get_reward(buy_or_sell, price=0, sl=stop_loss, tp=take_profit)
        else:
            reward = 0

        done = True if self.iter >= len(self) - 2 else False
        if self._relative_max_iter is not None and self.iter >= self._relative_max_iter:
            done = True

        next_state = self._get_next_observation()

        if done:
            self._log_episode_end()

        self._total_reward += reward
        return next_state, reward, done, {}

    def reset(self, idx: Union[None, int] = None):
        """
        Reset the environment state.

        :param idx: If you are using multiple files as input, you can specify an index to choose a specific file for the
            next episode.
        :return:
        """
        Broker.reset(self)
        if len(self._data_frames_list) > 1:
            if idx is not None:
                assert idx < len(self._data_frames_list), "idx must be lower than available data frames list"
                self._current_choice = self._data_frames_list[idx]
            else:
                self._current_choice = random.choice(self._data_frames_list)
            self._data_frame = self._set_up_dataframe(self._current_choice)
            self._set_indicators()

        self._hit_count = 0
        self._total_reward = 0

        if self._random_start:
            idx = np.random.randint(self.iter, len(self) - 2)
            self._relative_max_iter = idx + self._max_iter if self._max_iter is not None else None
            self.goto_iteration(idx)
        return self._get_next_observation()

    def render(self, mode: str = "human") -> None:
        pass

    def goto_iteration(self, iteration: int) -> np.ndarray:
        """
        Move the environment to a specific step in the future. It is not possible to go back in the past.

        :param iteration: Iteration to go to.
        :return:
        """
        assert iteration < len(self), f"The iteration must be less than the number of iterations. iter: {len(self) - 1}"
        assert iteration >= self.iter, "It's not possible move to the past."
        while self.iter < iteration and self.has_position:
            next(self)
        self._iter = iteration - 1
        return self._get_next_observation()

    @property
    def state_space(self) -> int:
        """
        How many features are currently used by the environment.

        :return:
        """
        if self._use_raw_candle_as_feature:
            return len(self.columns) - 1
        else:
            return len(self.columns) - 1 - 4

    @property
    def total_reward(self) -> float:
        """
        Sum of all rewards earned in the episode.

        :return: Total reward value.
        """
        return self._total_reward

    @property
    def data_frames_list(self) -> List[Union[str, pd.DataFrame]]:
        """
        List of all input data files.

        :return: Ths list can be mixed between pd.DataFrames and path of the datafiles.
        """
        return self._data_frames_list

    def add_indicator(self, indicator_name: str) -> None:
        """
        When some indicator are add in the gym environment we must recalculate our state space and remove all nan values
            from our dataframe.

        :param indicator_name: See the Broker documentation for more information.
        :return:
        """
        super().add_indicator(indicator_name)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space, self._window_size))
        # because of ml stuffs we cannot work with nan
        self._data_frame_values[1:] = np.nan_to_num(self._data_frame_values[1:].astype("float64"))

    def drop_feature(self, feature: str) -> None:
        """
        When some feature are drop in the gym environment we must recalculate our state space.

        :param feature: See the Broker documentation for more information.
        :return:
        """
        super().drop_feature(feature)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space, self._window_size))

    @property
    def bid(self) -> float:
        """
        Override the bid property from the Broker.

        Because of the mt5 we are working with open prices as the last price.
        :return:
        """
        return self._data_frame["open"].values[self._iter + 1]

    @property
    def ask(self) -> float:
        """
        Override the ask property from the Broker.

        Because of the mt5 we are working with open prices as the last price.
        :return:
        """
        if self._ignore_spread:
            return self._data_frame["open"].values[self._iter]
        else:
            return self._data_frame["open"].values[self._iter + 1] + self._data_frame["spread"].values[self._iter + 1]
