from abc import ABC, abstractmethod
from enum import IntEnum

import pandas as pd
import numpy as np


class OrderTypes(IntEnum):
    BUY = 0
    SELL = 1


class Order:
    """
    A order is a request to BUY or SELL some stock in the market.
    Some order can be executed instantly when the set price is at market, it means that you are going to pay the
        current price of the market, or you can set a specified price to pay in some stock. Also you can set the
        take profit (TP) stop loss (SL) values.

    **The current state of this lib don't use a order to close a position, just for opening a new one.**
    """
    _ticket_counter = 0

    def __init__(self, price, sl, tp, volume, order_type, create_time):
        """

        :param price:[float] The price that are going to pay for some stock, set as 0 if you want to buy at market
            price.
        :param sl:[float] Stop loss value.
        :param tp:[float] Take profit value.
        :param volume:[float] Amount of some stock.
        :param order_type:[OrderTypes] BUY for long position and SELL for short position.
        :param create_time: [numpy.datetime64] The current time when this Order was created.
        """
        self._price = price
        self._sl = sl
        self._tp = tp
        self._volume = volume
        self._order_type = order_type
        self._create_time = create_time

        self._ticket = Order._ticket_counter
        Order._ticket_counter += 1

    @property
    def price(self):
        return self._price

    @property
    def sl(self):
        return self._sl

    @property
    def tp(self):
        return self._tp

    @property
    def volume(self):
        return self._volume

    @property
    def order_type(self):
        return self._order_type

    @property
    def create_time(self):
        return self._create_time

    @property
    def ticket(self):
        """
        The ticket is an unique ID for order tracking purposes
        :return: [int] The ID of an order
        """
        return self._ticket


class Position:
    """
    A Position it is when you have some stock in your wallet, a position is always created by an executed order, you
        can have two types of position, short and long. Long position happens when you buy a stock and close the
        position selling it, in another hand the short position is the opposite of the long, first you need to sell
        and close it buying.
    """
    _ticket_counter = 0

    def __init__(self, open_order, opened_price, open_time):
        """

        :param open_order: [Order] The order which opened the current position.
        :param opened_price: [float] The current price when the order was executed.
        :param open_time: [numpy.datetime64] The current time when this Position was opened.
        """
        self._open_order = open_order
        self._opened_price = opened_price
        self._last_price = 0

        self._ticket = Position._ticket_counter
        Position._ticket_counter += 1

        self._open_time = open_time
        self._close_time = None
        self._closed_price = None

    @property
    def open_order(self):
        return self._open_order

    @property
    def last_price(self):
        return self._last_price

    @last_price.setter
    def last_price(self, new_price):
        """
        The broker must update the last price of the current opened positions.
        :param new_price: [float] The last price from the market for the stock.
        :return: [None]
        """
        assert new_price > 0, "The new price must be a positive value."
        self._last_price = new_price

    @property
    def opened_price(self):
        return self._opened_price

    @property
    def closed_price(self):
        return self._closed_price

    @property
    def profit(self):
        """
        Calculate the current profit of the position. This property returns the profit in points, not in the currency.
        :return: [float] The current profit in points.
        """
        if self._open_order.order_type == OrderTypes.BUY:
            return self._last_price - self._opened_price
        else:
            return self._opened_price - self._last_price

    @property
    def open_time(self):
        return self._open_time

    @property
    def close_time(self):
        return self._close_time

    @close_time.setter
    def close_time(self, timestamp):
        assert self._close_time is None, "close time can be set only once."
        self._close_time = timestamp
        self._closed_price = self.last_price

    @property
    def ticket(self):
        return self._ticket


class Market(ABC):
    """
    This class simulate the market for a single stock. The stock data is read from a CSV file extracted from the
        metatrader 5. This class doesn't simulate the market ticks, the simulation are based on the maximum and minimums
        of each bar, and if you go to execute some order at market, the based price will be the open price of the
        current bar.
    The Market is responsible for generating values of the stock in each iteration.
    """

    def __init__(self, csv_file_path, window_size=1):
        """
        It is  a iterable class, for each iteration N history bars will be returned to the user. It is really useful
            for analyzing the past to predict the future.
        :param csv_file_path: [string] The csv file path with the stock data
        :param window_size: [int] Number of the past bars that will be returned in each iteration
        """
        assert window_size > 0, "window_size must be > 0."
        self._window_size = window_size

        self._data_frame = pd.read_csv(csv_file_path)
        self._data_frame["time"] = pd.to_datetime(self._data_frame["time"])
        self._data_frame = self._data_frame.set_index("time")

        self._points2currency = 1
        self._iter = 0
        self._time = np.datetime64("nat")

    def __iter__(self):
        for self._iter in range(self._window_size, len(self._data_frame), 1):
            self._time = self._data_frame.index.values[self._iter]
            self._broker_callback()
            yield self._data_frame[self._iter - self._window_size:self._iter]

    def __len__(self):
        return len(self._data_frame)

    @property
    def bid(self):
        """
        Price that you need to look if you are going to sell.
        :return:
        """
        return self._data_frame["close"].values[self._iter]

    @property
    def ask(self):
        """
        Price that you need to look if you are going to buy.
        :return:
        """
        return self._data_frame["close"].values[self._iter] + self._data_frame["spread"].values[self._iter]

    @property
    def time(self):
        """
        Timestamp from the last iteration bar
        :return: [numpy.datetime64]
        """
        return self._time

    @property
    def points2currency(self):
        return self._points2currency

    @points2currency.setter
    def points2currency(self, new_value):
        """
        In some stocks the current value is not based on the currency, so with this factor we can convert the points
        value to the currency value.
        E.g for each 5 points in the WIN stock it equivalent to 1BRL.
        :param new_value: [float] the factor for convert points to currency.
        :return: [None]
        """
        assert new_value >= 0, "points2currency value must be >= 0."

    @abstractmethod
    def _broker_callback(self):
        """
        For each market iteration the Broker can overload this method to execute specific task in each market iteration.
        :return: [None]
        """
        pass


class Broker(Market):
    """
    The broker is responsible for executing and creating orders
    """

    def __init__(self, *args, init_deposit=0, ignore_spread=False, spread_precision=0.01, **kwargs):
        """

        :param args: See more information in Market class.
        :param init_deposit: Initial amount of money.
        :param ignore_spread: [Boolean] If true will not apply the spread difference between ask and bid price.
        :param spread_precision: [float] The precision of the spread in the currency, e.g: 1 spread point = 0.01BRL, so
            the spread precision would be 0.01
        :param kwargs: See more information in Market class.
        """
        super().__init__(*args, **kwargs)
        self._init_deposit = init_deposit
        self._ignore_spread = ignore_spread
        self._spread_precision = spread_precision

        if self._ignore_spread:
            self._data_frame["spread"].values[:] = 0
        else:
            self._data_frame["spread"].values[:] = self._data_frame["spread"].values[:] * self._spread_precision

        self._orders_list = []
        self._opened_positions_list = []
        self._history_position_list = []

    def buy(self, price=0, sl=0, tp=0, volume=1):
        """

        :param price: [float] The price that you wanna pay in some stock. (0 for buying at market)
        :param sl: [float] Stop loss price.
        :param tp: [float] Take profit price.
        :param volume: [float] Volume to buy.
        :return: [None]
        """
        if price == 0:
            self._opened_positions_list.append(Position(Order(price, sl, tp, volume, OrderTypes.BUY, self._time),
                                                        self.ask, self._time))
        else:
            self._orders_list.append(Order(price, sl, tp, volume, OrderTypes.BUY, self._time))

    def sell(self, price=0, sl=0, tp=0, volume=1):
        """

        :param price: [float] The price that you wanna pay in some stock. (0 for selling at market)
        :param sl: [float] Stop loss price.
        :param tp: [float] Take profit price.
        :param volume: [float] Volume to buy.
        :return: [None]
        """
        if price == 0:
            self._opened_positions_list.append(Position(Order(price, sl, tp, volume, OrderTypes.SELL, self._time),
                                                        self.bid, self._time))
        else:
            self._orders_list.append(Order(price, sl, tp, volume, OrderTypes.SELL, self._time))

    def close_all_positions(self):
        """
        Close all opened positions.
        :return: [None]
        """
        for position in self._opened_positions_list[:]:
            position.last_price = self.bid if position.open_order.order_type == OrderTypes.BUY else self.ask
            position.close_time = self._time
            self._opened_positions_list.remove(position)
            self._history_position_list.append(position)

    def _check_orders(self):
        """
        For each market iteration, check if some order must be executed or not.
        :return: [None]
        """
        for order in self._orders_list[:]:
            # order.price == 0 means at market
            if order.order_type == OrderTypes.BUY and (self.ask >= order.price or order.price == 0):
                self._opened_positions_list.append(Position(order, self.ask, self._time))
                self._orders_list.remove(order)
            elif order.order_type == OrderTypes.SELL and (self.bid <= order.price or order.price == 0):
                self._opened_positions_list.append(Position(order, self.bid, self._time))
                self._orders_list.remove(order)

    def _check_positions(self):
        """
        For each market iteration, check if some position must be closed or not.
        :return: [None]
        """
        for position in self._opened_positions_list[:]:
            if position.open_order.order_type == OrderTypes.BUY:
                position.last_price = self.bid
                if position.open_order.sl != 0 and position.open_order.sl >= self.bid:
                    position.last_price = position.open_order.sl
                    position.close_time = self._time
                    self._opened_positions_list.remove(position)
                    self._history_position_list.append(position)
                elif position.open_order.tp != 0 and position.open_order.tp <= self.bid:
                    position.last_price = position.open_order.tp
                    position.close_time = self._time
                    self._opened_positions_list.remove(position)
                    self._history_position_list.append(position)
            else:
                position.last_price = self.ask
                if position.open_order.sl != 0 and position.open_order.sl <= self.ask:
                    position.last_price = position.open_order.sl
                    position.close_time = self._time
                    self._opened_positions_list.remove(position)
                    self._history_position_list.append(position)
                elif position.open_order.tp != 0 and position.open_order.tp >= self.ask:
                    position.last_price = position.open_order.tp
                    position.close_time = self._time
                    self._opened_positions_list.remove(position)
                    self._history_position_list.append(position)

    def _broker_callback(self):
        """
        Overload the abstract method inherited from Market class
        :return: [None]
        """
        self._check_orders()
        self._check_positions()

    @property
    def has_order(self):
        """
        Check if have a order waiting to be executed.
        :return: [bool] True if has a opened order, false if not.
        """
        return True if len(self._orders_list) > 0 else False

    @property
    def has_position(self):
        """
        Check if have a opened position.
        :return: [bool] True if has a opened position, false if not.
        """
        return True if len(self._opened_positions_list) > 0 else False

    @property
    def balance(self):
        """
        Return the balance of the account in the current currency.
        :return: [float]
        """
        return self._init_deposit + sum(
            [position.profit * position.open_order.volume for position in
             self._history_position_list]) / self.points2currency

    @property
    def history_positions(self):
        """
        A list containing all closed positions.
        :return: [list[Position]] A list of Positions instances.
        """
        return self._history_position_list.copy()

    @property
    def opened_positions(self):
        return self._opened_positions_list.copy()


if __name__ == "__main__":
    broker = Broker(csv_file_path="../datasets/WIN$N_m5.csv", window_size=20)
    broker.points2currency = 5

    from tqdm import tqdm
    import random

    for data in tqdm(broker):
        if random.randint(0, 100) == 0 and not broker.has_position:
            if random.random() > 0.5:
                _price = broker.ask
                broker.buy(sl=_price - 100, tp=_price + 100)
            else:
                _price = broker.bid
                broker.sell(sl=_price + 100, tp=_price - 100)
        if len(broker.history_positions) > 10:
            break

    debug = broker.history_positions
    print("finish")
