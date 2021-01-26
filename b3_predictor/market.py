import pandas as pd
from abc import ABC, abstractmethod
from enum import IntEnum


class OrderTypes(IntEnum):
    BUY = 0
    SELL = 1


class Order:
    _ticket_counter = 0

    def __init__(self, price, sl, tp, volume, order_type):
        self._price = price
        self._sl = sl
        self._tp = tp
        self._volume = volume
        self._order_type = order_type

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
    def ticket(self):
        return self._ticket


class Position:
    _ticket_counter = 0

    def __init__(self, open_order, opened_price):
        self._open_order = open_order
        self._opened_price = opened_price
        self._last_price = 0

        self._ticket = Position._ticket_counter
        Position._ticket_counter += 1

    @property
    def open_order(self):
        return self._open_order

    @property
    def last_price(self):
        return self._last_price

    @last_price.setter
    def last_price(self, new_price):
        assert new_price > 0, "The new price must be a positive value"
        self._last_price = new_price

    @property
    def opened_price(self):
        return self._opened_price

    @property
    def profit(self):
        if self._open_order.order_type == OrderTypes.BUY:
            return self._last_price - self._opened_price
        else:
            return self._opened_price - self._last_price

    @property
    def ticket(self):
        return self._ticket


class Market(ABC):
    def __init__(self, csv_file_path, window_size=1):
        assert window_size > 0, "window_size must be > 0."
        self._window_size = window_size

        self._data_frame = pd.read_csv(csv_file_path)
        self._data_frame["time"] = pd.to_datetime(self._data_frame["time"])
        self._data_frame = self._data_frame.set_index("time")

        self._points2currency = 1
        self._iter = 0

    def __iter__(self):
        for self._iter in range(self._window_size, len(self._data_frame), 1):
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
    def points2currency(self):
        return self._points2currency

    @points2currency.setter
    def points2currency(self, new_value):
        assert new_value >= 0, "points2currency value must be >= 0."

    @abstractmethod
    def _broker_callback(self):
        pass


class Broker(Market):
    def __init__(self, *args, init_deposit=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_deposit = init_deposit

        self._orders_list = []
        self._opened_positions_list = []
        self._history_position_list = []

    def buy(self, price=0, sl=0, tp=0, volume=1):
        if price == 0:
            self._opened_positions_list.append(Position(Order(price, sl, tp, volume, OrderTypes.BUY), self.ask))
        else:
            self._orders_list.append(Order(price, sl, tp, volume, OrderTypes.BUY))

    def sell(self, price=0, sl=0, tp=0, volume=1):
        if price == 0:
            self._opened_positions_list.append(Position(Order(price, sl, tp, volume, OrderTypes.SELL), self.bid))
        else:
            self._orders_list.append(Order(price, sl, tp, volume, OrderTypes.SELL))

    def close_all_positions(self):
        for position in self._opened_positions_list[:]:
            position.last_price = self.bid if position.open_order.order_type == OrderTypes.BUY else self.ask
            self._opened_positions_list.remove(position)
            self._history_position_list.append(position)

    def _check_orders(self):
        for order in self._orders_list[:]:
            # order.price == 0 means at market
            if order.order_type == OrderTypes.BUY and (self.ask >= order.price or order.price == 0):
                self._opened_positions_list.append(Position(order, self.ask))
                self._orders_list.remove(order)
            elif order.order_type == OrderTypes.SELL and (self.bid <= order.price or order.price == 0):
                self._opened_positions_list.append(Position(order, self.bid))
                self._orders_list.remove(order)

    def _check_positions(self):
        for position in self._opened_positions_list[:]:
            if position.open_order.order_type == OrderTypes.BUY:
                if position.open_order.sl != 0 and position.open_order.sl >= self.bid:
                    position.last_price = position.open_order.sl
                    self._opened_positions_list.remove(position)
                    self._history_position_list.append(position)
                elif position.open_order.tp != 0 and position.open_order.tp <= self.bid:
                    position.last_price = position.open_order.tp
                    self._opened_positions_list.remove(position)
                    self._history_position_list.append(position)
            else:
                position.last_price = self.ask
                if position.open_order.sl != 0 and position.open_order.sl <= self.ask:
                    position.last_price = position.open_order.sl
                    self._opened_positions_list.remove(position)
                    self._history_position_list.append(position)
                elif position.open_order.tp != 0 and position.open_order.tp >= self.ask:
                    position.last_price = position.open_order.tp
                    self._opened_positions_list.remove(position)
                    self._history_position_list.append(position)

    def _broker_callback(self):
        self._check_orders()
        self._check_positions()

    @property
    def has_order(self):
        return True if len(self._orders_list) > 0 else False

    @property
    def has_position(self):
        return True if len(self._opened_positions_list) > 0 else False

    @property
    def balance(self):
        return self._init_deposit + sum(
            [position.profit * position.open_order.volume for position in
             self._history_position_list]) / self.points2currency

    @property
    def history_positions(self):
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
