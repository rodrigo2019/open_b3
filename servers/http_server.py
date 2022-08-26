import argparse
import datetime
import json
import logging
import socket
import sys
import time
from datetime import datetime, timedelta
from threading import Thread

import pandas as pd
import requests
from flask import Flask, request
from gevent.pywsgi import WSGIServer
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

sys.path.append("../")
from stock_market_simulator.envs import SingleStockTradingEnv  # noqa: E402

MODEL = "PPO"
MODEL_PATH = "../training/drl/trains/PPO_2/best_model.zip"

parser = argparse.ArgumentParser()
parser.add_argument('--bind-all', action="store_true", help='Use webserver on local network')

args = parser.parse_args()

__version__ = "0.0.1"
app = Flask(__name__)

# enable logging
log_level = logging.DEBUG
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(log_level)
stream_handler.setFormatter(formatter)

file_handler = logging.FileHandler("./webserver.log")
file_handler.setLevel(log_level)
file_handler.setFormatter(formatter)

logger = logging.getLogger("webserver")
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
logger.setLevel(log_level)

models = {
    "A2C": A2C,
    "DDPG": DDPG,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}

agent = models[MODEL].load("../training/drl/trains/PPO_2/best_model.zip")


@app.route("/get-action", methods=['GET'])
def get_action():
    try:
        start_time = time.time()
        json_dict = request.get_json()
        df = pd.DataFrame(json_dict["data"])

        env = SingleStockTradingEnv(data_frame=df, window_size=20, ignore_spread=True)
        observation = env.goto_iteration(len(env) - 1)
        action, _ = agent.predict(observation, deterministic=True)
        buy_or_sell, take_profit, stop_loss = env.compute_action(action)

        result = {"long_or_short": bool(round(buy_or_sell)), "take_profit": float(take_profit),
                  "stop_loss": float(stop_loss), "price": 0}

        logger.debug(f"Processing time: {time.time() - start_time:.4f} seconds")

        return json.dumps(result), 200, {"Content-Type": "application/json"}
    except Exception as e:
        logger.error(e, exc_info=True)
        return json.dumps({"error": f"{e}", "error_type": f"{type(e)}"}), 400, {'Content-Type': 'application/json'}


def _run_wsgi(ip):
    server = WSGIServer((ip, 5000), app)
    server.serve_forever()


def _log_uptime(update_time):
    started_time = time.time()
    while True:
        time.sleep(update_time)
        uptime = int(time.time() - started_time)
        uptime = str(timedelta(seconds=uptime))
        logger.info(f"Uptime: {uptime}")


def _self_test():
    try:
        timestamp = datetime.now()
        _size = 200
        data = {"data": {
            "time": [(timestamp + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(_size)],
            "high": [10] * _size,
            "low": [3] * _size,
            "close": [7] * _size,
            "open": [5] * _size,
            "tick_volume": [10] * _size,
            "spread": [0] * _size,
            "real_volume": [10] * _size,
        }}
        response = requests.get("http://localhost:5000/get-action", json=data)
        if response.status_code != 200:
            raise Exception(f"Response status code is {response.status_code}")
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e


if __name__ == "__main__":
    try:
        logger.info(f"websever version: {__version__}; lib version: xxx")
        if args.bind_all:
            ip_ = socket.gethostbyname(socket.gethostname())
        else:
            ip_ = "127.0.0.1"
        webserver_thread = Thread(target=_run_wsgi, args=(ip_,))
        webserver_thread.daemon = True
        webserver_thread.start()

        _self_test()
        logger.info(f"Webserver running. ({ip_})\n\n\n")
        _log_uptime(300)  # loop forever
    except Exception as e_:
        logger.critical(e_, exc_info=True)
        raise e_
