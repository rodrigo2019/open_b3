import os
import sys
from typing import Dict, Any, Callable, Union

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from torch import nn as nn

sys.path.append("../../")
from stock_market_simulator.envs import SingleStockTradingEnv  # noqa: E402

TRAIN_PATH = "./train_files/WIN$N_m30.csv"
VAL_PATH = "./val_files/WIN$N_m30.csv"
MODEL = "PPO"  # working just for PPO for now
STUDY_NAME = "ppo_hyperparam_tuning"
SERVER_URL = "mysql://optuna:optuna@localhost/optuna"


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    # ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        # "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


def sample_env_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for ENV hyperparams.
    :param trial:
    :return:
    """
    window_size = trial.suggest_int("window_size", 1, 120, 1)
    return {"window_size": window_size}


def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    model_params = sample_ppo_params(trial)
    env_params = sample_env_params(trial)
    train_env = SingleStockTradingEnv(data_frame=TRAIN_PATH, ignore_spread=True,
                                      one_trade_at_time=True,
                                      **env_params)
    val_env = SingleStockTradingEnv(data_frame=VAL_PATH, ignore_spread=True,
                                    one_trade_at_time=True,
                                    **env_params)

    env = make_vec_env(lambda: train_env, n_envs=16, seed=0)
    model = PPO("MlpPolicy", env, verbose=0, **model_params)
    model.learn(100_000)
    mean_reward, _ = evaluate_policy(model, val_env, n_eval_episodes=10, deterministic=False)

    return mean_reward * -1


if __name__ == '__main__':
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=SERVER_URL)
    except KeyError:
        study = optuna.create_study(study_name=STUDY_NAME, storage=SERVER_URL)

    study.optimize(optimize_agent, catch=(ValueError,))
