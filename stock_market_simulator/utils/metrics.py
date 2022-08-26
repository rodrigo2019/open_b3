import numpy as np


def _std_dev(data):
    # Get number of observations
    n = len(data)
    # Calculate mean
    mean = sum(data) / n
    # Calculate deviations from the mean
    deviations = sum([(x - mean) ** 2 for x in data])
    # Calculate Variance & Standard Deviation

    variance = deviations / (n - 1)
    s = variance ** (1 / 2)
    return s


def sharpe_ratio(data, risk_free_rate=0.0):
    # Calculate Average Daily Return
    mean_daily_return = sum(data) / len(data)
    # Calculate Standard Deviation
    s = _std_dev(data)
    # Calculate Daily Sharpe Ratio
    daily_sharpe_ratio = (mean_daily_return - risk_free_rate) / s
    # Annualize Daily Sharpe Ratio

    # return 252 ** (1 / 2) * daily_sharpe_ratio
    return daily_sharpe_ratio * len(data) ** (1 / 2)
