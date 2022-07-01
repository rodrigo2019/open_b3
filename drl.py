from stock_market_simulator.envs import SingleStockTradingEnv
from stable_baselines3 import PPO

env = SingleStockTradingEnv(csv_file_path="./data/sin_wave_sample.csv", window_size=20)
env.drop_feature("volume")
env.drop_feature("tick_volume")
env.add_indicator("macd")
# env.add_indicator("rsi")
# env.add_indicator("boll_ub")
# env.add_indicator("boll_lb")


agent = PPO("MlpPolicy", env, verbose=1)

agent.learn(total_timesteps=100_000_000)


