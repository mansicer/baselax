import gym
import numpy as np

from stable_baselines3 import DQN

env = gym.make("CartPole-v1")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5_000_000, log_interval=100)

returns_list = []
for _ in range(50):
    obs = env.reset()
    returns = 0.0
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        returns += reward
    returns_list.append(returns)
print(f"Final average return: {np.mean(returns_list)}")

env.close()
