from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from race_env import MultiRaceEnv


class Car1Env(MultiRaceEnv):

    def step(self, action):
        obs, rewards, done, trunc, info = super().step((action, 4))
        return obs[0], rewards[0], done, trunc, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        return obs[0], info


class Car2Env(MultiRaceEnv):

    def step(self, action):
        obs, rewards, done, trunc, info = super().step((4, action))
        return obs[1], rewards[1], done, trunc, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        return obs[1], info


print("Training Blue Car...")

env1 = DummyVecEnv([lambda: Monitor(Car1Env())])

model1 = PPO(
    "MlpPolicy",
    env1,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
)

model1.learn(total_timesteps=150000)
model1.save("car1")

print("Blue Done!")


print("Training Red Car...")

env2 = DummyVecEnv([lambda: Monitor(Car2Env())])

model2 = PPO(
    "MlpPolicy",
    env2,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
)

model2.learn(total_timesteps=150000)
model2.save("car2")

print("Red Done!")