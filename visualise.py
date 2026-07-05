from stable_baselines3 import PPO
from race_env import MultiRaceEnv

env = MultiRaceEnv(render_mode="human")

print("Loading models...")

model1 = PPO.load("car1")
model2 = PPO.load("car2")

(obs1, obs2), _ = env.reset()

running = True

while running:

    action1 = env.action_space.sample()
    action2 = env.action_space.sample()

    (obs1, obs2), rewards, done, trunc, info = env.step(
        (int(action1), int(action2))
    )

    env.render()

    if done:
        (obs1, obs2), _ = env.reset()

env.close()