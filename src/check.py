from gym.envs.registration import register

from src.environment import EnergyEnvironment


register(
    id='EnergyEnvironment-v0',
    entry_point='environment:EnergyEnvironment',
)

env = EnergyEnvironment(time_period=24, max_energy=1000)
env.action_space.seed(42)

observation, info = env.reset(seed=42)

print("Checking is starting...")

for _ in range(1000):
    action = env.action_space.sample()
    power = env.power_space.sample()[0]
    observation, reward, terminated, info = env.step([action, power])

    if terminated:
        observation, info = env.reset()

    if _ % 100 == 0:
        print(f"Checking progress: {_/10}%")

print("Checking was finished successfully!")
env.close()