from src.train import train
from src.environment import EnergyEnvironment


if __name__ == "__main__":
    env = EnergyEnvironment(time_period=24, max_energy=1000)
    train(env)
