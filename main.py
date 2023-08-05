from src.train import train
from src.environment import EnergyEnvironment


if __name__ == "__main__":
    # Create an instance of the EnergyEnvironment class
    env = EnergyEnvironment(time_period=24, max_energy=1000)

    # Start training
    train(env)
