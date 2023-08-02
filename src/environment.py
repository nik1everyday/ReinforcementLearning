import torch.nn as nn
import torch.nn.functional as F

from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from typing import Tuple


class EnergyEnvironment(Env):

    def __init__(self, time_period: int, max_energy: int):
        super(EnergyEnvironment, self).__init__()
        self.time_period = time_period  # Временной период в часах
        self.max_energy = max_energy  # Максимальная энергия в МВт*ч

        # Источники энергии
        self.energy_sources = [
            {'name': 'coal',  # источник энергии
             'power_per_hour': 500,  # МВт*ч в час
             'CO2': 1500},  # граммов СО2 на кВт*ч
            {'name': 'wind', 'power_per_hour': 250, 'CO2': 12},
            {'name': 'solar', 'power_per_hour': 400, 'CO2': 0},
            {'name': 'hydro', 'power_per_hour': 800, 'CO2': 4},
            {'name': 'nuclear', 'power_per_hour': 1450, 'CO2': 3}
        ]

        # Дискретное действие - источник энергии
        self.action_space = Discrete(len(self.energy_sources))

        # Непрерывное действие - мощность
        self.power_space = Box(0, max(s['power_per_hour'] for s in self.energy_sources), shape=(1,))

        self.time = 0
        self.energy = 0

    def step(self, action):
        energy = action[0]
        co2 = self.energy_sources[energy]['CO2'] * action[1] / 1000
        self.energy += action[1]
        self.time += 1
        observation = (self.time, self.energy, co2)

        if co2 == 0:
            reward = 1
        else:
            reward = 1 / co2
        done = (self.energy >= self.max_energy) or (self.time >= self.time_period)
        info = {}
        return observation, reward, done, info

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            random.seed(seed)

        self.time = 0
        self.energy = 0

        observation = np.array((0, 0, 0))
        info = {}

        return observation, info
