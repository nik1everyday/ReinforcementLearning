from typing import List
import numpy as np


def mean(values: List[float], probs: List[float] = None) -> float:
    if probs is None:
        return np.mean(values)

    if len(values) != len(probs):
        raise AssertionError("Number of values must be equal to the number of probabilities")

    return np.sum(np.array(values) * np.array(probs))
