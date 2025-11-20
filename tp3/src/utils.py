import numpy as np
from typing import Optional


def uniform_circle(n: int, radius: float, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n):
        r = radius * np.sqrt(rng.uniform(0, 1))
        theta = rng.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        samples.append((x, y))

    return np.array(samples)


def uniform_square(
    n: int,
    side_length: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a, b = 0, side_length
    samples = []

    for _ in range(n):
        x = rng.uniform(a, b)
        y = rng.uniform(a, b)
        samples.append((x, y))

    return np.array(samples)


def noisy_circunference(
    n: int,
    radius: float,
    radius_noise_percentage: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = radius * (1 - radius_noise_percentage)
    b = radius * (1 + radius_noise_percentage)
    samples = []

    for t in range(n):
        theta = 2 * np.pi * t / n
        r = rng.uniform(a, b)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        samples.append((x, y))

    return np.array(samples)
