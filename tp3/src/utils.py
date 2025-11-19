import numpy as np
from typing import Optional


def uniform_circle(n: int, radius: float, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n):
        r = rng.uniform(0, radius)
        theta = rng.uniform(0, 360)
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
