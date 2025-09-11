from typing import Callable, Self
import numpy as np


class Perceptron:
    def __init__(self, n: int, seed: int | None = None):
        rng = np.random.default_rng(seed)

        self.W = rng.normal(loc=0.0, scale=0.1, size=n)
        self.B = rng.normal(loc=0.0, scale=0.1)
        self.rng = rng
        self.n = n

    def forward(self, X: np.ndarray) -> np.ndarray:
        H = X @ self.W + self.B
        Z = np.where(H >= 0, 1, -1)
        return Z

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epoch_f: Callable[[Self], None] = lambda _: None,
        eta: float = 0.5,
    ):
        idxs = np.arange(len(X))
        epoch_f(self)

        while np.any(self.mse(X, Y) > 0.0):
            self.rng.shuffle(idxs)

            for i in idxs:
                dyz = float(Y[i]) - self.forward(X[i])
                self.W += eta * dyz * X[i]
                self.B += eta * dyz

            epoch_f(self)

    def mse(self, X: np.ndarray, Y: np.ndarray) -> float:
        Z = self.forward(X)
        return float(np.mean((Y.flatten() - Z) ** 2))
