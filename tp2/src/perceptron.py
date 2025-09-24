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
        max_iter: int = 100,
        epoch_f: Callable[[Self], None] = lambda _: None,
        eta: float = 0.5,
    ):
        idxs = np.arange(len(X))
        epoch_f(self)
        epoch = 0

        while epoch < max_iter and np.any(self.mse(X, Y) > 0.0):
            self.rng.shuffle(idxs)

            for i in idxs:
                dyz = float(Y[i]) - self.forward(X[i])
                self.W += eta * dyz * X[i]
                self.B += eta * dyz

            epoch_f(self)
            epoch += 1

    def mse(self, X: np.ndarray, Y: np.ndarray) -> float:
        Z = self.forward(X)
        return float(np.mean((Y.flatten() - Z) ** 2))

    @classmethod
    def capacity(
        cls,
        n: int,
        n_p: int,
        n_t: int,
        seed: int | None = None,
        max_iter: int = 100,
    ) -> float:
        rng = np.random.default_rng(seed)
        p = cls(n)
        hits = 0

        for _ in range(n_t):
            X = rng.uniform(-1, 1, (n_p, n))
            Y = rng.choice([-1, 1], n_p)

            p.train(X, Y, max_iter=max_iter)
            hits += int(p.mse(X, Y) == 0.0)

        return hits / n_t
