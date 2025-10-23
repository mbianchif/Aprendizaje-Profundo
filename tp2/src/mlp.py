from typing import Callable, Optional, Self
from src.layer import Layer
import numpy as np


class MLP:
    def __init__(
        self,
        arch: list[Layer],
        err_f: Callable[[Self, np.ndarray, np.ndarray], float],
        seed: Optional[int] = None,
    ):
        self._rng = np.random.default_rng(seed)
        self._err_f = err_f

        for layer in arch:
            layer.init(self._rng)

        self._layers = arch

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self._layers:
            X = layer.forward(X)

        return X

    def backward(self, Z: np.ndarray, Y: np.ndarray):
        D = (Z - Y) / len(Y)

        for layer in reversed(self._layers):
            D = layer.backward(D)

    def apply_deltas(self, lr: float):
        for layer in self._layers:
            layer.apply_delta(lr)

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: Optional[int] = None,
        max_iter: int = 5000,
        lr: float = 0.01,
        err: float = 1e-4,
        epoch_f: Callable[[Self], None] = lambda _: None,
    ):
        n = len(X)
        idxs = np.arange(n)
        batch_size = batch_size or n

        epoch_f(self)

        for _ in range(max_iter):
            self._rng.shuffle(idxs)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = idxs[start:end]

                Z = self.forward(X[batch])
                self.backward(Z, Y[batch])
                self.apply_deltas(lr)

            epoch_f(self)

            if self.error(X, Y) <= err:
                break

    def error(self, X: np.ndarray, Y: np.ndarray) -> float:
        return self._err_f(self, X, Y)
