import numpy as np
from src.layer import Layer


class MLP:
    def __init__(self, arch: list[Layer], seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self._layers = arch

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self._layers:
            X = layer.forward(X)

        return X

    def backward(self, Z: np.ndarray, Y: np.ndarray):
        D = 2 * (Z - Y)

        for layer in reversed(self._layers):
            D = layer.backward(D)

    def apply_deltas(self, lr: float, batch_size: int):
        for layer in self._layers:
            layer.apply_delta(lr, batch_size)

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        max_iter: int = 5000,
        lr: float = 0.01,
        err: float = 1e-4,
    ):
        n = len(X)
        idxs = np.arange(n)

        for _ in range(max_iter):
            self._rng.shuffle(idxs)

            for i in idxs:
                Z = self.forward(X[i].reshape(1, -1))
                self.backward(Z, Y[i].reshape(1, -1))

            self.apply_deltas(lr, 4)
            Z = self.forward(X)
            if self.mse(Z, Y) <= err:
                break

    def mse(self, Z: np.ndarray, Y: np.ndarray) -> float:
        return float(0.5 * np.mean((Z - Y) ** 2))
