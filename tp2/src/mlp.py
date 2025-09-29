import numpy as np
from layer import Layer


class MLP:
    def __init__(self, arch: list[Layer], seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self._layers = arch

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self._layers:
            X = layer.forward(X)

        return X

    def backward(self, Y: np.ndarray):
        rest, last = self._layers[:-1], self._layers[-1]

        W, D = last.backward_last(Y)
        for layer in reversed(rest):
            W, D = layer.backward(W, D)

    def apply_deltas(self, lr: float):
        for layer in self._layers:
            layer.apply_delta(lr)

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        max_iter: int = 100,
        lr: float = 0.1,
        err: float = 0.05,
    ):
        idxs = np.arange(len(X))

        for epoch in range(max_iter):
            self._rng.shuffle(idxs)

            for i in idxs:
                self.forward(X[i])
                self.backward(Y[i])
                self.apply_deltas(lr / epoch)

            Z = self.forward(X)
            if self.mse(Z, Y) <= err:
                break

    def mse(self, Z: np.ndarray, Y: np.ndarray) -> float:
        return float(0.5 * np.mean(Y.flatten() - Z.flatten()) ** 2)
