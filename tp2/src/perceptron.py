from typing import Callable, Self
import numpy as np


class Perceptron:
    def __init__(self, n: int):
        self.W = np.zeros(shape=(n, 1))
        self.B = 0.0

    def forward(self, X: np.ndarray) -> np.ndarray:
        H = X @ self.W + self.B
        Z = np.where(H >= 0, 1, -1)
        return Z

    def train(
        self, X: np.ndarray, Y: np.ndarray, f: Callable[[Self], None], etha: float = 1.0
    ):
        while True:
            f(self)
            Z = self.forward(X)
            self.W += etha * X.T @ (Y - Z)
            self.B += etha * np.sum(Y - Z)

            if self.mse(X, Y) == 0.0:
                f(self)
                break

    def mse(self, X: np.ndarray, Y: np.ndarray) -> float:
        Z = self.forward(X)
        return 0.5 * np.sum(np.square(Y - Z))
