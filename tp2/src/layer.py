from abc import ABC, abstractmethod
from typing import Callable
import numpy as np


class Layer(ABC):
    @abstractmethod
    def weights(self) -> np.ndarray:
        pass

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(
        self, Wnext: np.ndarray, Dnext: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def backward_last(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def apply_delta(self, lr: float):
        pass


class Dense(Layer):
    def __init__(
        self,
        n: int,
        m: int,
        g: Callable[[np.ndarray], np.ndarray],
        dg: Callable[[np.ndarray], np.ndarray],
        seed: int | None = None,
    ):
        rng = np.random.default_rng(seed)

        self.X = np.zeros(0)
        self.W = rng.normal(loc=0, scale=0.1, size=(m, n))
        self.B = rng.normal(loc=0, scale=0.1, size=(n,))
        self.H = np.zeros(0)
        self.Y = np.zeros(0)

        self.dW = np.zeros((n, m))
        self.dB = np.zeros((n,))

        self.g = g
        self.dg = dg

    def weights(self) -> np.ndarray:
        return self.W

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        self.H = X @ self.W + self.B
        self.Y = self.g(self.H)
        return self.Y

    def backward(
        self,
        Wnext: np.ndarray,
        Dnext: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        D = self.dg(self.H) * (Wnext @ Dnext)
        self.dW += self.X.T @ D
        self.dB += np.sum(D, axis=0)
        return self.W, D

    def backward_last(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        D = -self.dg(self.H) * (Y - self.Y)
        self.dW += self.X.T @ D
        return self.W, D

    def apply_delta(self, lr: float):
        self.W -= lr * self.dW
        self.dW.fill(0)

        self.B -= lr * self.dB
        self.dB.fill(0)
