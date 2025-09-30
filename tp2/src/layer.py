from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, D: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def apply_delta(self, lr: float, batch_size: int):
        pass


class Tanh(Layer):
    def __init__(self, m: int):
        self.Y = np.zeros((1, m))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Y = np.tanh(X)
        return self.Y

    def backward(self, D: np.ndarray) -> np.ndarray:
        dZ = 1 - self.Y**2
        return D * dZ

    def apply_delta(self, lr: float, batch_size: int):
        pass


class Dense(Layer):
    def __init__(
        self,
        n: int,
        m: int,
        seed: int | None = None,
    ):
        rng = np.random.default_rng(seed)

        self.X = np.zeros((1, n))
        self.W = rng.normal(loc=0, scale=0.1, size=(n, m))
        self.B = rng.normal(loc=0, scale=0.1, size=(1, m))

        self.dW = np.zeros_like(self.W)
        self.dB = np.zeros_like(self.B)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return X @ self.W + self.B

    def backward(self, D: np.ndarray) -> np.ndarray:
        self.dW += self.X.T @ D
        self.dB += np.sum(D, axis=0)
        return D @ self.W.T

    def apply_delta(self, lr: float, batch_size: int):
        k = lr / batch_size
        self.W -= k * self.dW
        self.B -= k * self.dB
        self.dW.fill(0)
        self.dB.fill(0)
