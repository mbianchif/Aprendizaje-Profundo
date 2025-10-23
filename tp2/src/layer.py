from abc import ABC, abstractmethod
import numpy as np
from numpy.random import Generator


class Layer(ABC):
    @abstractmethod
    def init(self, rng: Generator):
        pass

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, D: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def apply_delta(self, lr: float):
        pass


class Tanh(Layer):
    def __init__(self, a: float = 1.0):
        self.a = a

    def init(self, rng: Generator):
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Y = np.tanh(X)
        return self.Y * self.a

    def backward(self, D: np.ndarray) -> np.ndarray:
        return D * self.a * (1 - self.Y**2)

    def apply_delta(self, lr: float):
        pass


class SoftMax(Layer):
    def init(self, rng: Generator):
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        exp_x = np.exp(X - X.max())
        return exp_x / exp_x.sum()

    def backward(self, D: np.ndarray) -> np.ndarray:
        return D

    def apply_delta(self, lr: float):
        pass


class Dense(Layer):
    def __init__(self, n: int, m: int):
        self.n, self.m = n, m

    def init(self, rng: Generator):
        limit = 6 / (self.n + self.m)
        self.W = rng.uniform(-limit, limit, size=(self.n, self.m))
        self.B = np.zeros((1, self.m))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return X @ self.W + self.B

    def backward(self, D: np.ndarray) -> np.ndarray:
        self.dW = self.X.T @ D / len(D)
        self.dB = np.mean(D, axis=0, keepdims=True)
        return D @ self.W.T

    def apply_delta(self, lr: float):
        self.W -= lr * self.dW
        self.B -= lr * self.dB
