import numpy as np

from src.mlp import MLP


def mse(m: MLP, X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.mean((m.forward(X) - Y) ** 2))


def cross_entropy(m: MLP, X: np.ndarray, Y: np.ndarray) -> float:
    eps = 1e-12
    Z = np.clip(m.forward(X), eps, 1 - eps)
    return float(-np.mean(np.sum(Y * np.log(Z), axis=1)))
