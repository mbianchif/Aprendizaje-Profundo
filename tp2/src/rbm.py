from typing import Callable, Optional, Self
import numpy as np


class RBM:
    def __init__(self, v: int, h: int, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

        self.W = self._rng.normal(0, 0.01, size=(v, h))
        self.Bv = np.zeros((1, v))
        self.Bh = np.zeros((1, h))

    def _g(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def forward(self, M: np.ndarray) -> np.ndarray:
        H = self._g(M @ self.W + self.Bh)
        return self._g(H @ self.W.T + self.Bv)

    def train(
        self,
        M: np.ndarray,
        batch_size: Optional[int] = None,
        max_iter: int = 5000,
        lr: float = 0.01,
        err: float = 1e-4,
        epoch_f: Callable[[Self], None] = lambda _: None,
    ):
        n = len(M)
        idxs = np.arange(n)
        batch_size = batch_size or n

        epoch_f(self)

        for _ in range(max_iter):
            self._rng.shuffle(idxs)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = idxs[start:end]

                Vd = M[batch]
                Pd = self._g(Vd @ self.W + self.Bh)
                Hd = (self._rng.uniform(0, 1, Pd.shape) < Pd).astype(float)

                Pr = self._g(Hd @ self.W.T + self.Bv)

                Vr = Pr
                Ph = self._g(Vr @ self.W + self.Bh)

                self.W += lr * ((Vd.T @ Pd) - (Vr.T @ Ph)) / (end - start)
                self.Bv += lr * np.mean(Vd - Vr, axis=0, keepdims=True)
                self.Bh += lr * np.mean(Pd - Ph, axis=0, keepdims=True)

            epoch_f(self)

            if self.mse(M) <= err:
                break

    def mse(self, M: np.ndarray) -> float:
        return float(np.mean((self.forward(M) - M) ** 2))
