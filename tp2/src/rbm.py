from typing import Optional
import numpy as np


class RBM:
    def __init__(self, v: int, h: int, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

        limit = 6 / (v + h)
        self.W = self._rng.uniform(-limit, limit, size=(v, h))
        self.Bv = np.zeros((1, v))
        self.Bh = np.zeros((1, h))

    def _g(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def _data_pass(self, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        V = self._rng.normal(loc=M, scale=1)
        P = self._g(V @ self.W + self.Bh)

        r = self._rng.uniform(0, 1)
        H = np.where(r < P, 1, 0)
        return V, H

    def _recon_pass(self, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M = H @ self.W.T + self.Bv
        return self._data_pass(M)

    def forward(self, M: np.ndarray) -> np.ndarray:
        P = self._g(M @ self.W + self.Bh)
        r = self._rng.uniform(0, 1)
        H = np.where(r < P, 1, 0)
        return self._g(H @ self.W.T + self.Bv)

    def train(
        self,
        M: np.ndarray,
        batch_size: Optional[int] = None,
        max_iter: int = 5000,
        lr: float = 0.01,
        err: float = 1e-4,
    ):
        n = len(M)
        idxs = np.arange(n)
        batch_size = batch_size or n

        for _ in range(max_iter):
            self._rng.shuffle(idxs)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)

                Vd, Hd = self._data_pass(M[start:end])
                Vr, Hr = self._recon_pass(Hd)

                self.W += lr * np.mean(Vd.T @ Hd - Vr.T @ Hr)
                self.Bv += lr * np.mean(Vd - Vr)
                self.Bh += lr * np.mean(Hd - Hr)

            if self.mse(M) <= err:
                break

    def mse(self, M: np.ndarray) -> float:
        return float(np.mean((self.forward(M) - M) ** 2))
