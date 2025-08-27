import numpy as np
from typing import Iterable
from src.pattern import Pattern


class Hopfield:
    def __init__(self, n: int):
        self.W = np.zeros((n, n), dtype=float)
        self.n = n

    def train(self, P: np.ndarray, etha: float = 1.0):
        self.W = etha * (P.T @ P) / self.n
        np.fill_diagonal(self.W, 0)

    def is_stable(self, p: Pattern) -> bool:
        self._validate_size(len(p))
        ss = self._sign_vec(self.W @ p.data)
        return np.array_equal(ss, p.data)

    def recall(self, p: Pattern) -> tuple[Pattern, int]:
        self._validate_size(n := len(p))

        p = p.copy()
        idxs = np.arange(n, dtype=int)
        modified = True
        steps = 0

        while modified:
            modified = False
            np.random.shuffle(idxs)
            steps += 1

            for i in idxs:
                s = self._eval(p, i)
                modified |= p[i] != s
                p[i] = s

        return p, steps

    def __len__(self) -> int:
        return self.n

    def _out(self, p: Pattern, i: int) -> float:
        return self.W[i] @ p.data

    def _sign(self, x: float) -> int:
        return 1 if x >= 0 else -1

    def _sign_vec(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1, -1)

    def _eval(self, p: Pattern, i: int) -> int:
        return self._sign(self._out(p, i))

    def _validate_size(self, n: int):
        if self.n != n:
            raise ValueError(f"size of pattern {n} doesn't match network size {self.n}")
