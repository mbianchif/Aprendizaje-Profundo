from typing import Callable
import numpy as np


class Hopfield:
    def __init__(self, n: int):
        self.W = np.zeros((n, n), dtype=float)
        self.n = n

    @classmethod
    def capacity(
        cls,
        n: int,
        p_error: float,
        p_prov: Callable[[int, int], np.ndarray],
        mask: np.ndarray | None = None,
    ) -> int:
        a, b = 1, n
        m = cls(n)
        best = 1

        while a <= b:
            mid = (a + b) >> 1

            P = p_prov(mid, n)
            m.train(P)

            if mask is not None:
                m.W *= mask

            S = np.where(P @ m.W >= 0.0, 1, -1)
            e_total = np.mean(S != P)

            if e_total < p_error:
                best = mid
                a = mid + 1
            else:
                b = mid - 1

        return best

    def train(self, P: np.ndarray):
        self.W = (P.T @ P) / self.n
        np.fill_diagonal(self.W, 0)

    def is_stable(self, p: np.ndarray) -> bool:
        ss = np.where(self.W @ p >= 0.0, 1, -1)
        return np.array_equal(ss, p)

    def recall(self, p: np.ndarray) -> tuple[np.ndarray, int]:
        p = p.copy()

        n = len(p)
        idxs = np.arange(n, dtype=int)
        modified = True
        steps = 0

        while modified:
            modified = False
            np.random.shuffle(idxs)
            steps += 1

            for i in idxs:
                s = np.where(self.W[i] @ p >= 0.0, 1, -1)
                modified |= p[i] != s
                p[i] = s

        return p, steps
