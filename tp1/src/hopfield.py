import numpy as np


class Hopfield:
    def __init__(self, n: int):
        self.W = np.zeros((n, n), dtype=float)
        self.n = n

    @classmethod
    def capacity(cls, p_error: float, n: int) -> int:
        a, b = 1, n
        best = 0

        while a <= b:
            mid = (a + b) >> 1

            m = cls(n)
            P = np.random.choice([-1, 1], size=(mid, n))
            m.train(P)

            S = m._sign(P @ m.W)
            e_total = np.sum(S != P) / (n * mid)

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
        ss = self._sign(self.W @ p)
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
                s = self._sign(self.W[i] @ p)
                modified |= p[i] != s
                p[i] = s

        return p, steps

    def __len__(self) -> int:
        return self.n

    def _sign(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1, -1)
