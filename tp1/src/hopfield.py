from typing import Iterable
from src.pattern import Pattern
import random


class Hopfield:
    def __init__(self, n: int):
        self.ws = [[0.0] * n for _ in range(n)]
        self.n = n

    def train(self, ps: Iterable[Pattern], etha: float = 1.0):
        for p in ps:
            self._validate_size(len(p))

        n = self.n

        for i in range(n):
            for j in range(i + 1, n):
                dwij = etha * sum(p[i] * p[j] for p in ps) / n
                self.ws[i][j] = dwij
                self.ws[j][i] = dwij

    def is_stable(self, p: Pattern) -> bool:
        self._validate_size(n := len(p))
        return all(self._eval(p, i) == p[i] for i in range(n))

    def recall(self, p: Pattern) -> tuple[Pattern, int]:
        self._validate_size(n := len(p))

        p = p.copy()
        idxs = list(range(n))
        modified = True
        steps = 0

        while modified:
            modified = False
            random.shuffle(idxs)
            steps += 1

            for i in idxs:
                s = self._eval(p, i)
                modified |= p[i] != s
                p[i] = s

        return p, steps

    def _out(self, p: Pattern, i: int) -> float:
        return sum(self.ws[i][j] * p[j] for j in range(len(p)) if i != j)

    def _sign(self, x: float) -> int:
        return 1 if x >= 0 else -1

    def _eval(self, p: Pattern, i: int) -> int:
        return self._sign(self._out(p, i))

    def _validate_size(self, n: int):
        if self.n != n:
            raise ValueError(f"size of pattern {n} doesn't match network size {self.n}")
