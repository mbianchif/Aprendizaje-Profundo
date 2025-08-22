from typing import Iterable
from src.pattern import Pattern
import random


class Hopfield:
    def __init__(self, n: int):
        self.ws = [[0] * n for _ in range(n)]
        self.n = n

    @classmethod
    def sign(cls, x: int) -> int:
        return 1 if x >= 0 else -1

    def train(self, p: Pattern, etha: int = 1):
        self._validate_size(n := len(p))

        for i in range(n):
            for j in range(n):
                if i != j:
                    self.ws[i][j] += etha * p[i] * p[j]

    def train_batch(self, ps: Iterable[Pattern], etha: int = 1):
        for p in ps:
            self.train(p, etha)

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

    def _out(self, p: Pattern, i: int) -> int:
        return sum(self.ws[i][j] * p[j] for j in range(len(p)) if i != j)

    def _eval(self, p: Pattern, i: int) -> int:
        return self.__class__.sign(self._out(p, i))

    def _validate_size(self, n: int):
        if self.n != n:
            raise ValueError(f"size of pattern {n} doesn't match network size {self.n}")
