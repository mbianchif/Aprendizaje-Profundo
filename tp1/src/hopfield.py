import random
from src.pattern import Pattern


class Hopfield:
    def __init__(self, n: int):
        self.ws = [[0] * n for _ in range(n)]

    def train(self, p: Pattern, etha: int = 1):
        idxs = list(range(len(p)))
        random.shuffle(idxs)

        for i in idxs:
            for j in idxs:
                if i != j:
                    dwij = etha * p[i] * p[j]
                    self.ws[i][j] += dwij

    def learned(self, p: Pattern) -> bool:
        return all(self._eval(p, i) == p[i] for i in range(len(p)))

    def converge(self, p: Pattern) -> int:
        n = len(p)
        idxs = list(range(n))
        modified = True
        steps = 0

        while modified:
            modified = False
            random.shuffle(idxs)

            for i in range(n):
                s = self._eval(p, i)
                modified |= p[i] != s
                p[i] = s

            steps += 1

        return steps

    def _sign(self, x: int) -> int:
        return 1 if x >= 0 else -1

    def _out(self, p: Pattern, i: int) -> int:
        return sum(self.ws[i][j] * p[j] for j in range(len(p)) if i != j)

    def _eval(self, p: Pattern, i: int) -> int:
        return self._sign(self._out(p, i))
