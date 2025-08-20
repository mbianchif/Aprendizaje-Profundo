import random


class Hopfield:
    def __init__(self, n: int):
        self.ws = [[0] * n for _ in range(n)]

    def h(self, p: list[int], i: int) -> int:
        n = len(p)
        return sum(self.ws[i][j] * p[j] for j in range(n) if i != j)

    def g(self, x: int) -> int:
        return 1 if x >= 0 else -1

    def train(self, p: list[int], etha: int = 1):
        n = len(p)
        idxs = list(range(n))
        random.shuffle(idxs)

        for i in idxs:
            for j in idxs:
                if i != j:
                    dwij = etha * p[i] * p[j]
                    self.ws[i][j] += dwij

    def learnt(self, p: list[int]) -> bool:
        n = len(p)

        for i in range(n):
            s = self.g(self.h(p, i))
            if s != p[i]:
                return False

        return True

    def converge(self, p: list[int]) -> int:
        n = len(p)
        idxs = list(range(n))
        modified = True
        steps = 0

        while modified:
            modified = False
            random.shuffle(idxs)

            for i in range(n):
                s = self.g(self.h(p, i))
                if p[i] != s:
                    modified = True

                p[i] = s

            steps += 1

        return steps
