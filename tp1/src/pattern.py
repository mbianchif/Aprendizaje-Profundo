from typing import Callable, Self


class Pattern:
    def __init__(self, name: str, data: list[int]):
        self._data = data
        self.name = name

    def copy(self) -> Self:
        return self.__class__(self.name, self._data.copy())

    def normalize(self):
        for i in range(len(self)):
            if self[i] == 0:
                raise Exception("invalid value `0` found in pattern")

            self[i] //= abs(self[i])

    def __getitem__(self, i: int) -> int:
        return self._data[i]

    def __setitem__(self, i: int, x: int):
        self._data[i] = x

    def __len__(self) -> int:
        return len(self._data)

    def _map(self, p: Self, f: Callable) -> Self:
        c = self.copy()

        for i in range(len(self)):
            c[i] = f(c[i], p[i])

        return c

    def __add__(self, p: Self) -> Self:
        return self._map(p, lambda x, y: x + y)

    def __sub__(self, p: Self) -> Self:
        return self._map(p, lambda x, y: x - y)
