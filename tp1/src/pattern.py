import numpy as np
from math import sqrt
from typing import Iterable, Self


class Pattern:
    def __init__(self, name: str, data: np.ndarray, height: int, width: int):
        self.name = name
        self._validate_data(data)
        self.data = data
        self.height = height
        self.width = width

    @classmethod
    def random(cls, n: int) -> Self:
        data = np.random.choice([-1, 1], size=n)
        return cls("random", data, int(sqrt(n)), int(sqrt(n)))

    def copy(self) -> Self:
        return self.__class__(self.name, self.data.copy(), self.height, self.width)

    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def __getitem__(self, i: int) -> int:
        return self.data[i]

    def __setitem__(self, i: int, x: int):
        self.data[i] = x

    def __len__(self) -> int:
        return len(self.data)

    def __neg__(self) -> Self:
        negated = np.array([-bit for bit in self.data])
        return self.__class__(f"neg({self.name})", negated, self.height, self.width)

    def __repr__(self) -> str:
        return f"{self.name}: {self.data}"

    def __iter__(self) -> Iterable[int]:
        return iter(self.data)

    def _validate_data(self, data: np.ndarray):
        for b in data:
            if abs(b) != 1:
                raise ValueError(f"invalid bit value {b} for {self.name}")
