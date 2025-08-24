from typing import Iterable, Self


class Pattern:
    def __init__(self, name: str, data: list[int], height: int, width: int):
        self.name = name
        self._validate_data(data)
        self._data = data
        self.height = height
        self.width = width

    def copy(self) -> Self:
        return self.__class__(self.name, self._data.copy(), self.height, self.width)

    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def __getitem__(self, i: int) -> int:
        return self._data[i]

    def __setitem__(self, i: int, x: int):
        self._data[i] = x

    def __len__(self) -> int:
        return len(self._data)

    def __neg__(self) -> Self:
        negated = [-bit for bit in self._data]
        return self.__class__(f"neg({self.name})", negated, self.height, self.width)

    def __repr__(self) -> str:
        return f"{self.name}: {self._data}"

    def __iter__(self) -> Iterable[int]:
        return iter(self._data)

    def _validate_data(self, data: list[int]):
        for b in data:
            if abs(b) != 1:
                raise ValueError(f"invalid bit value {b} for {self.name}")
