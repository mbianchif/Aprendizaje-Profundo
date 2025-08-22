from typing import Self


class Pattern:
    def __init__(self, name: str, data: list[int]):
        self.name = name
        self._validate_data(data)
        self._data = data

    def copy(self) -> Self:
        return self.__class__(self.name, self._data.copy())

    def __getitem__(self, i: int) -> int:
        return self._data[i]

    def __setitem__(self, i: int, x: int):
        self._data[i] = x

    def __len__(self) -> int:
        return len(self._data)

    def __neg__(self) -> Self:
        negated = [-bit for bit in self._data]
        return self.__class__(f"neg({self.name})", negated)

    def __repr__(self) -> str:
        return f"{self.name}: {self._data}"

    def _validate_data(self, data: list[int]):
        for b in data:
            if abs(b) != 1:
                raise ValueError(f"invalid bit value {b} for {self.name}")
