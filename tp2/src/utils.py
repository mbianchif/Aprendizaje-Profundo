import numpy as np
from typing import Callable


def truth_table(n: int, f: Callable[[list[int]], int]) -> tuple[np.ndarray, np.ndarray]:
    def bt(xs, ys, acc):
        if len(acc) == n:
            xs.append(np.array(acc))
            ys.append(np.array([f(acc)]))
            return xs, ys

        acc.append(1)
        bt(xs, ys, acc)
        acc.pop()

        acc.append(-1)
        bt(xs, ys, acc)
        acc.pop()

        return xs, ys

    xs, ys = bt([], [], [])
    return np.array(xs), np.array(ys)
