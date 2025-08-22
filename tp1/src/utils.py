from src.pattern import Pattern
from typing import Callable, Generator, Iterable
from PIL import Image
import random
import os


def retrieve_patterns(path: str) -> Iterable[Pattern]:
    open_file = lambda x: Image.open(f"{path}/{x}").convert("1")

    for filename in os.listdir(path):
        file = open_file(filename)
        data = [2 * int(b < 255) - 1 for b in file.getdata()]
        file.close()
        yield Pattern(filename, data)


def _random_map(p: Pattern, k: int, f: Callable):
    idxs = list(range(len(p)))
    random.shuffle(idxs)

    p = p.copy()
    altered = 0

    for i in idxs:
        if altered == k:
            break

        fp = f(p[i])
        if fp != p[i]:
            p[i] = fp
            altered += 1

    return p


def flip_bytes(p: Pattern, k: int) -> Pattern:
    return _random_map(p, k, lambda x: -x)


def paint_bytes(p: Pattern, k: int, val: int) -> Pattern:
    return _random_map(p, k, lambda _: val)


def linear_comb(ps: list[Pattern]) -> Pattern:
    k = len(ps)

    if k & 1 == 0:
        raise ValueError(f"there must be an odd amount of patterns, given {k}")

    acc = ps[0].copy()
    n = len(acc)

    for mu in range(1, k):
        for i in range(n):
            acc[i] += ps[mu][i]

    for i in range(n):
        if acc[i] == 0:
            raise ValueError("found 0 in pattern at linear combination")

        acc[i] //= abs(acc[i])

    acc.name = f"linear_comb({', '.join(p.name for p in ps)})"
    return acc


def all_linear_combs(ps: list[Pattern]) -> Generator[Pattern]:
    n = len(ps)

    def bt(acc):
        i = len(acc)

        if i == n:
            yield linear_comb(acc)
            return

        acc.append(ps[i])
        yield from bt(acc)
        acc.pop()

        acc.append(-ps[i])
        yield from bt(acc)
        acc.pop()

    yield from bt([])
