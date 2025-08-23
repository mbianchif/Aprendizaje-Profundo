import matplotlib.image as mpimg
import PIL.Image as im
import random
import os
from src.pattern import Pattern
from typing import Callable, Generator, Iterable


def pad_to_size(img: im.Image, dim: tuple[int, int]) -> im.Image:
    res = im.new("L", dim, 0)
    x = (dim[0] - img.width) // 2
    y = (dim[1] - img.height) // 2
    res.paste(img, (x, y))
    return res


def retrieve_patterns(
    path: str,
    dim: tuple[int, int] | None = None,
) -> Iterable[Pattern]:
    open_file = lambda x: im.open(f"{path}/{x}").convert("L")

    for filename in os.listdir(path):
        file = open_file(filename)
        if dim:
            tmp = file
            file = pad_to_size(file, dim)
            tmp.close()

        data = [2 * int(b > 127) - 1 for b in file.getdata()]
        if not dim:
            file.close()

        yield Pattern(filename, data)


def retrieve_images(path: str) -> Iterable:
    open_file = lambda x: mpimg.imread(f"{path}/{x}")

    for filename in os.listdir(path):
        yield open_file(filename)


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

    for i in range(n):
        acc[i] = sum(ps[mu][i] for mu in range(1, k))
        acc[i] //= abs(acc[i])

    acc.name = f"linear_comb({', '.join(p.name for p in ps)})"
    return acc


def all_linear_combs(ps: list[Pattern]) -> Generator[Pattern]:
    n = len(ps)

    def bt(acc):
        if (i := len(acc)) == n:
            yield linear_comb(acc)
            return

        acc.append(ps[i])
        yield from bt(acc)
        acc.pop()

        acc.append(-ps[i])
        yield from bt(acc)
        acc.pop()

    yield from bt([])
