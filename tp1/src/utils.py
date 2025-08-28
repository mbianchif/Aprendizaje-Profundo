from collections import defaultdict
import matplotlib.image as mpimg
import PIL.Image as im
import numpy as np
import random
import os
from typing import Callable, Generator, Iterable


def pad_to_size(img: im.Image, dim: tuple[int, int]) -> im.Image:
    res = im.frombytes(
        "L",
        dim,
        bytes(random.choice((0, 255)) for _ in range(dim[0] * dim[1])),
    )

    x = (dim[0] - img.width) // 2
    y = (dim[1] - img.height) // 2
    res.paste(img, (x, y))
    return res


def retrieve_patterns(
    path: str,
    dim: tuple[int, int] | None = None,
) -> Iterable[np.ndarray]:
    open_file = lambda x: im.open(f"{path}/{x}").convert("L")
    patterns = defaultdict(list)

    for filename in os.listdir(path):
        with open_file(filename) as f:
            if dim:
                f = pad_to_size(f, dim)

            data = np.array(f, dtype=int).flatten()
            data = 2 * (data > 127) - 1

            patterns[len(data)].append(data)

    for ps in patterns.values():
        yield np.array(ps, dtype=int)


def retrieve_images(path: str) -> Iterable[np.ndarray]:
    open_file = lambda x: mpimg.imread(f"{path}/{x}")

    for filename in os.listdir(path):
        yield open_file(filename)


def _random_map(p: np.ndarray, k: int, f: Callable):
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


def flip_bytes(p: np.ndarray, k: int) -> np.ndarray:
    return _random_map(p, k, lambda x: -x)


def paint_bytes(p: np.ndarray, k: int, val: int) -> np.ndarray:
    return _random_map(p, k, lambda _: val)


def all_linear_combs(P: np.ndarray) -> Generator[np.ndarray]:
    n = len(P)

    def bt(acc):
        if (i := len(acc)) == n:
            yield np.sum(acc, axis=0)
            return

        acc.append(P[i])
        yield from bt(acc)
        acc.pop()

        acc.append(-P[i])
        yield from bt(acc)
        acc.pop()

    yield from bt([])
