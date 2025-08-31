import PIL.Image as im
import numpy as np
import random
import os
from collections import defaultdict
from typing import Callable, Iterable
from numpy.random import Generator


def pad_to_size(
    img: im.Image,
    dim: tuple[int, int],
    rng: Generator,
) -> im.Image:
    res = im.frombytes(
        "L",
        dim,
        bytes(rng.choice((0, 255)) for _ in range(dim[0] * dim[1])),
    )

    x = (dim[0] - img.width) // 2
    y = (dim[1] - img.height) // 2
    res.paste(img, (x, y))
    return res


def retrieve_patterns(
    path: str,
    dim: tuple[int, int] | None = None,
    seed: int | None = None,
) -> Iterable[tuple[np.ndarray, tuple[int, int]]]:
    open_file = lambda x: im.open(f"{path}/{x}").convert("L")
    patterns = defaultdict(list)
    rng = np.random.default_rng(seed)
    shapes = {}

    for filename in os.listdir(path):
        with open_file(filename) as f:
            if dim:
                f = pad_to_size(f, dim, rng)

            data = np.array(f, dtype=int)
            data = 2 * (data > 127) - 1

            flattened = data.flatten()
            n = len(flattened)

            shapes[n] = data.shape
            patterns[n].append(flattened)

    for n, Ps in patterns.items():
        yield np.array(Ps, dtype=int), shapes[n]


def flip_bytes(p: np.ndarray, k: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idxs = list(range(len(p)))
    rng.shuffle(idxs)
    p = p.copy()
    altered = 0

    for i in idxs:
        if altered == k:
            break

        fp = -p[i]
        if fp != p[i]:
            p[i] = fp
            altered += 1

    return p


def paint_bytes(p: np.ndarray, k: int, val: int) -> np.ndarray:
    p = p.copy()
    p[:k] = val
    return p


def all_linear_combs(P: np.ndarray) -> np.ndarray:
    n = len(P)

    def bt(res, acc):
        if (i := len(acc)) == n:
            p = np.where(np.sum(acc, axis=0) >= 0, 1, -1)
            res.append(p)
            return res

        acc.append(P[i])
        bt(res, acc)
        acc.pop()

        acc.append(-P[i])
        bt(res, acc)
        acc.pop()

        return res

    return np.array(bt([], []))
