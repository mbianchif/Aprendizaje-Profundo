from typing import Iterable
from PIL import Image
import random
import os


def retrieve_images(path: str) -> Iterable[list[int]]:
    images = (
        Image.open(f"{path}/{file_path}").convert("1") for file_path in os.listdir(path)
    )

    for image in images:
        yield [1 if p else -1 for p in image.getdata()]
        image.close()


def add_noise(p: list[int], n: int) -> list[int]:
    idxs = list(range(len(p)))
    random.shuffle(idxs)
    altered = p.copy()

    for i in range(n):
        altered[idxs[i]] = -altered[idxs[i]]

    return altered
