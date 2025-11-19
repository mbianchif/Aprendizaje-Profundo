import numpy as np


def euclidean_distance(
    dist: np.ndarray,
    coords: np.ndarray,
    bmu_idx: tuple[np.intp, ...],
):
    for i, c in enumerate(coords):
        dist += (c - bmu_idx[i]) ** 2


def chain_distance(
    dist: np.ndarray,
    coords: np.ndarray,
    bmu_idx: tuple[np.intp, ...],
):
    nodes = coords[0]
    bmu_i = bmu_idx[0]
    N = dist.size

    dist_direct = np.abs(nodes - bmu_i)
    dist_wrap = N - dist_direct

    dist[:] = np.minimum(dist_direct, dist_wrap) ** 2
