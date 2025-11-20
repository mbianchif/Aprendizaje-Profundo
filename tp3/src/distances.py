import numpy as np


def euclidean_distance(
    dist: np.ndarray,
    coords: np.ndarray,
    bmu_idx: tuple[np.intp, ...],
):
    d = coords - np.array(bmu_idx)[:, None, None]
    dist[:] = np.sum(d * d, axis=0)


def chain_distance(
    dist: np.ndarray,
    coords: np.ndarray,
    bmu_idx: tuple[np.intp, ...],
):
    nodes = coords[0]
    bmu_i = int(bmu_idx[0])
    N = nodes.shape[0]

    dist_direct = np.abs(nodes - bmu_i)
    dist_wrap = N - dist_direct
    dist[:] = np.minimum(dist_direct, dist_wrap) ** 2
