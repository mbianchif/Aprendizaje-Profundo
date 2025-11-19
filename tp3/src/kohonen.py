import numpy as np
from typing import Callable, Optional, Self


class Kohonen:
    def __init__(
        self,
        n: int,
        shape: tuple[int, ...],
        dist_f: Callable[
            [
                np.ndarray,
                np.ndarray,
                tuple[np.intp, ...],
            ],
        ],
        seed: Optional[int] = None,
    ):
        shape = (*shape, n)
        self._rng = np.random.default_rng(seed)
        self.R = self._rng.uniform(-1, 1, size=shape)
        self._dist_f = dist_f

    def _find_bmu(self, x: np.ndarray) -> tuple[np.intp, ...]:
        diff = self.R - x
        dist = np.sum(diff**2, axis=-1)
        return np.unravel_index(np.argmin(dist), dist.shape)

    def _update(
        self,
        x: np.ndarray,
        bmu_idx: tuple[np.intp, ...],
        eta: float,
        sigma: float,
    ):
        unit_dim = self.R.shape[:-1]
        coords = np.indices(unit_dim)
        dist = np.zeros(unit_dim)
        self._dist_f(dist, coords, bmu_idx)

        # dist2 = np.zeros(self.R.shape[:-1])
        # for i, c in enumerate(coords):
        #     dist2 += (c - bmu_idx[i]) ** 2

        v = np.exp(-dist / (2 * sigma**2))[..., np.newaxis]
        self.R += eta * v * (x - self.R)

    def component_arrays(self) -> tuple[np.ndarray, ...]:
        return tuple(self.R[..., i].copy() for i in range(self.R.shape[-1]))

    def train(
        self,
        X: np.ndarray,
        eta0: float = 0.1,
        max_iter: int = 10000,
        sigma0: Optional[float] = None,
        epoch_f: Callable[[Self], None] = lambda _: None,
    ):
        if sigma0 is None:
            sigma0 = max(self.R.shape[:-1]) / 2

        epoch_f(self)

        tau = max_iter / np.log(len(X))

        for t in range(max_iter):
            decay_factor = np.exp(-t / tau)
            eta = decay_factor * eta0
            sigma = decay_factor * sigma0

            for x in X:
                bmu_idx = self._find_bmu(x)
                self._update(x, bmu_idx, eta, sigma)

            epoch_f(self)
