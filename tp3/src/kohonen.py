import numpy as np
from typing import Callable, Optional, Self
from itertools import product
from scipy.ndimage import shift


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
        self._units_shape = shape
        self._coords = np.indices(shape)
        self._dist_buf = np.zeros(shape, dtype=float)
        self._dist_f = dist_f

        self._rng = np.random.default_rng(seed)
        self.R = self._rng.normal(0, 1, size=(*shape, n))

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
        self._dist_f(self._dist_buf, self._coords, bmu_idx)
        neigh = np.exp(-self._dist_buf / (2 * (sigma**2)))[..., np.newaxis]
        self.R += eta * neigh * (x - self.R)

    def component_arrays(self) -> tuple[np.ndarray, ...]:
        return tuple(self.R[..., i].copy() for i in range(self.R.shape[-1]))

    def u_matrix(self) -> np.ndarray:
        U = np.zeros(self._units_shape, dtype=float)
        counts = np.zeros(self._units_shape, dtype=int)

        n = len(self._units_shape)
        offsets = (offset for offset in product((-1, 0, 1), repeat=n) if any(x != 0 for x in offset))

        for offset in offsets:
            Rn = shift(self.R, shift=(*offset, 0), order=0, mode="constant", cval=np.nan)
            valid = ~np.isnan(Rn).any(axis=-1)

            dist = np.zeros(self._units_shape)
            diff = self.R - np.nan_to_num(Rn, nan=0.0)
            dist[valid] = np.linalg.norm(diff[valid], axis=-1)

            U += dist
            counts[valid] += 1

        return np.divide(U, counts, out=np.zeros_like(U), where=counts > 0)

    def train(
        self,
        X: np.ndarray,
        eta0: float = 0.1,
        etaf: float = 0.05,
        sigma0: Optional[float] = None,
        sigmaf: float = 0.05,
        max_iter: int = 10000,
        epoch_f: Callable[[Self], None] = lambda _: None,
        f_calls: int = 100,
    ):
        epoch_f(self)

        n = len(X)
        call_div = max_iter // f_calls

        if sigma0 is None or sigma0 <= 1:
            sigma0 = max(self._units_shape) / 2

        for t in range(max_iter):
            sigma = sigma0 * (sigmaf / sigma0) ** (t / max_iter)
            eta = eta0 * (etaf / eta0) ** (t / max_iter)

            x = X[self._rng.integers(0, n)]
            bmu_idx = self._find_bmu(x)
            self._update(x, bmu_idx, eta, sigma)

            if t % call_div == 0:
                epoch_f(self)
