from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def corr_bootstrap(X: np.ndarray, bootstraps: int = 1000) -> np.ndarray:
    """Calculate the bootstrapping correlation matrix."""
    size = X.shape[1]
    num_positive_definite, delta_bootstrap = 0, np.zeros((size, size))
    for _ in prange(bootstraps):
        delta = np.eye(size)
        for i in range(size):
            for j in range(i + 1, size):
                A = X[:, i][~np.isnan(X[:, i]) & ~np.isnan(X[:, j])]
                B = X[:, j][~np.isnan(X[:, i]) & ~np.isnan(X[:, j])]
                idx = np.random.choice(np.arange(0, A.size), size=A.size)
                A, B = A[idx], B[idx]
                delta[i, j] = delta[j, i] = (A * B).sum() / (np.std(A) * np.std(B) * A.size)
        if np.all(np.linalg.eigvals(delta) > 0):
            delta_bootstrap += delta
            num_positive_definite += 1
    return delta_bootstrap / num_positive_definite
