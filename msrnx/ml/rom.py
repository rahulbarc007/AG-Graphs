"""Proper Orthogonal Decomposition (POD) reduced-order model of field data
(e.g. 2D power distributions): SVD compression + ML regression of the modal
coefficients from the design parameters."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .surrogate import make_surrogate


class PODModel:
    def __init__(self, energy: float = 0.9999, max_rank: int = 20, kind: str = "gp",
                 space=None, seed: int = 0):
        self.energy, self.max_rank, self.kind, self.space, self.seed = energy, max_rank, kind, space, seed

    def fit(self, X, snapshots) -> "PODModel":
        S = np.asarray(snapshots, float)            # (n_samples, n_dof)
        self.mean = S.mean(axis=0)
        U, s, Vt = np.linalg.svd(S - self.mean, full_matrices=False)
        self.singular_values = s
        cum = np.cumsum(s ** 2) / np.sum(s ** 2)
        self.cumulative_energy = cum
        self.rank = int(min(np.searchsorted(cum, self.energy) + 1, self.max_rank, len(s)))
        self.modes = Vt[: self.rank]                # (r, n_dof)
        coeff = (S - self.mean) @ self.modes.T      # (n, r)
        self.models = [make_surrogate(self.kind, self.space, seed=self.seed).fit(X, coeff[:, j])
                       for j in range(self.rank)]
        return self

    def project(self, snapshots) -> np.ndarray:
        S = np.atleast_2d(snapshots)
        return (S - self.mean) @ self.modes.T @ self.modes + self.mean

    def predict(self, X) -> np.ndarray:
        C = np.column_stack([m.predict(X) for m in self.models])
        return C @ self.modes + self.mean

    def compression_ratio(self, n_dof: Optional[int] = None) -> float:
        n_dof = n_dof or self.modes.shape[1]
        return n_dof / self.rank
