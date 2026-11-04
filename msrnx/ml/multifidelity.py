"""Multi-fidelity learning: an ML model of the discrepancy between a cheap
low-fidelity physics model and the high-fidelity solver.

    y_HF(x) ~ y_LF(x) + delta(x, y_LF(x))

The low-fidelity value enters both as the baseline and as an input feature,
so few expensive runs suffice to recover high-fidelity accuracy at
low-fidelity cost.
"""
from __future__ import annotations

import numpy as np

from .design_space import DesignSpace
from .surrogate import make_surrogate


class MultiFidelitySurrogate:
    def __init__(self, space: DesignSpace, kind: str = "gp", seed: int = 0,
                 use_lf_feature: bool = True):
        self.space, self.kind, self.seed, self.use_lf = space, kind, seed, use_lf_feature

    def _features(self, X, y_lf):
        U = self.space.to_unit(X)
        if self.use_lf:
            z = (np.asarray(y_lf) - self._lf_mu) / self._lf_sd
            return np.column_stack([U, z])
        return U

    def fit(self, X, y_lf, y_hf):
        y_lf = np.asarray(y_lf, float)
        self._lf_mu, self._lf_sd = y_lf.mean(), y_lf.std() or 1.0
        F = self._features(X, y_lf)
        self.model = make_surrogate(self.kind, None, seed=self.seed)
        self.model.fit(F, np.asarray(y_hf, float) - y_lf)
        return self

    def predict(self, X, y_lf, return_std: bool = False):
        F = self._features(X, y_lf)
        if return_std:
            d, s = self.model.predict(F, return_std=True)
            return np.asarray(y_lf) + d, s
        return np.asarray(y_lf) + self.model.predict(F)
