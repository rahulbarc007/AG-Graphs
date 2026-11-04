"""Machine-learning surrogate models with predictive uncertainty.

* ``GPSurrogate``          Gaussian process, anisotropic (ARD) Matern-5/2 kernel
* ``MLPEnsembleSurrogate`` deep ensemble of bootstrapped neural networks
* ``GBRSurrogate``         bagged histogram gradient boosting
* ``PolySurrogate``        polynomial ridge regression (response surface)
* ``AutoSurrogate``        k-fold cross-validated selection among the above

All accept physical inputs; when a :class:`DesignSpace` is supplied inputs
are mapped to the unit hypercube (log-scaled parameters handled).
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures

from .design_space import DesignSpace


class Surrogate:
    name = "base"

    def __init__(self, space: Optional[DesignSpace] = None, seed: int = 0):
        self.space = space
        self.seed = seed
        self._lo = self._hi = None

    def _u(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if self.space is not None:
            return self.space.to_unit(X)
        if self._lo is None:
            self._lo, self._hi = X.min(axis=0), X.max(axis=0)
        return (X - self._lo) / np.where(self._hi > self._lo, self._hi - self._lo, 1.0)

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X, return_std: bool = False):
        raise NotImplementedError

    def clone(self) -> "Surrogate":
        return type(self)(**self._init_kw())

    def _init_kw(self):
        return {"space": self.space, "seed": self.seed}


class GPSurrogate(Surrogate):
    name = "GP (ARD Matern)"

    def __init__(self, space=None, seed: int = 0, nu: float = 2.5, n_restarts: int = 3):
        super().__init__(space, seed)
        self.nu, self.n_restarts = nu, n_restarts

    def _init_kw(self):
        return {**super()._init_kw(), "nu": self.nu, "n_restarts": self.n_restarts}

    def fit(self, X, y):
        self._lo = None
        U = self._u(X)
        y = np.asarray(y, float)
        d = U.shape[1]
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.full(d, 0.5), length_scale_bounds=(1e-2, 1e3), nu=self.nu)
                  + WhiteKernel(1e-6, (1e-12, 1e-1)))
        self.gp = GaussianProcessRegressor(kernel, normalize_y=True,
                                           n_restarts_optimizer=self.n_restarts,
                                           random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            self.gp.fit(U, y)
        return self

    def predict(self, X, return_std: bool = False):
        U = self._u(X)
        if return_std:
            m, s = self.gp.predict(U, return_std=True)
            return m, s
        return self.gp.predict(U)

    @property
    def length_scales(self) -> np.ndarray:
        return np.asarray(self.gp.kernel_.k1.k2.length_scale)


class MLPEnsembleSurrogate(Surrogate):
    name = "Deep ensemble (MLP)"

    def __init__(self, space=None, seed: int = 0, n_members: int = 6, hidden=(48, 48),
                 alpha: float = 1e-5, max_iter: int = 3000):
        super().__init__(space, seed)
        self.n_members, self.hidden, self.alpha, self.max_iter = n_members, hidden, alpha, max_iter

    def _init_kw(self):
        return {**super()._init_kw(), "n_members": self.n_members, "hidden": self.hidden,
                "alpha": self.alpha, "max_iter": self.max_iter}

    def fit(self, X, y):
        self._lo = None
        U = self._u(X)
        y = np.asarray(y, float)
        self.mu, self.sd = y.mean(), y.std() or 1.0
        z = (y - self.mu) / self.sd
        rng = np.random.default_rng(self.seed)
        self.members = []
        for m in range(self.n_members):
            idx = rng.integers(0, len(z), len(z)) if m > 0 else np.arange(len(z))
            net = MLPRegressor(hidden_layer_sizes=self.hidden, activation="tanh", solver="lbfgs",
                               alpha=self.alpha,
                               max_iter=self.max_iter, random_state=self.seed + m)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                net.fit(U[idx], z[idx])
            self.members.append(net)
        return self

    def predict(self, X, return_std: bool = False):
        U = self._u(X)
        P = np.array([m.predict(U) for m in self.members]) * self.sd + self.mu
        if return_std:
            return P.mean(axis=0), P.std(axis=0) + 1e-12
        return P.mean(axis=0)


class GBRSurrogate(Surrogate):
    name = "Gradient boosting (bagged)"

    def __init__(self, space=None, seed: int = 0, n_members: int = 8, max_iter: int = 400):
        super().__init__(space, seed)
        self.n_members, self.max_iter = n_members, max_iter

    def _init_kw(self):
        return {**super()._init_kw(), "n_members": self.n_members, "max_iter": self.max_iter}

    def fit(self, X, y):
        self._lo = None
        U = self._u(X)
        y = np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        self.members = []
        for m in range(self.n_members):
            idx = rng.integers(0, len(y), len(y))
            g = HistGradientBoostingRegressor(max_iter=self.max_iter, learning_rate=0.05,
                                              max_leaf_nodes=15, min_samples_leaf=3,
                                              random_state=self.seed + m)
            g.fit(U[idx], y[idx])
            self.members.append(g)
        return self

    def predict(self, X, return_std: bool = False):
        U = self._u(X)
        P = np.array([m.predict(U) for m in self.members])
        if return_std:
            return P.mean(axis=0), P.std(axis=0) + 1e-12
        return P.mean(axis=0)


class PolySurrogate(Surrogate):
    name = "Polynomial ridge"

    def __init__(self, space=None, seed: int = 0, degree: int = 3):
        super().__init__(space, seed)
        self.degree = degree

    def _init_kw(self):
        return {**super()._init_kw(), "degree": self.degree}

    def fit(self, X, y):
        self._lo = None
        U = self._u(X)
        self.pf = PolynomialFeatures(self.degree, include_bias=True)
        F = self.pf.fit_transform(U)
        self.model = RidgeCV(alphas=np.logspace(-10, 1, 40)).fit(F, y)
        res = np.asarray(y) - self.model.predict(F)
        dof = max(len(y) - F.shape[1], 1)
        self.sigma = float(np.sqrt((res ** 2).sum() / dof))
        return self

    def predict(self, X, return_std: bool = False):
        F = self.pf.transform(self._u(X))
        m = self.model.predict(F)
        if return_std:
            return m, np.full_like(m, self.sigma)
        return m


KINDS = {"gp": GPSurrogate, "mlp": MLPEnsembleSurrogate, "gbr": GBRSurrogate, "poly": PolySurrogate}


def make_surrogate(kind: str = "gp", space: Optional[DesignSpace] = None, **kw) -> Surrogate:
    if kind == "auto":
        return AutoSurrogate(space, **kw)
    return KINDS[kind](space=space, **kw)


def regression_metrics(y_true, y_pred, scale: float = 1.0) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    e = (y_pred - y_true) * scale
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum()) or 1e-30
    return {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
            "max_abs": float(np.abs(e).max()), "bias": float(e.mean()), "r2": 1.0 - ss_res / ss_tot}


def cross_validate(kind, X, y, k: int = 5, seed: int = 0, space=None, scale: float = 1.0,
                   **kw) -> dict:
    """k-fold cross-validation; returns metrics and out-of-fold predictions."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    pred = np.zeros_like(y)
    std = np.zeros_like(y)
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
        m = make_surrogate(kind, space, seed=seed, **kw) if isinstance(kind, str) else kind.clone()
        m.fit(X[tr], y[tr])
        pred[te], std[te] = m.predict(X[te], return_std=True)
    met = regression_metrics(y, pred, scale)
    z = (pred - y) / np.maximum(std, 1e-300)
    met["coverage_95"] = float(np.mean(np.abs(z) < 1.96))
    return {"metrics": met, "y_pred": pred, "y_std": std}


class AutoSurrogate(Surrogate):
    """Selects the best surrogate family by k-fold CV RMSE."""
    name = "Auto (CV-selected)"

    def __init__(self, space=None, seed: int = 0, candidates=("gp", "mlp", "gbr", "poly"),
                 k: int = 5):
        super().__init__(space, seed)
        self.candidates, self.k = candidates, k

    def _init_kw(self):
        return {**super()._init_kw(), "candidates": self.candidates, "k": self.k}

    def fit(self, X, y):
        self.cv = {}
        for c in self.candidates:
            self.cv[c] = cross_validate(c, X, y, self.k, self.seed, self.space)["metrics"]
        self.best = min(self.cv, key=lambda c: self.cv[c]["rmse"])
        self.model = make_surrogate(self.best, self.space, seed=self.seed).fit(X, y)
        self.name = f"Auto -> {self.model.name}"
        return self

    def predict(self, X, return_std: bool = False):
        return self.model.predict(X, return_std)


class MultiOutputSurrogate:
    """Independent surrogate per output quantity."""

    def __init__(self, kind: str = "gp", space=None, seed: int = 0, **kw):
        self.kind, self.space, self.seed, self.kw = kind, space, seed, kw
        self.models = {}

    def fit(self, X, Y: dict):
        for name, y in Y.items():
            ok = np.isfinite(y)
            self.models[name] = make_surrogate(self.kind, self.space, seed=self.seed, **self.kw).fit(
                np.asarray(X)[ok], np.asarray(y)[ok])
        return self

    def predict(self, X, return_std: bool = False) -> dict:
        return {n: m.predict(X, return_std) for n, m in self.models.items()}
