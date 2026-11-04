"""Accuracy control for ML surrogates.

* ``ActiveLearner``     uncertainty-driven sequential design: new physics runs
                        are placed where the surrogate is least certain until a
                        target validation accuracy is met.
* ``GuardedSurrogate``  hybrid physics/ML predictor: the surrogate answers only
                        when its predictive uncertainty is below a tolerance and
                        the query lies inside the trained domain; otherwise the
                        physics solver is run and the surrogate is retrained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .design_space import DesignSpace
from .surrogate import make_surrogate, regression_metrics


@dataclass
class ALHistory:
    n_samples: list = field(default_factory=list)
    rmse: list = field(default_factory=list)
    max_abs: list = field(default_factory=list)
    mean_std: list = field(default_factory=list)
    strategy: str = "max_std"


class ActiveLearner:
    """Batch active learning on a precomputed candidate pool or a live oracle.

    ``oracle(X) -> y`` evaluates the physics model for an (n, d) array.
    """

    def __init__(self, oracle: Callable, space: DesignSpace, kind: str = "gp",
                 n_init: int = 20, batch: int = 10, n_candidates: int = 3000, seed: int = 0,
                 strategy: str = "imse", scale: float = 1.0):
        self.oracle, self.space, self.kind = oracle, space, kind
        self.n_init, self.batch, self.seed = n_init, batch, seed
        self.strategy, self.scale = strategy, scale
        self.cand = space.sample(n_candidates, "sobol", seed + 101)

    def _select(self, model, X_have):
        if self.strategy == "random":
            rng = np.random.default_rng(self.seed + len(X_have))
            return self.cand[rng.choice(len(self.cand), self.batch, replace=False)]
        if self.strategy == "imse" and hasattr(model, "gp"):
            return self._select_imse(model, X_have)
        _, s = model.predict(self.cand, return_std=True)
        # greedy batch: pick max std, then penalize neighbours (distance in unit cube)
        U = self.space.to_unit(self.cand)
        chosen = []
        score = s.copy()
        for _ in range(self.batch):
            i = int(np.argmax(score))
            chosen.append(i)
            d = np.linalg.norm(U - U[i], axis=1)
            score = score * (1.0 - np.exp(-(d / 0.15) ** 2))
        return self.cand[chosen]

    def _select_imse(self, model, X_have, n_cand: int = 1200, n_ref: int = 600):
        """Greedy integrated-variance reduction using the GP posterior covariance.

        Score(c) = sum_r cov(r, c)^2 / var(c): the total predictive-variance
        reduction over a reference set if c were observed.  After each pick
        the posterior covariance is updated with the rank-1 conditioning
        formula, so a batch spreads out naturally.
        """
        rng = np.random.default_rng(self.seed + len(X_have))
        ci = rng.choice(len(self.cand), min(n_cand, len(self.cand)), replace=False)
        C = self.cand[ci]
        R = self.space.sample(n_ref, "lhs", self.seed + 7 + len(X_have))
        _, cov = model.gp.predict(self.space.to_unit(np.vstack([C, R])), return_cov=True)
        nc = len(C)
        chosen = []
        for _ in range(self.batch):
            var_c = np.maximum(np.diag(cov)[:nc], 1e-300)
            cross = cov[nc:, :nc]                       # (n_ref, n_cand)
            score = (cross ** 2).sum(axis=0) / var_c
            score[chosen] = -np.inf
            j = int(np.argmax(score))
            chosen.append(j)
            col = cov[:, j].copy()
            cov = cov - np.outer(col, col) / max(col[j], 1e-300)
        return C[chosen]

    def run(self, X_test, y_test, max_samples: int = 150, target_rmse: Optional[float] = None,
            X0=None, y0=None) -> tuple:
        X = self.space.sample(self.n_init, "lhs", self.seed) if X0 is None else np.asarray(X0)
        y = self.oracle(X) if y0 is None else np.asarray(y0)
        hist = ALHistory(strategy=self.strategy)
        while True:
            model = make_surrogate(self.kind, self.space, seed=self.seed).fit(X, y)
            pred, std = model.predict(X_test, return_std=True)
            m = regression_metrics(y_test, pred, self.scale)
            hist.n_samples.append(len(X))
            hist.rmse.append(m["rmse"])
            hist.max_abs.append(m["max_abs"])
            hist.mean_std.append(float(np.mean(std)) * self.scale)
            if (target_rmse is not None and m["rmse"] <= target_rmse) or len(X) >= max_samples:
                break
            Xn = self._select(model, X)
            yn = self.oracle(Xn)
            X = np.vstack([X, Xn])
            y = np.concatenate([y, yn])
        return model, X, y, hist


class GuardedSurrogate:
    """Physics-guarded ML predictor (fallback to physics when uncertain).

    With ``lf`` (a cheap low-fidelity model ``lf(X) -> y``) the internal model
    is a multi-fidelity discrepancy surrogate, which is far more accurate for
    the same number of expensive physics runs.
    """

    def __init__(self, physics: Callable, space: DesignSpace, kind: str = "gp",
                 std_tol: float = 1e-3, X0=None, y0=None, refit_every: int = 1, seed: int = 0,
                 lf: Optional[Callable] = None):
        self.physics, self.space, self.kind = physics, space, kind
        self.std_tol, self.refit_every, self.seed = std_tol, refit_every, seed
        self.lf = lf
        self.X = np.empty((0, space.dim)) if X0 is None else np.asarray(X0, float)
        self.y = np.empty(0) if y0 is None else np.asarray(y0, float)
        self.ylf = np.asarray(lf(self.X)) if (lf is not None and len(self.X)) else np.empty(0)
        self.n_ml = 0
        self.n_physics = 0
        self.log = []
        self._pending = 0
        self.model = None
        if len(self.y) >= 3:
            self._refit()

    def _refit(self):
        if self.lf is not None:
            from .multifidelity import MultiFidelitySurrogate
            self.model = MultiFidelitySurrogate(self.space, self.kind, self.seed).fit(
                self.X, self.ylf, self.y)
        else:
            self.model = make_surrogate(self.kind, self.space, seed=self.seed).fit(self.X, self.y)
        self._pending = 0

    def _model_predict(self, x):
        if self.lf is not None:
            ylf = np.asarray(self.lf(x), float)
            m, s = self.model.predict(x, ylf, return_std=True)
            return m, s, ylf
        m, s = self.model.predict(x, return_std=True)
        return m, s, None

    def predict_one(self, x) -> tuple:
        x = np.asarray(x, float)[None, :]
        inside = bool(self.space.contains(x)[0])
        ylf = None
        if self.model is not None and inside:
            m, s, ylf = self._model_predict(x)
            if s[0] <= self.std_tol:
                self.n_ml += 1
                self.log.append(("ml", float(m[0]), float(s[0])))
                return float(m[0]), float(s[0]), "ml"
        y = float(np.asarray(self.physics(x)).ravel()[0])
        self.n_physics += 1
        self.X = np.vstack([self.X, x])
        self.y = np.append(self.y, y)
        if self.lf is not None:
            if ylf is None:
                ylf = np.asarray(self.lf(x), float)
            self.ylf = np.append(self.ylf, ylf)
        self._pending += 1
        if self._pending >= self.refit_every and len(self.y) >= 3:
            self._refit()
        self.log.append(("physics", y, 0.0))
        return y, 0.0, "physics"

    def predict(self, X) -> tuple:
        out = [self.predict_one(x) for x in np.atleast_2d(X)]
        return (np.array([o[0] for o in out]), np.array([o[1] for o in out]),
                [o[2] for o in out])

    @property
    def ml_fraction(self) -> float:
        tot = self.n_ml + self.n_physics
        return self.n_ml / tot if tot else 0.0
