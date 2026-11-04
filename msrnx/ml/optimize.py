"""AI-driven design optimization.

* ``surrogate_optimize``   global optimization (differential evolution) of an
                           objective built from surrogate predictions, with
                           penalty-handled constraints.
* ``bayesian_optimization`` constrained Bayesian optimization: Gaussian-process
                           models of objective and constraints, expected
                           improvement x probability of feasibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm

from .design_space import DesignSpace
from .surrogate import make_surrogate


def surrogate_optimize(predict: Callable[[np.ndarray], dict], space: DesignSpace,
                       objective: Callable[[dict], np.ndarray], constraints=(),
                       seed: int = 0, maxiter: int = 200, popsize: int = 25,
                       predict_std: Optional[Callable[[np.ndarray], dict]] = None,
                       kappa: float = 0.0) -> dict:
    """Minimize ``objective(pred)`` subject to constraints [(output, op, value)].

    ``predict_std(Z)`` (optional) returns {output: (mean, std)}.

    Constraints are enforced exactly by constrained differential evolution.
    With ``predict_std`` and ``kappa > 0`` they are *robust*: the surrogate's
    confidence bound (mean -/+ kappa*sigma) must satisfy the limit, so the
    model's own uncertainty protects feasibility.
    """
    from scipy.optimize import NonlinearConstraint

    cache = {}

    def _eval(u):
        """Memoized surrogate evaluation (objective and constraints share it)."""
        key = np.asarray(u, float).tobytes()
        if key not in cache:
            if len(cache) > 4096:
                cache.clear()
            Z = space.from_unit(np.atleast_2d(u))
            if predict_std is not None and kappa > 0:
                mean_std = predict_std(Z)            # returns {out: (mean, std)}
                cache[key] = ({k: v[0] for k, v in mean_std.items()},
                              {k: v[1] for k, v in mean_std.items()})
            else:
                cache[key] = (predict(Z), None)
        return cache[key]

    def P(u):
        return _eval(u)[0]

    def g_factory(out, op):
        def g(u):
            mean, std = _eval(u)
            v = float(np.asarray(mean[out]).ravel()[0])
            if std is not None:
                s = float(np.asarray(std[out]).ravel()[0])
                v = v - kappa * s if op == ">=" else v + kappa * s
            return v
        return g

    nlc = []
    for out, op, lim in constraints:
        g = g_factory(out, op)
        nlc.append(NonlinearConstraint(g, lim, np.inf) if op == ">=" else
                   NonlinearConstraint(g, -np.inf, lim))
    res = differential_evolution(lambda u: float(np.asarray(objective(P(u))).ravel()[0]),
                                 [(0.0, 1.0)] * space.dim, constraints=nlc, seed=seed,
                                 maxiter=maxiter, popsize=popsize, tol=1e-8, polish=False)
    x = space.from_unit(res.x[None, :])[0]
    return {"x": x, "params": dict(zip(space.names, x)), "fun": float(res.fun),
            "feasible": bool(getattr(res, "constr_violation", 0.0) <= 1e-9) if hasattr(res, "constr_violation") else True,
            "prediction": {k: float(np.asarray(v).ravel()[0]) for k, v in predict(x[None, :]).items()}}


@dataclass
class BOHistory:
    X: np.ndarray
    Y: dict
    best_feasible: list = field(default_factory=list)
    acquisition: list = field(default_factory=list)


def bayesian_optimization(evaluate: Callable[[np.ndarray], dict], space: DesignSpace,
                          objective: str, constraints=(), minimize: bool = True,
                          n_init: int = 20, n_iter: int = 25, seed: int = 0,
                          n_candidates: int = 4000, X0=None, Y0=None) -> BOHistory:
    """Constrained EI Bayesian optimization; ``evaluate(X) -> {output: array}``."""
    rng = np.random.default_rng(seed)
    if X0 is None:
        X = space.sample(n_init, "lhs", seed)
        Y = evaluate(X)
    else:
        X, Y = np.asarray(X0), {k: np.asarray(v) for k, v in Y0.items()}
    sign = 1.0 if minimize else -1.0
    hist = BOHistory(X, Y)

    def feasible(Yd, i):
        ok = True
        for out, op, lim in constraints:
            v = Yd[out][i]
            ok &= (v <= lim) if op == "<=" else (v >= lim)
        return bool(ok)

    def best_value(Yd):
        vals = [sign * Yd[objective][i] for i in range(len(Yd[objective])) if feasible(Yd, i)]
        return min(vals) if vals else np.inf

    hist.best_feasible.append(sign * best_value(Y))
    for it in range(n_iter):
        cand = space.sample(n_candidates, "random", seed + 1000 + it)
        mo = make_surrogate("gp", space, seed=seed).fit(X, sign * Y[objective])
        mu, sd = mo.predict(cand, return_std=True)
        fbest = best_value(Y)
        if not np.isfinite(fbest):
            fbest = np.min(sign * Y[objective])
        z = (fbest - mu) / np.maximum(sd, 1e-12)
        ei = (fbest - mu) * norm.cdf(z) + sd * norm.pdf(z)
        pf = np.ones(len(cand))
        for out, op, lim in constraints:
            mc = make_surrogate("gp", space, seed=seed).fit(X, Y[out])
            m, s = mc.predict(cand, return_std=True)
            zz = (lim - m) / np.maximum(s, 1e-12)
            pf *= norm.cdf(zz) if op == "<=" else norm.cdf(-zz)
        acq = ei * pf
        i = int(np.argmax(acq))
        xn = cand[i:i + 1]
        yn = evaluate(xn)
        X = np.vstack([X, xn])
        Y = {k: np.append(Y[k], yn[k]) for k in Y}
        hist.acquisition.append(float(acq[i]))
        hist.best_feasible.append(sign * best_value(Y))
    hist.X, hist.Y = X, Y
    return hist
