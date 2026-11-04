"""Global sensitivity analysis and parameter (dimension) reduction.

* Sobol first-order and total indices (Saltelli 2010 / Jansen estimators,
  Sobol' quasi-random sampling, bootstrap confidence intervals).
* Morris elementary-effects screening (mu*, sigma).
* Active subspaces: eigen-decomposition of C = E[grad f grad f^T].
* ``reduce_parameters``: keep the parameters carrying most of the variance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.stats import qmc

from .design_space import DesignSpace


@dataclass
class SobolResult:
    names: list
    S1: np.ndarray
    ST: np.ndarray
    S1_conf: np.ndarray
    ST_conf: np.ndarray
    variance: float
    n_evals: int

    def ranking(self) -> list:
        order = np.argsort(-self.ST)
        return [(self.names[i], float(self.ST[i]), float(self.S1[i])) for i in order]


def sobol_indices(f: Callable, space: DesignSpace, n: int = 4096, seed: int = 0,
                  n_boot: int = 200) -> SobolResult:
    """Saltelli/Jansen estimators; ``f`` maps an (N, d) physical array to (N,)."""
    d = space.dim
    m = int(np.ceil(np.log2(n)))
    base = qmc.Sobol(2 * d, scramble=True, seed=seed).random_base2(m)
    A, B = base[:, :d], base[:, d:]
    fA = np.asarray(f(space.from_unit(A)), float)
    fB = np.asarray(f(space.from_unit(B)), float)
    fAB = np.empty((d, len(A)))
    for i in range(d):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        fAB[i] = f(space.from_unit(ABi))
    var = np.var(np.concatenate([fA, fB]))

    def est(idx):
        a, b, ab = fA[idx], fB[idx], fAB[:, idx]
        v = np.var(np.concatenate([a, b]))
        S1 = np.mean(b[None, :] * (ab - a[None, :]), axis=1) / v
        ST = 0.5 * np.mean((a[None, :] - ab) ** 2, axis=1) / v
        return S1, ST

    S1, ST = est(np.arange(len(A)))
    rng = np.random.default_rng(seed)
    bs1, bst = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(A), len(A))
        s1, st = est(idx)
        bs1.append(s1)
        bst.append(st)
    S1c = 1.96 * np.std(bs1, axis=0)
    STc = 1.96 * np.std(bst, axis=0)
    return SobolResult(space.names, S1, ST, S1c, STc, float(var), len(A) * (d + 2))


@dataclass
class MorrisResult:
    names: list
    mu: np.ndarray
    mu_star: np.ndarray
    sigma: np.ndarray
    effects: np.ndarray = field(repr=False, default=None)


def morris_screening(f: Callable, space: DesignSpace, r: int = 40, levels: int = 4,
                     seed: int = 0) -> MorrisResult:
    """Morris one-at-a-time trajectories on a ``levels``-grid in the unit cube."""
    d = space.dim
    rng = np.random.default_rng(seed)
    delta = levels / (2.0 * (levels - 1))
    grid = np.arange(levels) / (levels - 1)
    EE = np.zeros((r, d))
    for t in range(r):
        x = rng.choice(grid[grid <= 1 - delta + 1e-12], d)
        order = rng.permutation(d)
        pts = [x.copy()]
        for i in order:
            x = x.copy()
            x[i] = x[i] + delta
            pts.append(x)
        Y = np.asarray(f(space.from_unit(np.array(pts))), float)
        for k, i in enumerate(order):
            EE[t, i] = (Y[k + 1] - Y[k]) / delta
    return MorrisResult(space.names, EE.mean(axis=0), np.abs(EE).mean(axis=0),
                        EE.std(axis=0), EE)


@dataclass
class ActiveSubspaceResult:
    names: list
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray        # columns = directions (unit-cube coordinates)
    activity_scores: np.ndarray
    samples_unit: np.ndarray
    values: np.ndarray

    def reduced_coordinates(self, k: int = 1) -> np.ndarray:
        return (2 * self.samples_unit - 1) @ self.eigenvectors[:, :k]


def active_subspace(f: Callable, space: DesignSpace, n: int = 400, h: float = 1e-3,
                    seed: int = 0, k_scores: int = 2) -> ActiveSubspaceResult:
    """Finite-difference gradients (unit-cube coordinates mapped to [-1, 1])."""
    d = space.dim
    U = space.sample_unit(n, "sobol", seed)
    U = np.clip(U, h, 1 - h)
    f0 = np.asarray(f(space.from_unit(U)), float)
    G = np.zeros((n, d))
    for i in range(d):
        Up = U.copy()
        Up[:, i] += h
        Um = U.copy()
        Um[:, i] -= h
        G[:, i] = (np.asarray(f(space.from_unit(Up))) - np.asarray(f(space.from_unit(Um)))) / (2 * h)
    G *= 0.5                                       # d/dx on [-1,1] = 0.5 d/du
    C = G.T @ G / n
    w, W = np.linalg.eigh(C)
    order = np.argsort(-w)
    w, W = w[order], W[:, order]
    for j in range(d):                              # sign convention
        if W[np.argmax(np.abs(W[:, j])), j] < 0:
            W[:, j] *= -1
    scores = (W[:, :k_scores] ** 2 * w[:k_scores][None, :]).sum(axis=1)
    return ActiveSubspaceResult(space.names, w, W, scores, U, f0)


def accuracy_threshold(target_rmse: float, output_variance: float) -> float:
    """Total-index threshold below which dropping a parameter costs less than
    ``target_rmse``: a parameter carries ~sqrt(S_T * Var(y)) of output spread."""
    return (target_rmse ** 2) / max(output_variance, 1e-300)


def reduce_parameters(sobol: SobolResult, threshold: float = 0.01,
                      target_rmse: float = None) -> tuple:
    """Split parameters into (influential, negligible) by total Sobol index.

    With ``target_rmse`` the threshold is set from the accuracy target
    (see :func:`accuracy_threshold`) instead of a fixed variance share.
    """
    if target_rmse is not None:
        threshold = accuracy_threshold(target_rmse, sobol.variance)
    keep = [n for n, st in zip(sobol.names, sobol.ST) if st >= threshold]
    drop = [n for n, st in zip(sobol.names, sobol.ST) if st < threshold]
    return keep, drop


def ishigami(X, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    """Ishigami test function on [-pi, pi]^3 (analytic Sobol indices known)."""
    X = np.atleast_2d(X)
    return np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])


def ishigami_analytic(a: float = 7.0, b: float = 0.1) -> dict:
    V1 = 0.5 * (1 + b * np.pi ** 4 / 5) ** 2
    V2 = a ** 2 / 8
    V13 = b ** 2 * np.pi ** 8 * (1 / 18 - 1 / 50)
    V = V1 + V2 + V13
    return {"S1": np.array([V1 / V, V2 / V, 0.0]),
            "ST": np.array([(V1 + V13) / V, V2 / V, V13 / V])}
