"""Design-space definition and space-filling sampling (LHS / Sobol / random)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import qmc


@dataclass
class Parameter:
    name: str
    low: float
    high: float
    nominal: Optional[float] = None
    log: bool = False
    unit: str = ""
    label: str = ""

    def __post_init__(self):
        if self.nominal is None:
            self.nominal = (np.sqrt(self.low * self.high) if self.log
                            else 0.5 * (self.low + self.high))
        if not self.label:
            self.label = self.name

    def to_unit(self, x):
        x = np.asarray(x, float)
        if self.log:
            return (np.log(x) - np.log(self.low)) / (np.log(self.high) - np.log(self.low))
        return (x - self.low) / (self.high - self.low)

    def from_unit(self, u):
        u = np.asarray(u, float)
        if self.log:
            return np.exp(np.log(self.low) + u * (np.log(self.high) - np.log(self.low)))
        return self.low + u * (self.high - self.low)


@dataclass
class DesignSpace:
    params: list = field(default_factory=list)

    @property
    def dim(self) -> int:
        return len(self.params)

    @property
    def names(self) -> list:
        return [p.name for p in self.params]

    @property
    def labels(self) -> list:
        return [p.label for p in self.params]

    def nominal(self) -> np.ndarray:
        return np.array([p.nominal for p in self.params])

    def to_unit(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return np.column_stack([p.to_unit(X[:, i]) for i, p in enumerate(self.params)])

    def from_unit(self, U) -> np.ndarray:
        U = np.atleast_2d(np.asarray(U, float))
        return np.column_stack([p.from_unit(U[:, i]) for i, p in enumerate(self.params)])

    def sample_unit(self, n: int, method: str = "lhs", seed: int = 0) -> np.ndarray:
        if method == "lhs":
            return qmc.LatinHypercube(d=self.dim, seed=seed).random(n)
        if method == "sobol":
            m = int(np.ceil(np.log2(max(n, 2))))
            return qmc.Sobol(d=self.dim, scramble=True, seed=seed).random_base2(m)[:n]
        return np.random.default_rng(seed).random((n, self.dim))

    def sample(self, n: int, method: str = "lhs", seed: int = 0) -> np.ndarray:
        return self.from_unit(self.sample_unit(n, method, seed))

    def as_dicts(self, X) -> list:
        X = np.atleast_2d(X)
        return [dict(zip(self.names, row)) for row in X]

    def subspace(self, keep) -> "DesignSpace":
        return DesignSpace([p for p in self.params if p.name in keep])

    def contains(self, X, tol: float = 1e-9) -> np.ndarray:
        U = self.to_unit(X)
        return np.all((U >= -tol) & (U <= 1 + tol), axis=1)
