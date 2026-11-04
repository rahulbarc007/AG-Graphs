"""Uncertainty quantification.

* Nuclear-data sensitivity coefficients S = (dk/k)/(dsigma/sigma) by direct
  perturbation of the multigroup library, and first-order ("sandwich")
  propagation of relative cross-section uncertainties.
* Monte Carlo propagation of input uncertainties through any (surrogate or
  physics) model, with standardized regression coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

# Illustrative 1-sigma relative uncertainties of one-group-relevant data
# (order of magnitude of modern covariance libraries; NOT evaluated data).
DEFAULT_ND_UNCERTAINTY = {
    ("U233", "nu"): 0.0030, ("U233", "fission"): 0.0060, ("U233", "capture"): 0.030,
    ("U235", "nu"): 0.0020, ("U235", "fission"): 0.0050, ("U235", "capture"): 0.020,
    ("Th232", "capture"): 0.020, ("Th232", "elastic"): 0.020, ("U238", "capture"): 0.015,
    ("Pa233", "capture"): 0.080, ("Li7", "capture"): 0.050, ("Li6", "nalpha"): 0.005,
    ("Be9", "n2n"): 0.080, ("Be9", "capture"): 0.100, ("F19", "capture"): 0.080,
    ("F19", "elastic"): 0.020, ("F19", "inelastic"): 0.100, ("C12", "capture"): 0.050,
    ("C12", "elastic"): 0.005,
}


@dataclass
class SensitivityResult:
    keys: list
    S: np.ndarray                 # (dk/k)/(dsigma/sigma)
    k_ref: float
    rel_unc: np.ndarray
    contribution_pcm: np.ndarray  # |S| * u * 1e5
    total_pcm: float


def nuclear_data_sensitivity(k_fn: Callable[[Optional[dict]], float], keys=None,
                             rel_step: float = 0.02, uncertainties: Optional[dict] = None
                             ) -> SensitivityResult:
    """Central-difference sensitivities of k to (nuclide, reaction) multipliers.

    ``k_fn(perturb)`` returns k_eff for a perturbation dict
    {(nuclide, reaction): factor}.
    """
    unc = uncertainties or DEFAULT_ND_UNCERTAINTY
    keys = list(keys or unc.keys())
    k0 = k_fn(None)
    S = []
    for key in keys:
        kp = k_fn({key: 1.0 + rel_step})
        km = k_fn({key: 1.0 - rel_step})
        S.append((kp - km) / (2 * rel_step * k0))
    S = np.array(S)
    u = np.array([unc.get(k, 0.0) for k in keys])
    contrib = np.abs(S) * u * 1e5
    # uncorrelated sandwich rule
    total = float(np.sqrt(np.sum((S * u) ** 2)) * 1e5)
    return SensitivityResult(keys, S, k0, u, contrib, total)


def monte_carlo(f: Callable, dists: dict, n: int = 5000, seed: int = 0) -> dict:
    """Propagate independent input distributions through ``f``.

    ``dists``: name -> ('normal', mean, sd) | ('uniform', lo, hi) | ('lognormal', mu, sigma).
    ``f`` accepts a dict of arrays and returns an array.
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for name, spec in dists.items():
        kind = spec[0]
        if kind == "normal":
            samples[name] = rng.normal(spec[1], spec[2], n)
        elif kind == "uniform":
            samples[name] = rng.uniform(spec[1], spec[2], n)
        elif kind == "lognormal":
            samples[name] = rng.lognormal(spec[1], spec[2], n)
        else:
            raise ValueError(kind)
    y = np.asarray(f(samples), float)
    X = np.column_stack([samples[k] for k in dists])
    Xs = (X - X.mean(0)) / X.std(0)
    ys = (y - y.mean()) / (y.std() or 1.0)
    src = np.linalg.lstsq(np.column_stack([np.ones(n), Xs]), ys, rcond=None)[0][1:]
    return {"samples": samples, "y": y, "mean": float(y.mean()), "std": float(y.std()),
            "p05": float(np.percentile(y, 5)), "p95": float(np.percentile(y, 95)),
            "src": dict(zip(dists.keys(), src))}
