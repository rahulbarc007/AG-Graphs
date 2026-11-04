"""Six-group delayed-neutron data (Keepin) per fissioning nuclide.

``beta`` is the total delayed fraction, ``a`` the relative group abundances
and ``lam`` the precursor decay constants (1/s).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DelayedData:
    beta: float
    a: tuple
    lam: tuple

    @property
    def beta_i(self) -> np.ndarray:
        return self.beta * np.asarray(self.a)

    @property
    def lam_i(self) -> np.ndarray:
        return np.asarray(self.lam)

    @property
    def mean_life(self) -> float:
        return float(np.sum(np.asarray(self.a) / np.asarray(self.lam)))


DELAYED = {
    "U233": DelayedData(0.00265, (0.086, 0.299, 0.252, 0.278, 0.051, 0.034),
                        (0.0126, 0.0337, 0.139, 0.325, 1.13, 2.50)),
    "U235": DelayedData(0.00650, (0.033, 0.219, 0.196, 0.395, 0.115, 0.042),
                        (0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01)),
    "Pu239": DelayedData(0.00212, (0.035, 0.298, 0.211, 0.326, 0.086, 0.044),
                         (0.0128, 0.0301, 0.124, 0.325, 1.12, 2.69)),
    "Pu241": DelayedData(0.00520, (0.010, 0.229, 0.173, 0.390, 0.182, 0.016),
                         (0.0128, 0.0297, 0.124, 0.352, 1.61, 3.47)),
    "U238": DelayedData(0.01480, (0.013, 0.137, 0.162, 0.388, 0.225, 0.075),
                        (0.0132, 0.0321, 0.139, 0.358, 1.41, 4.02)),
    "Th232": DelayedData(0.02030, (0.034, 0.150, 0.155, 0.446, 0.172, 0.043),
                         (0.0124, 0.0334, 0.121, 0.321, 1.21, 3.29)),
    "Pu240": DelayedData(0.00280, (0.028, 0.273, 0.192, 0.350, 0.128, 0.029),
                         (0.0129, 0.0313, 0.135, 0.333, 1.36, 4.04)),
}

_SURROGATE = {"U232": "U233", "U234": "U235", "U236": "U235", "U237": "U235",
              "Pa231": "Th232", "Pa233": "Th232", "Np237": "U235", "Pu238": "Pu239",
              "Pu242": "Pu240", "Am241": "Pu241", "Am243": "Pu241"}


def delayed_for(nuclide: str) -> DelayedData:
    if nuclide in DELAYED:
        return DELAYED[nuclide]
    return DELAYED[_SURROGATE.get(nuclide, "U235")]


def mix_delayed(fission_rates: dict) -> DelayedData:
    """Fission-rate weighted (nu-weighted) mixture of delayed-neutron data.

    ``fission_rates`` maps nuclide -> neutron production rate (nu*F).  The
    mixture group constants preserve the total delayed-neutron emission rate
    per group.
    """
    total = sum(fission_rates.values())
    if total <= 0:
        return DELAYED["U235"]
    beta_i = np.zeros(6)
    emission = np.zeros(6)  # beta_i * lambda-weighting via precursor inventory
    inventory = np.zeros(6)
    for nuc, rate in fission_rates.items():
        d = delayed_for(nuc)
        w = rate / total
        bi = w * d.beta_i
        beta_i += bi
        inventory += bi / d.lam_i
        emission += bi
    lam = emission / np.maximum(inventory, 1e-30)
    beta = beta_i.sum()
    return DelayedData(float(beta), tuple(beta_i / beta), tuple(lam))
