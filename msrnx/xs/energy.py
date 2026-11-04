"""Energy group structures.

``FINE45`` is the MSR-NX fine (lattice) structure: 45 groups from 20 MeV to
1e-5 eV with uniform lethargy width 0.5 between 10 MeV and 7.6 meV and 14
thermal groups below 3.06 eV (where the free-gas kernel is applied).
Few-group structures used by the core solver are defined by subsets of the
FINE45 boundaries so that condensation is exact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _fine45() -> np.ndarray:
    b = [2.0e7] + [1.0e7 * np.exp(-0.5 * k) for k in range(43)] + [3.0e-3, 1.0e-5]
    return np.array(b)


@dataclass(frozen=True)
class GroupStructure:
    name: str
    bounds: np.ndarray  # descending energy boundaries (eV), length G+1

    @property
    def G(self) -> int:
        return len(self.bounds) - 1

    @property
    def upper(self) -> np.ndarray:
        return self.bounds[:-1]

    @property
    def lower(self) -> np.ndarray:
        return self.bounds[1:]

    @property
    def midpoints(self) -> np.ndarray:
        """Geometric-mean energy of each group (eV)."""
        return np.sqrt(self.upper * self.lower)

    @property
    def lethargy_width(self) -> np.ndarray:
        return np.log(self.upper / self.lower)

    def group_of(self, E: float) -> int:
        """Index of the group containing energy E."""
        idx = np.searchsorted(-self.bounds, -E, side="right") - 1
        return int(np.clip(idx, 0, self.G - 1))

    def thermal_groups(self, cutoff: float = 3.06) -> np.ndarray:
        """Indices of groups lying entirely below ``cutoff`` eV."""
        return np.where(self.upper <= cutoff * (1 + 1e-9))[0]


FINE45 = GroupStructure("MSRNX-45", _fine45())
THERMAL_CUTOFF_EV = 3.06


def collapse_map(fine: GroupStructure, coarse_bounds) -> np.ndarray:
    """Map each fine group to its coarse group (coarse bounds must be a subset)."""
    coarse_bounds = np.asarray(coarse_bounds, dtype=float)
    mapping = np.empty(fine.G, dtype=int)
    for g in range(fine.G):
        Emid = fine.midpoints[g]
        c = np.searchsorted(-coarse_bounds, -Emid, side="right") - 1
        mapping[g] = int(np.clip(c, 0, len(coarse_bounds) - 2))
    # verify subset property
    for cb in coarse_bounds:
        if np.min(np.abs(np.log(fine.bounds / cb))) > 1e-6:
            raise ValueError(f"Coarse boundary {cb:g} eV is not a fine-group boundary")
    return mapping


def nearest_bounds(fine: GroupStructure, energies) -> np.ndarray:
    """Snap requested boundaries to the nearest fine-group boundaries."""
    out = []
    for E in energies:
        i = int(np.argmin(np.abs(np.log(fine.bounds / E))))
        out.append(fine.bounds[i])
    return np.array(sorted(set(out), reverse=True))


# Standard few-group structures (subsets of FINE45)
FEW_GROUP_PRESETS = {
    "2g": nearest_bounds(FINE45, [2.0e7, 0.6826, 1.0e-5]),
    "4g": nearest_bounds(FINE45, [2.0e7, 8.2085e5, 5.5308e3, 0.6826, 1.0e-5]),
    "6g": nearest_bounds(FINE45, [2.0e7, 8.2085e5, 6.7379e4, 1.0130e2, 3.059, 0.4140, 1.0e-5]),
    "8g": nearest_bounds(FINE45, [2.0e7, 2.2313e6, 4.9787e5, 6.7379e4, 5.5308e3, 1.0130e2,
                                  3.059, 0.4140, 1.0e-5]),
}


def few_group(name_or_bounds) -> GroupStructure:
    if isinstance(name_or_bounds, str):
        b = FEW_GROUP_PRESETS[name_or_bounds]
        return GroupStructure(name_or_bounds, b)
    b = nearest_bounds(FINE45, name_or_bounds)
    return GroupStructure(f"{len(b) - 1}g-custom", b)


def watt_spectrum(E_eV, a_MeV: float, b_invMeV: float) -> np.ndarray:
    E = np.asarray(E_eV) / 1.0e6
    return np.exp(-E / a_MeV) * np.sinh(np.sqrt(b_invMeV * E))


def maxwellian_flux(E_eV, kT_eV: float) -> np.ndarray:
    """Maxwellian flux shape E/(kT)^2 exp(-E/kT) (per eV)."""
    E = np.asarray(E_eV)
    return E / kT_eV ** 2 * np.exp(-E / kT_eV)


def group_spectrum(groups: GroupStructure, shape, n_per_group: int = 64) -> np.ndarray:
    """Integrate a spectrum shape(E) over each group (normalized to sum 1)."""
    vals = np.zeros(groups.G)
    for g in range(groups.G):
        u = np.linspace(np.log(groups.lower[g]), np.log(groups.upper[g]), n_per_group)
        E = np.exp(u)
        vals[g] = np.trapezoid(shape(E) * E, u)
    vals = np.maximum(vals, 0.0)
    return vals / vals.sum()


def weighting_spectrum(E_eV, T_w: float = 900.0) -> np.ndarray:
    """Library weighting spectrum: Maxwellian + 1/E + fission (per eV).

    Continuous at the junctions 5kT and 0.5 MeV.
    """
    from ..constants import kT
    E = np.asarray(E_eV, dtype=float)
    kt = kT(T_w)
    E1 = 5.0 * kt
    E2 = 5.0e5
    M = maxwellian_flux(E, kt)
    m1 = maxwellian_flux(E1, kt) * E1   # match c/E at E1
    W = np.where(E < E1, M, m1 / E)
    chi2 = watt_spectrum(E2, 0.988, 2.249)
    c2 = (m1 / E2) / chi2
    W = np.where(E > E2, c2 * watt_spectrum(E, 0.988, 2.249), W)
    return W
