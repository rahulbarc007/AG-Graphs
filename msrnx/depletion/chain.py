"""Depletion chain and burnup matrix with MSR online reprocessing.

The chain is generated from the nuclide database: decay (with branching),
(n,gamma), fission (to the explicit poisons and chemistry-class lumps),
(n,2n), (n,alpha)/(n,t) and the Li-7 (n,n't) tritium channel.  Online fuel
processing adds first-order removal terms per chemistry class (He sparging
of noble gases, noble-metal plate-out, rare-earth extraction, Pa isolation
in a decay tank that returns U-233 to the salt, salt discard).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..constants import SECONDS_PER_DAY
from ..data.fission_yields import yields_for
from ..data.nuclides import NUCLIDES

CHEM_CLASSES = ("noble_gas", "noble_metal", "rare_earth", "salt_seeker", "halogen", "tritium",
                "Pa", "U", "Pu", "Np", "Am", "Th")


@dataclass
class Reprocessing:
    """First-order online removal: class -> removal time constant (days).

    ``pa_decay_tank``: removed Pa-233 decays outside the flux in a tank and
    the U-233 is returned to the fuel salt (MSBR protactinium isolation).
    """
    removal_days: dict = field(default_factory=dict)
    pa_decay_tank: bool = True
    name: str = "custom"

    def rate(self, chem: str) -> float:
        T = self.removal_days.get(chem)
        return 0.0 if not T else 1.0 / (T * SECONDS_PER_DAY)

    @staticmethod
    def none() -> "Reprocessing":
        return Reprocessing({}, False, "no processing")

    @staticmethod
    def sparging_only(noble_gas_s: float = 60.0) -> "Reprocessing":
        return Reprocessing({"noble_gas": noble_gas_s / SECONDS_PER_DAY,
                             "tritium": 1.0}, False, "He sparging only")

    @staticmethod
    def msbr() -> "Reprocessing":
        """MSBR reference processing (ORNL-4541-like cycle times)."""
        s = 1.0 / SECONDS_PER_DAY
        return Reprocessing({"noble_gas": 50 * s, "noble_metal": 50 * s, "rare_earth": 50.0,
                             "Pa": 3.0, "salt_seeker": 3435.0, "halogen": 60.0,
                             "tritium": 1.0}, True, "MSBR reference")


class DepletionChain:
    """Nuclide indexing and burnup-matrix assembly."""

    TANK = "Pa233_tank"

    def __init__(self, nuclides=None, reprocessing: Optional[Reprocessing] = None):
        self.repro = reprocessing or Reprocessing.none()
        base = [n for n in (nuclides or NUCLIDES.keys())]
        self.salt_nuclides = base
        self.pseudo = [self.TANK] + [f"REMOVED_{c}" for c in CHEM_CLASSES]
        self.names = base + self.pseudo
        self.index = {n: i for i, n in enumerate(self.names)}
        self.n = len(self.names)

    def vector(self, N_dict: dict) -> np.ndarray:
        v = np.zeros(self.n)
        for k, x in N_dict.items():
            if k in self.index:
                v[self.index[k]] = x
        return v

    def salt_dict(self, vec: np.ndarray) -> dict:
        return {n: float(vec[self.index[n]]) for n in self.salt_nuclides
                if vec[self.index[n]] > 0}

    def matrix(self, one_group: dict, flux: float) -> np.ndarray:
        """Burnup matrix A (1/s); one-group XS in barns, flux in n/cm2/s."""
        A = np.zeros((self.n, self.n))
        idx = self.index
        phi = flux * 1e-24
        for name in self.salt_nuclides:
            j = idx[name]
            nuc = NUCLIDES[name]
            xs = one_group.get(name, {})
            # decay
            lam = nuc.decay_constant
            if lam > 0:
                A[j, j] -= lam
                for d, br in nuc.decay:
                    if d in idx:
                        A[idx[d], j] += br * lam
            # capture
            rc = xs.get("capture", 0.0) * phi
            if rc > 0 and nuc.capture_product != "self":
                A[j, j] -= rc
                if nuc.capture_product in idx:
                    A[idx[nuc.capture_product], j] += rc
            # fission
            rf = xs.get("fission", 0.0) * phi
            if rf > 0:
                A[j, j] -= rf
                for p, y in yields_for(name).items():
                    A[idx[p], j] += y * rf
            # (n,2n)
            r2 = xs.get("n2n", 0.0) * phi
            if r2 > 0:
                A[j, j] -= r2
                if nuc.n2n_product in idx:
                    A[idx[nuc.n2n_product], j] += r2
            # (n,alpha) / (n,t) / (n,p)
            ra = xs.get("nalpha", 0.0) * phi
            if ra > 0:
                A[j, j] -= ra
                if nuc.nalpha_product in idx:
                    A[idx[nuc.nalpha_product], j] += ra
            # Li-7 (n,n't) tritium channel
            rt = xs.get("tritium", 0.0) * phi
            if rt > 0 and "H3" in idx:
                A[j, j] -= rt
                A[idx["H3"], j] += rt
            # online processing
            chem = nuc.chem
            lr = self.repro.rate(chem)
            if lr > 0:
                A[j, j] -= lr
                if name == "Pa233" and self.repro.pa_decay_tank:
                    A[idx[self.TANK], j] += lr
                else:
                    A[idx[f"REMOVED_{chem}" if chem in CHEM_CLASSES else "REMOVED_salt_seeker"], j] += lr
        # Pa-233 decay tank: decays out of flux, U-233 returned to the salt
        t = idx[self.TANK]
        lamPa = NUCLIDES["Pa233"].decay_constant
        A[t, t] -= lamPa
        A[idx["U233"], t] += lamPa
        return A

    def fissile_flows(self, A: np.ndarray, N: np.ndarray, one_group: dict, flux: float) -> dict:
        """Instantaneous fissile production/destruction rates (atoms/b-cm/s)."""
        fissile = [n for n in ("U233", "U235", "Pu239", "Pu241") if n in self.index]
        fi = [self.index[n] for n in fissile]
        others = [i for i in range(self.n) if i not in fi]
        prod = 0.0
        for i in fi:
            row = A[i, others] * N[others]
            prod += row[row > 0].sum()
        destr = 0.0
        for n in fissile:
            destr += N[self.index[n]] * one_group.get(n, {}).get("absorption", 0.0) * flux * 1e-24

        def cap(n):
            return (N[self.index[n]] * one_group.get(n, {}).get("capture", 0.0) * flux * 1e-24
                    if n in self.index else 0.0)

        # capture-based breeding ratio (valid at all times, standard definition):
        # fertile captures that lead to fissile minus Pa-233 parasitic captures
        gain = cap("Th232") + cap("U238") + cap("Pu240") + cap("U234") - cap("Pa233")
        return {"production": prod, "destruction": destr,
                "breeding_ratio": gain / destr if destr > 0 else 0.0,
                "fissile_production_ratio": prod / destr if destr > 0 else 0.0}
