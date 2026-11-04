"""Materials: molten salts, graphite, structural alloys and absorbers.

Atom densities are in atoms/(barn cm).  Molten-salt densities follow an
ideal-mixing (additive molar volume) model with linear thermal expansion,
calibrated against the Janz FLiBe/FLiNaK correlations and the MSRE / MSBR
salts, or a user-supplied correlation rho = a - b*T.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np

from .constants import AVOGADRO
from .data.nuclides import NATURAL_ABUNDANCE, NUCLIDES, REACTOR_GRADE_PU

BARN_CM = 1.0e-24

# --- compounds: element stoichiometry ------------------------------------
COMPOUNDS = {
    "LiF": {"Li": 1, "F": 1}, "BeF2": {"Be": 1, "F": 2}, "ThF4": {"Th": 1, "F": 4},
    "UF4": {"U": 1, "F": 4}, "UF3": {"U": 1, "F": 3}, "PuF3": {"Pu": 1, "F": 3},
    "ZrF4": {"Zr": 1, "F": 4}, "NaF": {"Na": 1, "F": 1}, "KF": {"K": 1, "F": 1},
    "MgF2": {"Mg": 1, "F": 2},
    "NaCl": {"Na": 1, "Cl": 1}, "KCl": {"K": 1, "Cl": 1}, "MgCl2": {"Mg": 1, "Cl": 2},
    "UCl3": {"U": 1, "Cl": 3}, "UCl4": {"U": 1, "Cl": 4}, "PuCl3": {"Pu": 1, "Cl": 3},
    "ThCl4": {"Th": 1, "Cl": 4},
}

# partial molar volume at 900 K (cm3/mol) and volumetric expansion (1/K)
MOLAR_VOLUME_900K = {
    "LiF": (13.40, 2.5e-4), "BeF2": (23.30, 2.5e-4), "ThF4": (46.70, 2.5e-4),
    "UF4": (45.50, 2.5e-4), "UF3": (38.0, 2.5e-4), "PuF3": (37.0, 2.5e-4),
    "ZrF4": (47.0, 2.5e-4), "NaF": (19.40, 2.9e-4), "KF": (28.40, 3.0e-4),
    "MgF2": (24.0, 2.5e-4),
    "NaCl": (35.40, 3.5e-4), "KCl": (46.50, 3.5e-4), "MgCl2": (55.0, 3.5e-4),
    "UCl3": (58.0, 3.0e-4), "UCl4": (104.0, 3.0e-4), "PuCl3": (60.0, 3.0e-4),
    "ThCl4": (103.0, 3.0e-4),
}


def nuclide_mass(name: str) -> float:
    return NUCLIDES[name].mass


@dataclass
class Material:
    """A homogeneous material (or homogenized mixture)."""
    name: str
    N: dict                       # nuclide -> atom density (1/b-cm)
    T: float = 900.0              # default temperature (K)
    kind: str = "generic"
    T_nuc: dict = field(default_factory=dict)   # optional per-nuclide temperatures
    meta: dict = field(default_factory=dict)    # e.g. salt specification

    def temperature_of(self, nuc: str) -> float:
        return self.T_nuc.get(nuc, self.T)

    @property
    def density(self) -> float:
        """Mass density (g/cc)."""
        return sum(n * nuclide_mass(k) for k, n in self.N.items()) / (AVOGADRO * BARN_CM)

    @property
    def heavy_metal_density(self) -> float:
        """Actinide mass density (g/cc)."""
        return sum(n * nuclide_mass(k) for k, n in self.N.items()
                   if NUCLIDES[k].is_actinide) / (AVOGADRO * BARN_CM)

    @property
    def is_fissionable(self) -> bool:
        return any(NUCLIDES[k].fissionable and v > 0 for k, v in self.N.items())

    def copy(self, name: Optional[str] = None) -> "Material":
        return Material(name or self.name, dict(self.N), self.T, self.kind, dict(self.T_nuc),
                        dict(self.meta))

    def with_temperature(self, T: float, density_factor: float = 1.0) -> "Material":
        m = self.copy()
        m.T = T
        m.T_nuc = {k: T for k in self.T_nuc} if self.T_nuc else {}
        if density_factor != 1.0:
            m.N = {k: v * density_factor for k, v in m.N.items()}
        return m

    def scaled(self, factor: float) -> "Material":
        m = self.copy()
        m.N = {k: v * factor for k, v in m.N.items()}
        return m

    def set_nuclide(self, nuc: str, density: float):
        self.N[nuc] = density

    def atom_fractions(self) -> dict:
        tot = sum(self.N.values())
        return {k: v / tot for k, v in self.N.items()}

    def summary(self) -> str:
        lines = [f"Material '{self.name}' ({self.kind}), T = {self.T:.1f} K, "
                 f"rho = {self.density:.4f} g/cc, HM = {self.heavy_metal_density:.4f} g/cc"]
        for k, v in sorted(self.N.items(), key=lambda kv: -kv[1]):
            if v > 0:
                lines.append(f"   {k:8s} {v:12.5e} /b-cm")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Molten salts
# ---------------------------------------------------------------------------
def salt_density_ideal(mol_fractions: dict, T: float, molar_mass: float) -> float:
    V = 0.0
    for comp, x in mol_fractions.items():
        v900, beta = MOLAR_VOLUME_900K[comp]
        V += x * v900 * (1.0 + beta * (T - 900.0))
    return molar_mass / V


def _element_isotopes(element: str, li7: float, uranium, plutonium, cl37) -> dict:
    if element == "Li":
        return {"Li7": li7, "Li6": 1.0 - li7}
    if element == "U":
        return uranium
    if element == "Pu":
        return plutonium
    if element == "Cl":
        if cl37 is None:
            return dict(NATURAL_ABUNDANCE["Cl"])
        return {"Cl37": cl37, "Cl35": 1.0 - cl37}
    return dict(NATURAL_ABUNDANCE[element])


def uranium_vector(enrichment: float, fissile: str = "U235") -> dict:
    """Atom-fraction uranium vector for a given enrichment (U-234 ~ 0.0089*e)."""
    if fissile == "U233":
        return {"U233": enrichment, "U238": 1.0 - enrichment} if enrichment < 1 else {"U233": 1.0}
    u234 = 0.0089 * enrichment if fissile == "U235" else 0.0
    return {"U234": u234, "U235": enrichment, "U238": max(1.0 - enrichment - u234, 0.0)}


def molten_salt(composition: dict, T: float = 900.0, li7_enrichment: float = 0.99995,
                uranium: Union[dict, float, None] = None, plutonium: Optional[dict] = None,
                cl37_enrichment: Optional[float] = None,
                density: Union[float, tuple, Callable, None] = None,
                name: str = "fuel salt") -> Material:
    """Build a molten-salt material from compound mole fractions.

    Parameters
    ----------
    composition : dict, e.g. ``{"LiF": 0.717, "BeF2": 0.16, "ThF4": 0.12, "UF4": 0.003}``
    uranium : dict of atom fractions (e.g. ``{"U233": 1.0}``) or a float enrichment
              (U-235 atom fraction).
    density : None (ideal molar-volume model), float (g/cc), (a, b) for rho=a-b*T,
              or a callable rho(T).
    """
    tot = sum(composition.values())
    x = {k: v / tot for k, v in composition.items()}
    if uranium is None:
        uranium = uranium_vector(0.0072)
    elif isinstance(uranium, (float, int)):
        uranium = uranium_vector(float(uranium))
    plutonium = plutonium or dict(REACTOR_GRADE_PU)
    # atoms of each nuclide per formula unit (molecule) of the mixture
    per_mol = {}
    for comp, xc in x.items():
        for el, n in COMPOUNDS[comp].items():
            for nuc, f in _element_isotopes(el, li7_enrichment, uranium, plutonium,
                                            cl37_enrichment).items():
                if f > 0:
                    per_mol[nuc] = per_mol.get(nuc, 0.0) + xc * n * f
    molar_mass = sum(a * nuclide_mass(k) for k, a in per_mol.items())
    if density is None:
        rho = salt_density_ideal(x, T, molar_mass)
    elif callable(density):
        rho = float(density(T))
    elif isinstance(density, tuple):
        rho = density[0] - density[1] * T
    else:
        rho = float(density)
    n_mol = rho * AVOGADRO * BARN_CM / molar_mass
    N = {k: a * n_mol for k, a in per_mol.items()}
    m = Material(name, N, T, kind="fuel_salt")
    m.meta = {"composition": x, "molar_mass": molar_mass,
              "salt_spec": dict(composition=composition, li7_enrichment=li7_enrichment,
                                uranium=uranium, plutonium=plutonium,
                                cl37_enrichment=cl37_enrichment, density=density)}
    return m


def salt_density_model(salt: Material, T: float) -> float:
    """Density (g/cc) the salt's own density model predicts at temperature T."""
    spec = salt.meta["salt_spec"]
    return molten_salt(spec["composition"], T, spec["li7_enrichment"], spec["uranium"],
                       spec["plutonium"], spec["cl37_enrichment"], spec["density"]).density


def salt_at_temperature(salt: Material, T: float) -> Material:
    """Re-evaluate a salt at temperature T, keeping its (possibly depleted)
    composition and scaling atom densities by the thermal-expansion ratio."""
    if "salt_spec" not in salt.meta:
        return salt.with_temperature(T)
    factor = salt_density_model(salt, T) / salt_density_model(salt, salt.T)
    return salt.with_temperature(T, density_factor=factor)


# Reference salts ----------------------------------------------------------
def flibe(T: float = 900.0, li7_enrichment: float = 0.99995) -> Material:
    return molten_salt({"LiF": 0.66, "BeF2": 0.34}, T, li7_enrichment,
                       density=(2.413, 4.884e-4), name="FLiBe")


def flinak(T: float = 900.0) -> Material:
    return molten_salt({"LiF": 0.465, "NaF": 0.115, "KF": 0.42}, T, 0.9241,
                       density=(2.579, 6.24e-4), name="FLiNaK")


def msre_fuel_salt(T: float = 922.0, enrichment: float = 0.33, uf4: float = 0.009) -> Material:
    """MSRE-type LiF-BeF2-ZrF4-UF4 (65-29.1-5-0.9 mol%)."""
    lif = 0.65
    bef2 = 1.0 - lif - 0.05 - uf4
    return molten_salt({"LiF": lif, "BeF2": bef2, "ZrF4": 0.05, "UF4": uf4}, T, 0.99993,
                       uranium=enrichment, name="MSRE fuel salt")


def msbr_fuel_salt(T: float = 900.0, uf4: float = 0.0025, thf4: float = 0.12) -> Material:
    """MSBR single-fluid LiF-BeF2-ThF4-UF4 (71.7-16-12-0.3 mol%) with U-233."""
    bef2 = 0.16
    lif = 1.0 - bef2 - thf4 - uf4
    return molten_salt({"LiF": lif, "BeF2": bef2, "ThF4": thf4, "UF4": uf4}, T, 0.99995,
                       uranium={"U233": 1.0}, name="MSBR fuel salt")


# ---------------------------------------------------------------------------
# Solids
# ---------------------------------------------------------------------------
def graphite(density: float = 1.86, T: float = 900.0, boron_ppm: float = 0.0) -> Material:
    """Nuclear graphite with an optional natural-boron-equivalent impurity (wt ppm)."""
    nC = density * AVOGADRO * BARN_CM / NUCLIDES["C12"].mass
    N = {"C12": nC}
    if boron_ppm > 0:
        mB = density * boron_ppm * 1e-6
        mB_mol = 0.199 * NUCLIDES["B10"].mass + 0.801 * NUCLIDES["B11"].mass
        nB = mB * AVOGADRO * BARN_CM / mB_mol
        N["B10"] = 0.199 * nB
        N["B11"] = 0.801 * nB
    return Material("graphite", N, T, kind="graphite")


def from_weight_fractions(name: str, wt: dict, density: float, T: float = 900.0,
                          kind: str = "structural") -> Material:
    elem_nuc = {"Ni": "Ni58", "Mo": "Mo96", "Cr": "Cr52", "Fe": "Fe56", "C": "C12",
                "Zr": "Zr90", "B10": "B10", "B11": "B11", "O": "O16", "Mg": "Mg24"}
    tot = sum(wt.values())
    N = {}
    for el, w in wt.items():
        nuc = elem_nuc.get(el, el)
        N[nuc] = N.get(nuc, 0.0) + density * (w / tot) * AVOGADRO * BARN_CM / NUCLIDES[nuc].mass
    return Material(name, N, T, kind=kind)


def hastelloy_n(T: float = 900.0) -> Material:
    return from_weight_fractions("Hastelloy-N", {"Ni": 71.0, "Mo": 16.0, "Cr": 7.0, "Fe": 5.0},
                                 8.86, T)


def b4c(density: float = 2.52, b10_fraction: float = 0.199, T: float = 900.0) -> Material:
    mB = b10_fraction * NUCLIDES["B10"].mass + (1 - b10_fraction) * NUCLIDES["B11"].mass
    M = 4 * mB + NUCLIDES["C12"].mass
    n = density * AVOGADRO * BARN_CM / M
    return Material("B4C", {"B10": 4 * n * b10_fraction, "B11": 4 * n * (1 - b10_fraction),
                            "C12": n}, T, kind="absorber")


def mix(materials, volume_fractions, name: str = "mixture") -> Material:
    """Volume-weighted homogenization keeping per-nuclide temperatures."""
    vf = np.asarray(volume_fractions, dtype=float)
    N = {}
    Tw = {}
    for m, f in zip(materials, vf):
        for k, v in m.N.items():
            N[k] = N.get(k, 0.0) + f * v
            Tw[k] = Tw.get(k, 0.0) + f * v * m.temperature_of(k)
    T_nuc = {k: Tw[k] / N[k] for k in N if N[k] > 0}
    T_avg = float(np.sum([f * m.T for m, f in zip(materials, vf)]) / max(vf.sum(), 1e-30))
    return Material(name, N, T_avg, kind="mixture", T_nuc=T_nuc)


def void() -> Material:
    return Material("void", {}, 300.0, kind="void")
