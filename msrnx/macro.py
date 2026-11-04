"""Macroscopic multigroup cross sections, self-shielding backgrounds and condensation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .data.nuclides import NUCLIDES
from .materials import Material
from .xs.energy import GroupStructure, collapse_map, weighting_spectrum
from .xs.library import Library, load_library

_SCALAR_KEYS = ("total", "transport", "absorption", "capture", "fission", "nu_fission",
                "kappa_fission", "nalpha", "n2n", "tritium", "invv", "chi", "chi_d")


@dataclass
class MacroXS:
    """Macroscopic group constants of one homogeneous region (1/cm)."""
    groups: GroupStructure
    total: np.ndarray
    transport: np.ndarray
    absorption: np.ndarray
    capture: np.ndarray
    fission: np.ndarray
    nu_fission: np.ndarray
    kappa_fission: np.ndarray      # MeV/cm per unit flux
    nalpha: np.ndarray
    n2n: np.ndarray
    tritium: np.ndarray
    scatter: np.ndarray            # S[g, g'] = Sigma_s(g -> g') incl. (n,2n) multiplicity
    chi: np.ndarray
    chi_d: np.ndarray
    invv: np.ndarray
    name: str = "region"
    micro: dict = field(default_factory=dict)   # nuclide -> micro dict (fine-group)
    N: dict = field(default_factory=dict)

    @property
    def G(self) -> int:
        return self.groups.G

    @property
    def D(self) -> np.ndarray:
        return 1.0 / (3.0 * self.transport)

    @property
    def removal(self) -> np.ndarray:
        return self.total - np.diag(self.scatter)

    @property
    def mu_scatter(self) -> np.ndarray:
        """P1 (within-group) scattering approximated as Sigma_t - Sigma_tr."""
        return self.total - self.transport

    def copy(self) -> "MacroXS":
        kw = {k: np.array(getattr(self, k), copy=True) for k in _SCALAR_KEYS}
        return MacroXS(self.groups, scatter=self.scatter.copy(), name=self.name,
                       micro=self.micro, N=dict(self.N), **kw)

    def k_inf(self) -> float:
        A = np.diag(self.total) - self.scatter.T
        phi = np.linalg.solve(A, self.chi)
        return float(self.nu_fission @ phi)

    def infinite_flux(self) -> np.ndarray:
        A = np.diag(self.total) - self.scatter.T
        phi = np.linalg.solve(A, self.chi)
        return phi / phi.sum()


def default_group_weights(groups: GroupStructure) -> np.ndarray:
    w = np.zeros(groups.G)
    for g in range(groups.G):
        u = np.linspace(np.log(groups.lower[g]), np.log(groups.upper[g]), 32)
        E = np.exp(u)
        w[g] = np.trapezoid(weighting_spectrum(E) * E, u)
    return w / w.sum()


def homogeneous_sig0(mat: Material, escape_xs: float = 0.0) -> dict:
    """Bondarenko background cross section per nuclide (barns).

    sigma0_i = (sum_{j != i} N_j sigma_p,j + Sigma_e) / N_i
    """
    tot_p = sum(n * NUCLIDES[k].sigma_p for k, n in mat.N.items())
    out = {}
    for k, n in mat.N.items():
        if n <= 0:
            continue
        out[k] = (tot_p - n * NUCLIDES[k].sigma_p + escape_xs) / n
    return out


def build_macro(mat: Material, lib: Optional[Library] = None, sig0: Optional[dict] = None,
                perturb: Optional[dict] = None, flux_weight: Optional[np.ndarray] = None,
                keep_micro: bool = True) -> MacroXS:
    """Assemble macroscopic cross sections of a material from the library."""
    lib = lib or load_library()
    G = lib.G
    if sig0 is None:
        sig0 = homogeneous_sig0(mat)
    w = default_group_weights(lib.groups) if flux_weight is None else np.asarray(flux_weight)
    acc = {k: np.zeros(G) for k in ("total", "transport", "absorption", "capture", "fission",
                                    "nu_fission", "kappa_fission", "nalpha", "n2n", "tritium")}
    S = np.zeros((G, G))
    chi_num = np.zeros(G)
    micro = {}
    for nuc, Nd in mat.N.items():
        if Nd <= 0 or nuc not in lib:
            continue
        m = lib[nuc].micro(mat.temperature_of(nuc), sig0.get(nuc, 1e10), perturb)
        if keep_micro:
            micro[nuc] = m
        acc["total"] += Nd * m["total"]
        acc["transport"] += Nd * (m["total"] - m["mubar"] * m["elastic"])
        acc["absorption"] += Nd * m["absorption"]
        acc["capture"] += Nd * m["capture"]
        acc["fission"] += Nd * m["fission"]
        nsf = Nd * m["nu"] * m["fission"]
        acc["nu_fission"] += nsf
        acc["kappa_fission"] += Nd * m["kappa"] * m["fission"]
        acc["nalpha"] += Nd * m["nalpha"]
        acc["n2n"] += Nd * m["n2n"]
        acc["tritium"] += Nd * m["tritium"]
        S += Nd * (m["elastic"][:, None] * m["P_el"] + m["inelastic"][:, None] * m["P_in"]
                   + 2.0 * m["n2n"][:, None] * m["P_n2n"])
        if nsf.any():
            chi_num += m["chi"] * float(nsf @ w)
    chi = chi_num / chi_num.sum() if chi_num.sum() > 0 else np.zeros(G)
    return MacroXS(lib.groups, scatter=S, chi=chi, chi_d=lib.chi_delayed.copy(),
                   invv=lib.invv.copy(), name=mat.name, micro=micro, N=dict(mat.N), **acc)


def update_chi(macro: MacroXS, flux: np.ndarray) -> MacroXS:
    """Recompute the mixture fission spectrum with the actual flux."""
    num = np.zeros(macro.G)
    for nuc, m in macro.micro.items():
        Nd = macro.N.get(nuc, 0.0)
        p = Nd * float((m["nu"] * m["fission"]) @ flux)
        if p > 0:
            num += m["chi"] * p
    if num.sum() > 0:
        macro.chi = num / num.sum()
    return macro


def combine(regions, weights, name: str = "homogenized") -> MacroXS:
    """Flux-volume weighted homogenization of several MacroXS.

    ``weights[r]`` is an array (G,) of flux*volume for region r.
    """
    weights = [np.asarray(w, float) for w in weights]
    tot_w = np.sum(weights, axis=0)
    G = regions[0].G
    out = {}
    for k in ("total", "transport", "absorption", "capture", "fission", "nu_fission",
              "kappa_fission", "nalpha", "n2n", "tritium", "invv"):
        out[k] = np.sum([getattr(r, k) * w for r, w in zip(regions, weights)], axis=0) / tot_w
    S = np.sum([r.scatter * w[:, None] for r, w in zip(regions, weights)], axis=0) / tot_w[:, None]
    prod = [float(r.nu_fission @ w) for r, w in zip(regions, weights)]
    if sum(prod) > 0:
        chi = np.sum([r.chi * p for r, p in zip(regions, prod)], axis=0) / sum(prod)
    else:
        chi = np.zeros(G)
    return MacroXS(regions[0].groups, scatter=S, chi=chi, chi_d=regions[0].chi_d.copy(),
                   name=name, **out)


def condense(macro: MacroXS, flux: np.ndarray, coarse: GroupStructure,
             D: Optional[np.ndarray] = None) -> MacroXS:
    """Flux-weighted condensation onto a coarse structure (subset of fine bounds)."""
    cmap = collapse_map(macro.groups, coarse.bounds)
    Gc = coarse.G
    flux = np.asarray(flux, float)
    phi_c = np.bincount(cmap, weights=flux, minlength=Gc)

    def fw(x):
        return np.bincount(cmap, weights=x * flux, minlength=Gc) / np.maximum(phi_c, 1e-300)

    out = {k: fw(getattr(macro, k)) for k in ("total", "absorption", "capture", "fission",
                                              "nu_fission", "kappa_fission", "nalpha", "n2n",
                                              "tritium", "invv")}
    Dfine = macro.D if D is None else np.asarray(D)
    Dc = fw(Dfine)
    out["transport"] = 1.0 / (3.0 * Dc)
    Sc = np.zeros((Gc, Gc))
    Sw = macro.scatter * flux[:, None]
    for g in range(macro.G):
        np.add.at(Sc[cmap[g]], cmap, Sw[g])
    Sc /= np.maximum(phi_c, 1e-300)[:, None]
    chi = np.bincount(cmap, weights=macro.chi, minlength=Gc)
    chi_d = np.bincount(cmap, weights=macro.chi_d, minlength=Gc)
    return MacroXS(coarse, scatter=Sc, chi=chi, chi_d=chi_d, name=macro.name, N=dict(macro.N),
                   **out)


def collapse_micro(micro: dict, flux: np.ndarray, reactions=("capture", "fission", "nalpha",
                                                             "n2n", "tritium", "absorption")) -> dict:
    """One-group microscopic cross sections (barns) per nuclide."""
    flux = np.asarray(flux, float)
    phi = flux.sum()
    out = {}
    for nuc, m in micro.items():
        d = {rx: float(m[rx] @ flux) / phi for rx in reactions}
        d["nu_fission"] = float((m["nu"] * m["fission"]) @ flux) / phi
        out[nuc] = d
    return out
