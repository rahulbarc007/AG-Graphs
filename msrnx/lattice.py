"""Lattice (cell) physics for graphite-moderated and homogeneous MSR fuel.

* Resonance self-shielding by equivalence theory: Bondarenko background
  sigma_0 with an escape cross section from the Bell-Dancoff rational
  approximation (Sauer Dancoff factor for square / hexagonal lattices).
* Two-region (fuel channel / graphite) flat-flux collision-probability
  solution with Carlvik's two-term fuel escape probability, Dancoff
  correction and reciprocity -> k_inf, region spectra, disadvantage factors.
* Flux-volume homogenization, B1 / P1 fundamental-mode leakage, critical
  buckling search, group condensation and the six-factor formula.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from .data.nuclides import NUCLIDES
from .macro import MacroXS, build_macro, combine, homogeneous_sig0, update_chi
from .materials import Material
from .xs.energy import GroupStructure
from .xs.library import Library, load_library

E_CUT_SIX_FACTOR = 0.6826   # eV, thermal cut-off for the six-factor formula


@dataclass
class CellGeometry:
    """Equivalent cylindrical fuel channel in a graphite lattice cell."""
    channel_radius: float = 1.0     # cm
    salt_fraction: float = 0.20     # fuel-salt volume fraction of the cell
    lattice: str = "square"         # 'square' or 'hex'
    bell: float = 1.20              # Bell factor of the rational approximation

    @property
    def cell_area(self) -> float:
        return np.pi * self.channel_radius ** 2 / self.salt_fraction

    @property
    def pitch(self) -> float:
        if self.lattice == "hex":
            return float(np.sqrt(2.0 * self.cell_area / np.sqrt(3.0)))
        return float(np.sqrt(self.cell_area))

    @property
    def mean_chord_fuel(self) -> float:
        return 2.0 * self.channel_radius

    @property
    def mean_chord_mod(self) -> float:
        vm_vf = (1.0 - self.salt_fraction) / self.salt_fraction
        return 2.0 * self.channel_radius * vm_vf

    def dancoff(self, sigma_mod) -> np.ndarray:
        """Sauer's approximation of the Dancoff factor."""
        sigma_mod = np.asarray(sigma_mod, float)
        vm_vf = (1.0 - self.salt_fraction) / self.salt_fraction
        if self.lattice == "hex":
            tau = (np.sqrt(np.pi / (2 * np.sqrt(3.0)) * (1 + vm_vf)) - 1) / vm_vf - 0.12
        else:
            tau = (np.sqrt(np.pi / 4.0 * (1 + vm_vf)) - 1) / vm_vf - 0.08
        x = sigma_mod * self.mean_chord_mod
        return np.exp(-tau * x) / (1.0 + (1.0 - tau) * x)


@dataclass
class LatticeResult:
    groups: GroupStructure
    k_inf: float
    phi_fuel: np.ndarray
    phi_mod: np.ndarray
    phi_cell: np.ndarray
    hom: MacroXS
    fuel_macro: MacroXS
    mod_macro: Optional[MacroXS]
    sig0: dict
    dancoff: np.ndarray
    V_fuel: float
    V_mod: float
    cell: Optional[CellGeometry]
    heterogeneous: bool
    six_factor: dict = field(default_factory=dict)
    balance: dict = field(default_factory=dict)

    @property
    def fuel_flux_ratio(self) -> np.ndarray:
        """phi_fuel / phi_cell per group (for fuel reaction rates in homogenized cores)."""
        return self.phi_fuel / np.maximum(self.phi_cell, 1e-300)

    @property
    def thermal_disadvantage(self) -> float:
        th = self.groups.upper <= E_CUT_SIX_FACTOR * 1.0001
        return float(self.phi_mod[th].sum() / max(self.phi_fuel[th].sum(), 1e-300))

    def lethargy_spectrum(self, which: str = "cell") -> np.ndarray:
        phi = {"cell": self.phi_cell, "fuel": self.phi_fuel, "mod": self.phi_mod}[which]
        return phi / self.groups.lethargy_width


def heterogeneous_sig0(fuel: Material, cell: CellGeometry, sigma_mod_res: float) -> dict:
    """Equivalence-theory background cross sections for nuclides in the fuel channel."""
    C = float(cell.dancoff(sigma_mod_res))
    a = cell.bell
    Sigma_e = a * (1.0 - C) / (cell.mean_chord_fuel * (1.0 + (a - 1.0) * C))
    return homogeneous_sig0(fuel, escape_xs=Sigma_e)


def _carlvik_escape(x):
    x = np.maximum(x, 1e-12)
    return 4.0 / (x + 2.0) - 3.0 / (x + 3.0)


def _solve_rank_one(A, chi_vec, nsf_vec):
    phi = np.linalg.solve(A, chi_vec)
    k = float(nsf_vec @ phi)
    return k, phi / k


def _infinite_homogeneous(macro: MacroXS):
    A = np.diag(macro.transport) - (macro.scatter - np.diag(macro.total - macro.transport)).T
    k, phi = _solve_rank_one(A, macro.chi, macro.nu_fission)
    return k, phi


def run_lattice(fuel: Material, moderator: Optional[Material] = None,
                cell: Optional[CellGeometry] = None, lib: Optional[Library] = None,
                heterogeneous: bool = True, perturb: Optional[dict] = None,
                n_chi_iter: int = 3) -> LatticeResult:
    """Lattice calculation of a fuel-salt channel cell (or homogeneous salt).

    If ``moderator`` is None (or the salt fraction is 1) the fuel is treated as
    an infinite homogeneous medium (e.g. a fast-spectrum chloride MSR).
    """
    lib = lib or load_library()
    G = lib.G
    homogeneous_fuel_only = moderator is None or cell is None or cell.salt_fraction >= 0.999
    if homogeneous_fuel_only:
        sig0 = homogeneous_sig0(fuel)
        fm = build_macro(fuel, lib, sig0, perturb)
        for _ in range(n_chi_iter):
            k, phi = _infinite_homogeneous(fm)
            update_chi(fm, phi)
        k, phi = _infinite_homogeneous(fm)
        res = LatticeResult(lib.groups, k, phi, np.zeros(G), phi, fm, fm, None, sig0,
                            np.zeros(G), 1.0, 0.0, cell, False)
        res.six_factor = six_factor(fm, phi)
        res.balance = neutron_balance(fm, phi)
        return res

    Vf = cell.salt_fraction
    Vm = 1.0 - Vf
    mm = build_macro(moderator, lib, None, perturb)
    g_res = lib.groups.group_of(100.0)
    C_res = float(cell.dancoff(mm.total[g_res]))
    if heterogeneous:
        sig0 = heterogeneous_sig0(fuel, cell, mm.total[g_res])
    else:
        smeared = Material("smeared", {**{k: v * Vf for k, v in fuel.N.items()},
                                      **{k: v * Vm for k, v in moderator.N.items()}})
        s_all = homogeneous_sig0(smeared)
        sig0 = {k: s_all[k] for k in fuel.N if k in s_all}
    fm = build_macro(fuel, lib, sig0, perturb)

    if not heterogeneous:
        hom0 = combine([fm, mm], [np.full(G, Vf), np.full(G, Vm)], name="homogeneous cell")
        hom0.micro, hom0.N = fm.micro, {k: v * Vf for k, v in fuel.N.items()}
        for _ in range(n_chi_iter):
            k, phi = _infinite_homogeneous(hom0)
            update_chi(fm, phi)
            hom0.chi = fm.chi.copy()
        k, phi = _infinite_homogeneous(hom0)
        res = LatticeResult(lib.groups, k, phi, phi, phi, hom0, fm, mm, sig0,
                            np.full(G, C_res), Vf, Vm, cell, False)
        res.six_factor = six_factor(hom0, phi, fuel_macro=fm, fuel_flux=phi * Vf)
        res.balance = neutron_balance(hom0, phi)
        return res

    # ---- two-region collision probabilities (per group) ----
    Sf, Sm = fm.transport, mm.transport
    C = cell.dancoff(Sm)
    x = Sf * cell.mean_chord_fuel
    Pesc_iso = _carlvik_escape(x)
    Pesc = Pesc_iso * (1.0 - C) / (1.0 - C * (1.0 - x * Pesc_iso))
    Pfm = np.clip(Pesc, 0.0, 1.0)
    Pff = 1.0 - Pfm
    Pmf = np.clip(Sf * Vf * Pfm / (Sm * Vm), 0.0, 1.0)
    Pmm = 1.0 - Pmf
    # T maps collision sources to fluxes: phi = T Q
    T = np.zeros((2 * G, 2 * G))
    idx = np.arange(G)
    T[idx, idx] = Pff / Sf
    T[idx, G + idx] = Pmf * Vm / (Sf * Vf)
    T[G + idx, idx] = Pfm * Vf / (Sm * Vm)
    T[G + idx, G + idx] = Pmm / Sm
    Sblk = np.zeros((2 * G, 2 * G))
    Sf_tc = fm.scatter - np.diag(fm.total - fm.transport)
    Sm_tc = mm.scatter - np.diag(mm.total - mm.transport)
    Sblk[:G, :G] = Sf_tc.T
    Sblk[G:, G:] = Sm_tc.T
    A = np.eye(2 * G) - T @ Sblk
    nsf = np.concatenate([fm.nu_fission, np.zeros(G)])
    for it in range(n_chi_iter + 1):
        src = T @ np.concatenate([fm.chi, np.zeros(G)])
        k, phi = _solve_rank_one(A, src, nsf)
        if it < n_chi_iter:
            update_chi(fm, phi[:G])
    phi_f, phi_m = phi[:G], phi[G:]
    phi_cell = phi_f * Vf + phi_m * Vm
    hom = combine([fm, mm], [phi_f * Vf, phi_m * Vm], name="homogenized cell")
    hom.micro = fm.micro
    hom.N = {k: v * Vf for k, v in fuel.N.items()}
    res = LatticeResult(lib.groups, k, phi_f, phi_m, phi_cell, hom, fm, mm, sig0, C, Vf, Vm,
                        cell, True)
    res.six_factor = six_factor(hom, phi_cell, fuel_macro=fm, fuel_flux=phi_f * Vf)
    res.balance = neutron_balance(hom, phi_cell, fuel_macro=fm, fuel_flux=phi_f * Vf,
                                  mod_macro=mm, mod_flux=phi_m * Vm)
    return res


# ---------------------------------------------------------------------------
# Fundamental mode (B1 / P1) leakage
# ---------------------------------------------------------------------------
def _b1_alpha(x):
    """B1 correction alpha(x), x = B / Sigma_t (x^2 may be negative)."""
    x2 = np.asarray(x, float)
    out = np.ones_like(x2)
    small = np.abs(x2) < 1e-4
    out[small] = 1.0 + 4.0 * x2[small] / 15.0
    pos = (~small) & (x2 > 0)
    if np.any(pos):
        xx = np.sqrt(x2[pos])
        at = np.arctan(xx)
        out[pos] = xx ** 2 * at / (3.0 * (xx - at))
    neg = (~small) & (x2 < 0)
    if np.any(neg):
        xx = np.minimum(np.sqrt(-x2[neg]), 0.999)
        ath = np.arctanh(xx)
        out[neg] = xx ** 2 * ath / (3.0 * (ath - xx))
    return out


def leakage_D(macro: MacroXS, B2: float, method: str = "B1") -> np.ndarray:
    if method.upper() == "P1":
        return macro.D
    x2 = B2 / macro.total ** 2
    alpha = _b1_alpha(x2)
    return 1.0 / (3.0 * (alpha * macro.total - (macro.total - macro.transport)))


def fundamental_mode(macro: MacroXS, B2: float, method: str = "B1"):
    """k_eff and spectrum of an infinite medium with buckling B2 (1/cm^2)."""
    D = leakage_D(macro, B2, method)
    A = np.diag(macro.total + D * B2) - macro.scatter.T
    k, phi = _solve_rank_one(A, macro.chi, macro.nu_fission)
    return k, phi / phi.sum(), D


def critical_buckling(macro: MacroXS, method: str = "B1") -> float:
    kinf = macro.k_inf()
    f = lambda b2: fundamental_mode(macro, b2, method)[0] - 1.0
    if kinf >= 1.0:
        hi = 1e-4
        while f(hi) > 0 and hi < 10.0:
            hi *= 2.0
        return brentq(f, 0.0, hi, xtol=1e-12)
    lo = -1e-4
    while f(lo) < 0 and lo > -0.5 * float(np.min(macro.total)) ** 2:
        lo *= 2.0
    return brentq(f, lo, 0.0, xtol=1e-12)


def geometric_buckling(radius: float, height: float, extrap: float = 0.0) -> float:
    R = radius + extrap
    H = height + 2 * extrap
    return (2.405 / R) ** 2 + (np.pi / H) ** 2


# ---------------------------------------------------------------------------
# Neutron economy diagnostics
# ---------------------------------------------------------------------------
def _hm(name: str) -> bool:
    return NUCLIDES[name].is_actinide


def six_factor(macro: MacroXS, phi: np.ndarray, B2: float = 0.0, D: Optional[np.ndarray] = None,
               fuel_macro: Optional[MacroXS] = None, fuel_flux: Optional[np.ndarray] = None) -> dict:
    """Six-factor decomposition k = eta f p eps P_FNL P_TNL (heavy-metal 'fuel').

    ``phi`` is the homogenized flux; ``fuel_flux`` (flux*volume in the fuel
    region per unit cell volume) is used for heavy-metal reaction rates.
    """
    g = macro.groups
    th = g.upper <= E_CUT_SIX_FACTOR * 1.0001
    fa = ~th
    D = macro.D if D is None else D
    leak = D * B2 * phi
    prod = macro.nu_fission * phi
    absr = macro.absorption * phi - macro.n2n * phi
    src_micro = fuel_macro.micro if fuel_macro is not None else macro.micro
    src_N = fuel_macro.N if fuel_macro is not None else macro.N
    fflux = fuel_flux if fuel_flux is not None else phi
    hm_abs = np.zeros(g.G)
    for nuc, m in src_micro.items():
        if _hm(nuc):
            hm_abs += src_N.get(nuc, 0.0) * m["absorption"] * fflux
    S = macro.scatter
    q = float((S[np.ix_(fa, th)].sum(axis=1) * phi[fa]).sum()
              - (S[np.ix_(th, fa)].sum(axis=1) * phi[th]).sum())
    P, P_th = prod.sum(), prod[th].sum()
    A_f, A_th = absr[fa].sum(), absr[th].sum()
    L_f, L_th = leak[fa].sum(), leak[th].sum()
    born = A_f + L_f + q
    eps = P / P_th
    p = q / (A_f + q)
    f = hm_abs[th].sum() / A_th
    eta = P_th / hm_abs[th].sum()
    pfnl = 1.0 - L_f / born
    ptnl = A_th / (A_th + L_th)
    k = eta * f * p * eps * pfnl * ptnl
    return {"eta": eta, "f": f, "p": p, "epsilon": eps, "P_FNL": pfnl, "P_TNL": ptnl, "k": k}


def neutron_balance(macro: MacroXS, phi: np.ndarray, fuel_macro=None, fuel_flux=None,
                    mod_macro=None, mod_flux=None) -> dict:
    """Absorption fraction by nuclide (normalized to total absorption)."""
    out = {}
    if fuel_macro is None:
        fuel_macro, fuel_flux = macro, phi
    for nuc, m in fuel_macro.micro.items():
        out[nuc] = out.get(nuc, 0.0) + fuel_macro.N.get(nuc, 0.0) * float(
            (m["absorption"] - m["n2n"]) @ fuel_flux)
    if mod_macro is not None:
        for nuc, m in mod_macro.micro.items():
            out[nuc] = out.get(nuc, 0.0) + mod_macro.N.get(nuc, 0.0) * float(
                (m["absorption"] - m["n2n"]) @ mod_flux)
    tot = sum(v for v in out.values() if v > 0)
    return {k: v / tot for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


def conversion_ratio(fuel_macro: MacroXS, fuel_flux: np.ndarray) -> float:
    """Instantaneous conversion ratio: fertile captures / fissile absorptions."""
    m, N = fuel_macro.micro, fuel_macro.N

    def rate(nuc, rx):
        return N.get(nuc, 0.0) * float(m[nuc][rx] @ fuel_flux) if nuc in m else 0.0

    gain = (rate("Th232", "capture") + rate("U238", "capture") + rate("Pu240", "capture")
            + rate("U234", "capture") - rate("Pa233", "capture"))
    loss = sum(rate(n, "absorption") for n in ("U233", "U235", "Pu239", "Pu241"))
    return gain / loss if loss > 0 else 0.0
