"""Multigroup microscopic cross-section library with Bondarenko tables.

For every nuclide the library stores, on the FINE45 group structure,

* self-shielded capture / fission / elastic cross sections tabulated on a
  (temperature x background cross section sigma_0) grid, generated with the
  narrow-resonance flux  phi(E) = W(E) / (sigma_t(E) + sigma_0),
* smooth threshold reactions (inelastic, (n,2n), (n,alpha), tritium tally),
* row-normalized transfer matrices: elastic (free-gas thermal kernel for
  light nuclides, asymptotic kernel otherwise), inelastic and (n,2n),
* nu, chi (Watt) and recoverable energy per fission.

The library is generated once (about a minute) and cached on disk; the
cache key is a hash of the nuclide database so edits invalidate it.
External group constants can be injected with :meth:`Library.add_external`.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from ..constants import neutron_speed_cm_s
from ..data.nuclides import NUCLIDES, Nuclide
from .energy import FINE45, GroupStructure, THERMAL_CUTOFF_EV, group_spectrum, watt_spectrum, weighting_spectrum
from .kernels import (elastic_asymptotic_matrix, evaporation_matrix, free_gas_group_matrix)
from .pointwise import PointwiseNuclide, _potential

LIBRARY_VERSION = "msrnx-lib-1.0"
T_GRID = np.array([293.6, 600.0, 900.0, 1200.0, 1600.0])
SIG0_GRID = np.array([1.0, 3.0, 10.0, 30.0, 1.0e2, 3.0e2, 1.0e3, 3.0e3, 1.0e4, 3.0e4, 1.0e5,
                      1.0e6, 1.0e10])


@dataclass
class NuclideMG:
    name: str
    awr: float
    T_grid: np.ndarray
    sig0_grid: np.ndarray
    capture: np.ndarray      # (nT, nS, G)
    fission: np.ndarray
    elastic: np.ndarray
    nalpha: np.ndarray       # (G,)
    inelastic: np.ndarray
    n2n: np.ndarray
    tritium: np.ndarray
    P_el: np.ndarray         # (nTk, G, G)
    P_el_T: np.ndarray       # temperatures of P_el
    P_in: np.ndarray
    P_n2n: np.ndarray
    nu: np.ndarray
    chi: np.ndarray
    kappa: float
    mubar: np.ndarray
    resonant: bool = False

    def _interp_T(self, arr, T):
        """Linear interpolation in sqrt(T) along axis 0."""
        Tg = self.T_grid if arr.shape[0] == len(self.T_grid) else self.P_el_T
        if len(Tg) == 1:
            return arr[0]
        x = np.sqrt(np.clip(T, Tg[0], Tg[-1]))
        xs = np.sqrt(Tg)
        i = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
        f = (x - xs[i]) / (xs[i + 1] - xs[i])
        return (1 - f) * arr[i] + f * arr[i + 1]

    def _interp_sig0(self, arr2, sig0):
        """arr2: (nS, G); linear in log(sigma0)."""
        if arr2.shape[0] == 1:
            return arr2[0]
        ls = np.log(self.sig0_grid)
        x = np.log(np.clip(sig0, self.sig0_grid[0], self.sig0_grid[-1]))
        i = int(np.clip(np.searchsorted(ls, x) - 1, 0, len(ls) - 2))
        f = (x - ls[i]) / (ls[i + 1] - ls[i])
        return (1 - f) * arr2[i] + f * arr2[i + 1]

    def micro(self, T: float, sig0: float = 1e10, perturb: dict | None = None) -> dict:
        """Microscopic group cross sections at temperature T and dilution sig0."""
        out = {}
        for key in ("capture", "fission", "elastic"):
            arr = getattr(self, key)
            out[key] = self._interp_sig0(self._interp_T(arr, T), sig0)
        out["nalpha"] = self.nalpha
        out["inelastic"] = self.inelastic
        out["n2n"] = self.n2n
        out["tritium"] = self.tritium
        out["nu"] = self.nu
        out["chi"] = self.chi
        out["P_el"] = self._interp_T(self.P_el, T)
        out["P_in"] = self.P_in
        out["P_n2n"] = self.P_n2n
        out["mubar"] = self.mubar
        out["kappa"] = self.kappa
        if perturb:
            for (nuc, rx), fac in perturb.items():
                if nuc == self.name and rx in out:
                    out[rx] = out[rx] * fac
        out["absorption"] = out["capture"] + out["fission"] + out["nalpha"]
        out["total"] = out["absorption"] + out["elastic"] + out["inelastic"] + out["n2n"]
        return out


def _nr_group_average(E, sig_t, reactions: dict, weight, groups: GroupStructure, sig0_grid):
    """Narrow-resonance group averages for each sigma_0 (trapezoid on segments)."""
    mids = np.sqrt(E[:-1] * E[1:])
    seg_g = np.clip(np.searchsorted(-groups.bounds, -mids, side="right") - 1, 0, groups.G - 1)
    inside = (mids <= groups.bounds[0]) & (mids >= groups.bounds[-1])
    dE = np.diff(E)
    W = weight(E)
    res = {k: np.zeros((len(sig0_grid), groups.G)) for k in reactions}
    for s, sig0 in enumerate(sig0_grid):
        phi = W / (sig_t + sig0)
        fseg = 0.5 * (phi[:-1] + phi[1:]) * dE
        den = np.bincount(seg_g[inside], weights=fseg[inside], minlength=groups.G)
        for k, sx in reactions.items():
            num_seg = 0.5 * (sx[:-1] * phi[:-1] + sx[1:] * phi[1:]) * dE
            num = np.bincount(seg_g[inside], weights=num_seg[inside], minlength=groups.G)
            res[k][s] = num / np.maximum(den, 1e-300)
    return res


def build_nuclide(nuc: Nuclide, groups: GroupStructure = FINE45, T_grid=T_GRID,
                  sig0_grid=SIG0_GRID) -> NuclideMG:
    weight = weighting_spectrum
    pw = PointwiseNuclide(nuc, extra_points=groups.bounds)
    resonant = pw.has_resonances
    s_grid = np.asarray(sig0_grid) if resonant else np.array([1e10])
    nT = len(T_grid)
    G = groups.G
    cap = np.zeros((nT, len(s_grid), G))
    fis = np.zeros_like(cap)
    el = np.zeros_like(cap)
    smooth_rx = None
    for it, T in enumerate(T_grid):
        if not resonant and it > 0:
            cap[it], fis[it], el[it] = cap[0], fis[0], el[0]
            continue
        px = pw.at(float(T))
        rx = {"capture": px.capture, "fission": px.fission, "elastic": px.elastic}
        if smooth_rx is None:
            smooth_rx = {"nalpha": px.nalpha, "inelastic": px.inelastic, "n2n": px.n2n,
                         "tritium": px.tritium}
            sm = _nr_group_average(px.E, px.total, smooth_rx, weight, groups, [1e10])
        avg = _nr_group_average(px.E, px.total, rx, weight, groups, s_grid)
        cap[it], fis[it], el[it] = avg["capture"], avg["fission"], avg["elastic"]
    # --- transfer matrices ---
    awr = nuc.awr
    P_asym = elastic_asymptotic_matrix(groups, awr, weight)
    if nuc.free_gas:
        sig_free = float(_potential(np.array([1.0]), nuc)[0])
        P_list = []
        for it, T in enumerate(T_grid):
            P = P_asym.copy()
            rows, sig_th, th = free_gas_group_matrix(groups, awr, float(T), weight, sig_free,
                                                     cutoff=THERMAL_CUTOFF_EV)
            P[th] = rows
            el_T = el[it]                      # view, shape (nS, G)
            el_T[:, th] = sig_th[None, :]
            P_list.append(P)
        P_el = np.array(P_list)
        P_el_T = np.asarray(T_grid, float)
    else:
        P_el = P_asym[None]
        P_el_T = np.array([float(T_grid[0])])
    P_in = evaporation_matrix(groups, awr, nuc.inelastic, weight)
    P_n2n = evaporation_matrix(groups, awr, nuc.n2n, weight)
    # --- fission data ---
    Emean = np.zeros(G)
    for g in range(G):
        u = np.linspace(np.log(groups.lower[g]), np.log(groups.upper[g]), 64)
        Eg = np.exp(u)
        w = weight(Eg) * Eg
        Emean[g] = np.trapezoid(w * Eg, u) / np.trapezoid(w, u)
    if nuc.fissionable:
        nu = nuc.nu[0] + nuc.nu[1] * Emean / 1.0e6
        a, b = nuc.watt
        chi = group_spectrum(groups, lambda E: watt_spectrum(E, a, b))
    else:
        nu = np.zeros(G)
        chi = np.zeros(G)
    mubar = np.full(G, 2.0 / (3.0 * awr))
    return NuclideMG(nuc.name, awr, np.asarray(T_grid, float), s_grid, cap, fis, el,
                     sm["nalpha"][0], sm["inelastic"][0], sm["n2n"][0], sm["tritium"][0],
                     P_el, P_el_T, P_in, P_n2n, nu, chi, nuc.kappa, mubar, resonant)


@dataclass
class Library:
    groups: GroupStructure
    nuclides: dict = field(default_factory=dict)
    invv: np.ndarray = None          # group-averaged 1/v (s/cm)
    chi_delayed: np.ndarray = None
    E_mean: np.ndarray = None
    meta: dict = field(default_factory=dict)

    def __getitem__(self, name: str) -> NuclideMG:
        return self.nuclides[name]

    def __contains__(self, name: str) -> bool:
        return name in self.nuclides

    @property
    def G(self) -> int:
        return self.groups.G

    def add_external(self, mg: NuclideMG):
        """Register externally generated group constants (same group structure)."""
        if mg.capture.shape[-1] != self.G:
            raise ValueError("External data must use the library group structure")
        self.nuclides[mg.name] = mg


def _database_hash() -> str:
    h = hashlib.md5()
    h.update(LIBRARY_VERSION.encode())
    for name in sorted(NUCLIDES):
        h.update(repr(NUCLIDES[name]).encode())
    with open(__file__, "rb") as f:
        h.update(f.read())
    for mod in ("pointwise.py", "kernels.py", "energy.py"):
        with open(os.path.join(os.path.dirname(__file__), mod), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def cache_dir() -> str:
    d = os.environ.get("MSRNX_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "msrnx"))
    os.makedirs(d, exist_ok=True)
    return d


_LIBRARY_SINGLETON: Library | None = None


def _build_one(name):
    return build_nuclide(NUCLIDES[name])


def build_library(names=None, n_jobs: int | None = None, verbose: bool = True) -> Library:
    names = sorted(NUCLIDES) if names is None else list(names)
    groups = FINE45
    n_jobs = n_jobs or min(4, os.cpu_count() or 1)
    if verbose:
        print(f"[msrnx] generating {groups.G}-group library for {len(names)} nuclides "
              f"({n_jobs} processes) ...", flush=True)
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            mgs = list(ex.map(_build_one, names))
    else:
        mgs = [_build_one(n) for n in names]
    lib = Library(groups, {m.name: m for m in mgs})
    # library-level data
    G = groups.G
    invv = np.zeros(G)
    Emean = np.zeros(G)
    for g in range(G):
        u = np.linspace(np.log(groups.lower[g]), np.log(groups.upper[g]), 64)
        E = np.exp(u)
        w = weighting_spectrum(E) * E
        invv[g] = np.trapezoid(w / neutron_speed_cm_s(E), u) / np.trapezoid(w, u)
        Emean[g] = np.trapezoid(w * E, u) / np.trapezoid(w, u)
    lib.invv = invv
    lib.E_mean = Emean
    theta = 0.3e6
    lib.chi_delayed = group_spectrum(groups, lambda E: np.sqrt(E) * np.exp(-E / theta))
    lib.meta = {"version": LIBRARY_VERSION, "hash": _database_hash()}
    return lib


def load_library(rebuild: bool = False, verbose: bool = True) -> Library:
    """Return the (cached) default library."""
    global _LIBRARY_SINGLETON
    if _LIBRARY_SINGLETON is not None and not rebuild:
        return _LIBRARY_SINGLETON
    key = _database_hash()
    path = os.path.join(cache_dir(), f"library_{key}.pkl")
    if os.path.exists(path) and not rebuild:
        with open(path, "rb") as f:
            _LIBRARY_SINGLETON = pickle.load(f)
        return _LIBRARY_SINGLETON
    lib = build_library(verbose=verbose)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(lib, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    if verbose:
        print(f"[msrnx] library cached at {path}", flush=True)
    _LIBRARY_SINGLETON = lib
    return lib
