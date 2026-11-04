"""Coupled transport-depletion of a circulating, well-mixed MSR fuel salt.

* Transport models: full 2D core sequence (``CoreTransport``) or a fast,
  2D-calibrated lattice/B1 model (``LatticeTransport``).
* CE/CM predictor-corrector with CRAM-16 matrix exponentials.
* Online reprocessing (removal terms in the burnup matrix) and criticality-
  controlled refuelling: fissile feed (or removal) at every step to hold the
  target k_eff, plus fertile make-up to keep the fertile inventory constant.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..constants import AVOGADRO, MEV_TO_J, SECONDS_PER_DAY
from ..data.nuclides import NUCLIDES
from ..lattice import CellGeometry, critical_buckling, fundamental_mode, run_lattice
from ..macro import collapse_micro
from ..materials import Material, graphite
from ..xs.library import load_library
from .chain import DepletionChain, Reprocessing
from .cram import augment_with_feed, cram16

FISSILE = ("U233", "U235", "Pu239", "Pu241")


# ---------------------------------------------------------------------------
# Transport models
# ---------------------------------------------------------------------------
class CoreTransport:
    """Full lattice -> spectral-zone -> 2D R-Z evaluation for each salt state."""

    def __init__(self, design, few_groups: str = "6g", mesh_size: float = 10.0, lib=None):
        self.design, self.fg, self.h = design, few_groups, mesh_size
        self.lib = lib or load_library()
        self.calls = 0

    def evaluate(self, salt: Material, power_W: float):
        from ..core import solve_core
        self.calls += 1
        d = self.design.copy(power_MW=power_W / 1e6)
        r = solve_core(d, self.fg, self.h, lib=self.lib, salt=salt, adjoint=False, n_modes=1)
        return r.one_group, r.salt_flux_avg, r.k_eff


class LatticeTransport:
    """Zone-1 lattice + B1 leakage with fixed buckling, optionally calibrated
    against the 2D core at the initial state (k and one-group cross sections)."""

    def __init__(self, design, calibrate: bool = True, few_groups: str = "6g",
                 mesh_size: float = 10.0, lib=None):
        self.design = design
        self.lib = lib or load_library()
        z = design.zones[0]
        self.cell = CellGeometry(z.channel_radius, z.salt_fraction, z.lattice)
        self.mod = graphite(design.graphite_density, design.T_graphite, design.graphite_boron_ppm)
        self.B2 = ((2.405 / (design.core_radius + 0.5 * design.radial_reflector)) ** 2
                   + (np.pi / (design.core_height + 30.0)) ** 2)
        self.corr = {}
        self.calls = 0
        self.V_total = design.salt_volumes_m3()["total"] * 1e6
        if calibrate:
            from ..core import solve_core
            salt = design.fuel_salt()
            r2 = solve_core(design, few_groups, mesh_size, lib=self.lib, adjoint=False, n_modes=1)
            lat = self._lattice(salt)
            # buckling giving the 2D k_eff
            from scipy.optimize import brentq
            f = lambda b2: fundamental_mode(lat.hom, b2)[0] - r2.k_eff
            try:
                self.B2 = brentq(f, -0.5 * critical_buckling(lat.hom) - 1e-3, 0.05)
            except ValueError:
                pass
            og0, _, _ = self._raw(salt, design.power_MW * 1e6, lat)
            for nuc, d2 in r2.one_group.items():
                d0 = og0.get(nuc)
                if d0 is None:
                    continue
                self.corr[nuc] = {rx: (d2[rx] / d0[rx] if d0[rx] > 0 else 1.0) for rx in d2}
            self.k2d0 = r2.k_eff

    def _lattice(self, salt):
        return run_lattice(salt, self.mod if self.cell.salt_fraction < 0.999 else None,
                           self.cell, self.lib)

    def _raw(self, salt, power_W, lat=None):
        lat = lat or self._lattice(salt)
        k, phi_cell, _ = fundamental_mode(lat.hom, self.B2)
        phi_fuel = phi_cell * lat.fuel_flux_ratio
        micro = dict(lat.fuel_macro.micro)
        for nuc in self.lib.nuclides:
            if nuc not in micro:
                micro[nuc] = self.lib[nuc].micro(salt.T, 1e10)
        og = collapse_micro(micro, phi_fuel)
        # flux normalization: P = sum N kappa sigma_f * phi_inv * V_total
        kf = sum(salt.N.get(n, 0.0) * self.lib[n].kappa * og[n]["fission"] for n in og
                 if og[n]["fission"] > 0)          # MeV/cm per unit flux
        phi_inv = power_W / (kf * MEV_TO_J * self.V_total)
        return og, phi_inv, k

    def evaluate(self, salt: Material, power_W: float):
        self.calls += 1
        og, phi, k = self._raw(salt, power_W)
        if self.corr:
            og = {n: {rx: v * self.corr.get(n, {}).get(rx, 1.0) for rx, v in d.items()}
                  for n, d in og.items()}
            kf = sum(salt.N.get(n, 0.0) * self.lib[n].kappa * og[n]["fission"] for n in og
                     if og[n]["fission"] > 0)
            phi = power_W / (kf * MEV_TO_J * self.V_total)
        return og, phi, k


# ---------------------------------------------------------------------------
@dataclass
class FeedControl:
    target_k: float = 1.0
    fissile_feed: dict = field(default_factory=lambda: {"U233": 1.0})
    fertile_makeup: tuple = ("Th232",)
    allow_removal: bool = True
    tolerance_pcm: float = 20.0
    max_iter: int = 4


@dataclass
class DepletionResult:
    time_days: np.ndarray
    k_bos: np.ndarray                # before refuelling at each step start
    k_eff: np.ndarray                # after refuelling (operating)
    N: np.ndarray                    # (n_times, n_nuclides) atoms/b-cm
    names: list
    flux: np.ndarray
    breeding_ratio: np.ndarray
    fissile_production_ratio: np.ndarray
    fissile_feed_kg: np.ndarray      # cumulative
    fertile_feed_kg: np.ndarray      # cumulative
    burnup_MWd_kgHM: np.ndarray
    power_MW: float
    V_total_cm3: float
    poison_fraction: dict
    tritium_rate: np.ndarray
    removed_kg: dict
    wall_time_s: float
    transport_calls: int
    reprocessing: str

    def density(self, nuc: str) -> np.ndarray:
        return self.N[:, self.names.index(nuc)]

    def inventory_kg(self, nuc: str) -> np.ndarray:
        m = NUCLIDES[nuc].mass if nuc in NUCLIDES else NUCLIDES["Pa233"].mass
        return self.density(nuc) * 1e24 * self.V_total_cm3 * m / AVOGADRO / 1000.0

    def fissile_inventory_kg(self) -> np.ndarray:
        return sum(self.inventory_kg(n) for n in FISSILE if n in self.names)


def _mass_kg(n_atoms_per_bcm: float, nuc: str, V_cm3: float) -> float:
    return n_atoms_per_bcm * 1e24 * V_cm3 * NUCLIDES[nuc].mass / AVOGADRO / 1000.0


class DepletionSolver:
    def __init__(self, design, transport: str = "lattice", reprocessing: Optional[Reprocessing] = None,
                 feed: Optional[FeedControl] = None, few_groups: str = "6g",
                 mesh_size: float = 10.0, lib=None, transport_model=None):
        self.design = design
        self.lib = lib or load_library()
        if transport_model is not None:
            self.tm = transport_model
        elif transport == "core":
            self.tm = CoreTransport(design, few_groups, mesh_size, self.lib)
        else:
            self.tm = LatticeTransport(design, True, few_groups, mesh_size, self.lib)
        self.chain = DepletionChain(list(self.lib.nuclides.keys()), reprocessing)
        self.feed = feed
        self.V_total = design.salt_volumes_m3()["total"] * 1e6
        self.salt0 = design.fuel_salt()

    def _salt(self, vec) -> Material:
        m = self.salt0.copy()
        m.N = self.chain.salt_dict(np.maximum(vec, 0.0))
        return m

    def _refuel(self, vec, k, P):
        """Adjust fissile content to reach the target k (secant on feed amount)."""
        fc = self.feed
        idx = self.chain.index
        fv = np.zeros(self.chain.n)
        for n, f in fc.fissile_feed.items():
            fv[idx[n]] = f
        fis_now = sum(vec[idx[n]] for n in FISSILE)
        added = 0.0
        if abs(k - fc.target_k) * 1e5 < fc.tolerance_pcm:
            return vec, k, added
        delta = 0.01 * max(fis_now, 1e-8)
        for _ in range(fc.max_iter):
            trial = vec + delta * fv
            _, _, k2 = self.tm.evaluate(self._salt(trial), P)
            slope = (k2 - k) / delta
            if slope <= 0:
                break
            dN = (fc.target_k - k) / slope
            if dN < 0 and not fc.allow_removal:
                break
            dN = max(dN, -0.9 * fis_now)
            vec = np.maximum(vec + dN * fv, 0.0)
            added += dN
            _, _, k = self.tm.evaluate(self._salt(vec), P)
            if abs(k - fc.target_k) * 1e5 < fc.tolerance_pcm:
                break
            delta = max(abs(dN) * 0.5, 1e-3 * fis_now)
        return vec, k, added

    def run(self, steps_days, power_MW: Optional[float] = None, verbose: bool = False) -> DepletionResult:
        t0 = time.time()
        P = (power_MW or self.design.power_MW) * 1e6
        ch = self.chain
        vec = ch.vector(self.salt0.N)
        idx = ch.index
        fert0 = {n: vec[idx[n]] for n in (self.feed.fertile_makeup if self.feed else ())
                 if n in idx}
        hm0_kg = sum(_mass_kg(vec[idx[n]], n, self.V_total) for n in ch.salt_nuclides
                     if NUCLIDES[n].is_actinide)
        times = [0.0]
        Ns, kb, ka, fl, br, fpr, trit = [], [], [], [], [], [], []
        fis_feed, fert_feed = [0.0], [0.0]
        pf = {n: [] for n in ("Xe135", "Sm149", "Pa233", "LFP_RE", "LFP_NM", "LFP_SS", "LFP_NG")}
        t_now = 0.0
        steps = list(steps_days)
        og, phi, k = self.tm.evaluate(self._salt(vec), P)
        for istep in range(len(steps) + 1):
            # --- refuelling / fertile make-up at the step start ---
            k_before = k
            mass_fis = 0.0
            mass_fert = 0.0
            if self.feed is not None:
                for n, N0 in fert0.items():
                    dN = N0 - vec[idx[n]]
                    if dN > 0:
                        vec[idx[n]] = N0
                        mass_fert += _mass_kg(dN, n, self.V_total)
                if fert0:
                    og, phi, k = self.tm.evaluate(self._salt(vec), P)
                vec, k, added = self._refuel(vec, k, P)
                if added != 0.0:
                    og, phi, k = self.tm.evaluate(self._salt(vec), P)
                for n, f in self.feed.fissile_feed.items():
                    mass_fis += _mass_kg(added * f, n, self.V_total)
            A = ch.matrix(og, phi)
            flows = ch.fissile_flows(A, vec, og, phi)
            Ns.append(vec.copy())
            kb.append(k_before)
            ka.append(k)
            fl.append(phi)
            br.append(flows["breeding_ratio"])
            fpr.append(flows["fissile_production_ratio"])
            if istep > 0:
                fis_feed.append(fis_feed[-1] + mass_fis)
                fert_feed.append(fert_feed[-1] + mass_fert)
            else:
                fis_feed[0] = mass_fis
                fert_feed[0] = mass_fert
            tot_abs = sum(vec[idx[n]] * og[n]["absorption"] for n in ch.salt_nuclides if n in og)
            for n in pf:
                pf[n].append(vec[idx[n]] * og[n]["absorption"] / tot_abs if tot_abs > 0 else 0.0)
            li6 = vec[idx["Li6"]] * og["Li6"]["nalpha"] + vec[idx["Li7"]] * og["Li7"]["tritium"]
            trit.append(li6 * phi * self.V_total)
            if verbose:
                print(f"  t={t_now:8.1f} d  k_bos={k_before:.5f} k={k:.5f} BR={flows['breeding_ratio']:.4f}")
            if istep == len(steps):
                break
            dt = steps[istep] * SECONDS_PER_DAY
            # --- CE/CM predictor-corrector ---
            N_mid = cram16(A, vec, 0.5 * dt)
            og_m, phi_m, _ = self.tm.evaluate(self._salt(N_mid), P)
            A_m = ch.matrix(og_m, phi_m)
            vec = np.maximum(cram16(A_m, vec, dt), 0.0)
            t_now += steps[istep]
            times.append(t_now)
            og, phi, k = self.tm.evaluate(self._salt(vec), P)
        Ns = np.array(Ns)
        t = np.array(times)
        bu = P / 1e6 * t / max(hm0_kg, 1e-30)
        removed = {}
        molar = {"noble_gas": 133.0, "noble_metal": 101.0, "rare_earth": 145.0,
                 "salt_seeker": 110.0, "halogen": 131.0, "tritium": 3.0, "Pa": 233.0,
                 "U": 235.0, "Pu": 239.0, "Np": 237.0, "Am": 241.0, "Th": 232.0}
        for c in [n for n in ch.names if n.startswith("REMOVED_")]:
            cls = c.replace("REMOVED_", "")
            removed[cls] = Ns[:, idx[c]] * 1e24 * self.V_total * molar.get(cls, 130.0) / AVOGADRO / 1000.0
        return DepletionResult(t, np.array(kb), np.array(ka), Ns, ch.names, np.array(fl),
                               np.array(br), np.array(fpr), np.array(fis_feed), np.array(fert_feed), bu,
                               P / 1e6, self.V_total, {n: np.array(v) for n, v in pf.items()},
                               np.array(trit), removed, time.time() - t0,
                               getattr(self.tm, "calls", 0), ch.repro.name)
