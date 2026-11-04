"""Reactivity analyses: temperature / density / void coefficients, control-rod
worth, criticality searches and moderation (salt-fraction) studies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.optimize import brentq

from .core import MSRDesign, solve_core
from .lattice import CellGeometry, run_lattice
from .materials import graphite, salt_at_temperature


def rho(k):
    return (np.asarray(k, float) - 1.0) / np.asarray(k, float)


def drho_pcm(k1, k2):
    """Reactivity difference rho(k2) - rho(k1) in pcm."""
    return (1.0 / k1 - 1.0 / k2) * 1e5


@dataclass
class CoefficientResult:
    T_ref_fuel: float
    T_ref_graphite: float
    dT: float
    k_ref: float
    coefficients: dict = field(default_factory=dict)   # pcm/K
    k_values: dict = field(default_factory=dict)

    def table(self) -> list:
        return [(name, val) for name, val in self.coefficients.items()]


def _k(design, level, few_groups, mesh_size, **kw):
    if level == "lattice":
        from .lattice import fundamental_mode, critical_buckling
        salt = kw.get("salt") or design.fuel_salt(kw.get("T_fuel"))
        z = design.zones[0]
        Tg = kw.get("T_graphite", design.T_graphite)
        mod = graphite(design.graphite_density, Tg, design.graphite_boron_ppm)
        r = run_lattice(salt, mod if z.salt_fraction < 0.999 else None,
                        CellGeometry(z.channel_radius, z.salt_fraction, z.lattice),
                        perturb=kw.get("perturb"))
        B2 = kw.get("B2", (2.405 / (design.core_radius + 0.5 * design.radial_reflector)) ** 2
                    + (np.pi / (design.core_height + 30.0)) ** 2)
        return fundamental_mode(r.hom, B2)[0]
    res = solve_core(design, few_groups=few_groups, mesh_size=mesh_size, adjoint=False,
                     n_modes=1, **kw)
    return res.k_eff


def temperature_coefficients(design: MSRDesign, dT: float = 50.0, level: str = "core",
                             few_groups: str = "6g", mesh_size: float = 8.0,
                             components: bool = True) -> CoefficientResult:
    """Central-difference reactivity coefficients (pcm/K).

    * ``fuel_total``     fuel-salt temperature (Doppler + density/expansion)
    * ``fuel_doppler``   cross-section temperature only (density frozen)
    * ``fuel_density``   salt density only (cross sections frozen)
    * ``graphite``       moderator temperature (thermal spectrum shift)
    * ``isothermal``     fuel and graphite together
    """
    Tf, Tg = design.T_fuel, design.T_graphite
    salt0 = design.fuel_salt(Tf)
    out = CoefficientResult(Tf, Tg, dT, 0.0)
    kv = {}
    kv["ref"] = _k(design, level, few_groups, mesh_size, salt=salt0, T_fuel=Tf, T_graphite=Tg)
    out.k_ref = kv["ref"]
    for sgn in (+1, -1):
        T = Tf + sgn * dT
        kv[f"fuel{sgn:+d}"] = _k(design, level, few_groups, mesh_size,
                                 salt=salt_at_temperature(salt0, T), T_fuel=T, T_graphite=Tg)
        kv[f"graphite{sgn:+d}"] = _k(design, level, few_groups, mesh_size, salt=salt0,
                                     T_fuel=Tf, T_graphite=Tg + sgn * dT)
        kv[f"iso{sgn:+d}"] = _k(design, level, few_groups, mesh_size,
                                salt=salt_at_temperature(salt0, T), T_fuel=T,
                                T_graphite=Tg + sgn * dT)
        if components:
            s_dop = salt0.with_temperature(T)                 # XS at T, density at Tf
            kv[f"doppler{sgn:+d}"] = _k(design, level, few_groups, mesh_size, salt=s_dop,
                                        T_fuel=T, T_graphite=Tg)
            s_den = salt_at_temperature(salt0, T).with_temperature(Tf)  # density at T, XS at Tf
            kv[f"density{sgn:+d}"] = _k(design, level, few_groups, mesh_size, salt=s_den,
                                        T_fuel=Tf, T_graphite=Tg)

    def coef(key):
        return drho_pcm(kv[f"{key}-1"], kv[f"{key}+1"]) / (2 * dT)

    out.coefficients["fuel_total"] = coef("fuel")
    if components:
        out.coefficients["fuel_doppler"] = coef("doppler")
        out.coefficients["fuel_density"] = coef("density")
    out.coefficients["graphite"] = coef("graphite")
    out.coefficients["isothermal"] = coef("iso")
    out.k_values = kv
    return out


def temperature_scan(design: MSRDesign, temperatures, which: str = "fuel", level: str = "core",
                     few_groups: str = "6g", mesh_size: float = 8.0) -> np.ndarray:
    """k_eff versus fuel / graphite / isothermal temperature."""
    salt0 = design.fuel_salt(design.T_fuel)
    ks = []
    for T in temperatures:
        if which == "fuel":
            kw = dict(salt=salt_at_temperature(salt0, T), T_fuel=T, T_graphite=design.T_graphite)
        elif which == "graphite":
            kw = dict(salt=salt0, T_fuel=design.T_fuel, T_graphite=T)
        else:
            kw = dict(salt=salt_at_temperature(salt0, T), T_fuel=T, T_graphite=T + (
                design.T_graphite - design.T_fuel))
        ks.append(_k(design, level, few_groups, mesh_size, **kw))
    return np.array(ks)


def void_coefficient(design: MSRDesign, voids=(0.0, 0.01, 0.02, 0.03, 0.05),
                     few_groups: str = "6g", mesh_size: float = 8.0) -> dict:
    """k_eff versus helium void fraction in the fuel salt and slope (pcm/% void)."""
    ks = []
    for v in voids:
        d = design.copy(void_fraction=v)
        ks.append(_k(d, "core", few_groups, mesh_size))
    ks = np.array(ks)
    rhos = rho(ks) * 1e5
    slope = np.polyfit(np.asarray(voids) * 100.0, rhos, 1)[0]
    return {"void": np.asarray(voids), "k": ks, "rho_pcm": rhos, "pcm_per_percent_void": slope}


# ---------------------------------------------------------------------------
def rod_worth_curve(design: MSRDesign, insertions=None, rod=None, few_groups: str = "6g",
                    mesh_size: float = 6.0) -> dict:
    """Integral and differential worth of a central control rod (S-curve)."""
    from .core import ControlRod
    rod = rod or design.control_rod or ControlRod()
    H = design.core_height
    if insertions is None:
        insertions = np.linspace(0.0, H, 9)
    ks = []
    for z in insertions:
        r = ControlRod(rod.radius, float(z), rod.absorber_fraction, rod.b10_fraction)
        ks.append(_k(design.copy(control_rod=r), "core", few_groups, mesh_size))
    ks = np.array(ks)
    worth = (1.0 / ks[0] - 1.0 / ks) * 1e5      # pcm, negative = worth
    zmid = 0.5 * (np.asarray(insertions)[1:] + np.asarray(insertions)[:-1])
    diff = np.diff(worth) / np.diff(insertions)
    return {"insertion_cm": np.asarray(insertions), "k": ks, "worth_pcm": worth,
            "z_mid": zmid, "differential_pcm_per_cm": diff, "total_worth_pcm": worth[-1]}


def shutdown_margin(design: MSRDesign, rod_worths_pcm, stuck_rod: bool = True) -> dict:
    """Shutdown margin with the most reactive rod stuck out."""
    k = _k(design, "core", "6g", 8.0)
    excess = rho(k) * 1e5
    worths = sorted(np.abs(np.asarray(rod_worths_pcm, float)))
    available = sum(worths[:-1]) if stuck_rod and len(worths) > 1 else sum(worths)
    return {"excess_pcm": excess, "available_worth_pcm": available,
            "shutdown_margin_pcm": available - excess}


# ---------------------------------------------------------------------------
def set_fissile_fraction(design: MSRDesign, x: float, fissile_compound: str = "UF4",
                         balance_compound: str = "LiF") -> MSRDesign:
    comp = dict(design.salt_composition)
    delta = x - comp.get(fissile_compound, 0.0)
    comp[fissile_compound] = x
    comp[balance_compound] = comp[balance_compound] - delta
    return design.copy(salt_composition=comp)


def criticality_search(design: MSRDesign, setter: Callable, lo: float, hi: float,
                       target_k: float = 1.0, level: str = "core", few_groups: str = "6g",
                       mesh_size: float = 8.0, xtol: float = 1e-6) -> dict:
    """Find the parameter value giving k_eff = target via Brent's method.

    ``setter(design, value) -> design``.  Returns the root and the search
    history for plotting.
    """
    hist = []

    def f(x):
        k = _k(setter(design, x), level, few_groups, mesh_size)
        hist.append((x, k))
        return k - target_k

    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise ValueError(f"Target k={target_k} not bracketed: k({lo})={flo + target_k:.5f}, "
                         f"k({hi})={fhi + target_k:.5f}")
    x = brentq(f, lo, hi, xtol=xtol)
    hist.sort()
    return {"value": x, "history": np.array(hist), "design": setter(design, x)}


def moderation_curve(design: MSRDesign, salt_fractions=None, channel_radius=None) -> dict:
    """k_inf of the lattice cell versus fuel-salt volume fraction."""
    if salt_fractions is None:
        salt_fractions = np.linspace(0.04, 0.6, 25)
    salt = design.fuel_salt()
    mod = design.moderator()
    z = design.zones[0]
    r = channel_radius or z.channel_radius
    kinf, cr, p, f = [], [], [], []
    from .lattice import conversion_ratio
    for e in salt_fractions:
        res = run_lattice(salt, mod, CellGeometry(r, float(e), z.lattice))
        kinf.append(res.k_inf)
        cr.append(conversion_ratio(res.fuel_macro, res.phi_fuel * res.V_fuel))
        p.append(res.six_factor["p"])
        f.append(res.six_factor["f"])
    kinf = np.array(kinf)
    return {"salt_fraction": np.asarray(salt_fractions), "k_inf": kinf,
            "conversion_ratio": np.array(cr), "p": np.array(p), "f": np.array(f),
            "optimum_salt_fraction": float(np.asarray(salt_fractions)[np.argmax(kinf)])}
