"""Xenon-135 and samarium-149 poisoning with MSR off-gas sparging and
rare-earth removal: equilibrium values, worths and transients."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .data.fission_yields import yields_for
from .data.nuclides import NUCLIDES

BARN = 1e-24


@dataclass
class PoisonParams:
    fission_rate: float        # fissions / cm3 / s (salt-inventory average)
    flux: float                # n / cm2 / s (salt-inventory average)
    production_rate: float     # neutrons / cm3 / s (nu * fission rate)
    sigma_Xe: float            # cm2 (one-group, salt-averaged)
    sigma_Sm: float
    sigma_Pm: float
    gamma_I: float
    gamma_Xe: float
    gamma_Pm: float
    lam_I: float = NUCLIDES["I135"].decay_constant
    lam_Xe: float = NUCLIDES["Xe135"].decay_constant
    lam_Pm: float = NUCLIDES["Pm149"].decay_constant
    lam_sparge: float = 0.0    # Xe removal by He sparging (1/s)
    lam_RE: float = 0.0        # rare-earth (Pm/Sm) chemical removal (1/s)
    worth_scale: float = 1.0   # direct-transport / first-order worth ratio


def poison_params_from_core(core, salt=None, lam_sparge: float = 0.0,
                            lam_RE: float = 0.0) -> PoisonParams:
    """One-group poison parameters from a solved :class:`CoreResult`."""
    salt = salt or core.design.fuel_salt()
    og = core.one_group
    phi = core.salt_flux_avg
    fis = {n: salt.N.get(n, 0.0) * og[n]["fission"] for n in og if og[n]["fission"] > 0}
    F = sum(fis.values())
    nuF = sum(salt.N.get(n, 0.0) * og[n]["nu_fission"] for n in og)
    g = {"I135": 0.0, "Xe135": 0.0, "Pm149": 0.0}
    for n, f in fis.items():
        y = yields_for(n)
        for k in g:
            g[k] += y[k] * f / F
    sXe = og.get("Xe135", {}).get("absorption", 2.0e6) * BARN
    sSm = og.get("Sm149", {}).get("absorption", 4.0e4) * BARN
    sPm = og.get("Pm149", {}).get("absorption", 1.0e3) * BARN
    return PoisonParams(F * phi, phi, nuF * phi, sXe, sSm, sPm, g["I135"], g["Xe135"],
                        g["Pm149"], lam_sparge=lam_sparge, lam_RE=lam_RE)


def equilibrium(p: PoisonParams, power_fraction: float = 1.0) -> dict:
    F = p.fission_rate * power_fraction
    phi = p.flux * power_fraction
    I = p.gamma_I * F / p.lam_I
    Xe = (p.gamma_Xe * F + p.lam_I * I) / (p.lam_Xe + p.sigma_Xe * phi + p.lam_sparge)
    Pm = p.gamma_Pm * F / (p.lam_Pm + p.sigma_Pm * phi + p.lam_RE)
    Sm = p.lam_Pm * Pm / (p.sigma_Sm * phi + p.lam_RE) if (p.sigma_Sm * phi + p.lam_RE) > 0 else 0.0
    return {"I135": I, "Xe135": Xe, "Pm149": Pm, "Sm149": Sm,
            "rho_Xe_pcm": worth_estimate(p, Xe, "Xe"), "rho_Sm_pcm": worth_estimate(p, Sm, "Sm")}


def worth_estimate(p: PoisonParams, N_cm3, which: str = "Xe") -> float:
    """First-order worth  d_rho = - (absorption rate) / (production rate)  in pcm."""
    sig = p.sigma_Xe if which == "Xe" else p.sigma_Sm
    return -np.asarray(N_cm3) * sig * p.flux / p.production_rate * 1e5 * p.worth_scale


def transient(p: PoisonParams, t_end: float, power: Callable[[float], float],
              y0: Optional[dict] = None, n_points: int = 600) -> dict:
    """I/Xe/Pm/Sm evolution for a power history ``power(t)`` (fraction of nominal)."""
    if y0 is None:
        y0 = equilibrium(p, power(0.0))
    Y0 = [y0["I135"], y0["Xe135"], y0["Pm149"], y0["Sm149"]]

    def rhs(t, y):
        I, X, Pm, Sm = y
        f = power(t)
        F, phi = p.fission_rate * f, p.flux * f
        return [p.gamma_I * F - p.lam_I * I,
                p.gamma_Xe * F + p.lam_I * I - (p.lam_Xe + p.sigma_Xe * phi + p.lam_sparge) * X,
                p.gamma_Pm * F - (p.lam_Pm + p.sigma_Pm * phi + p.lam_RE) * Pm,
                p.lam_Pm * Pm - (p.sigma_Sm * phi + p.lam_RE) * Sm]

    t = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(rhs, (0.0, t_end), Y0, t_eval=t, method="LSODA", rtol=1e-8, atol=1e-2,
                    max_step=t_end / 300)
    I, X, Pm, Sm = sol.y
    # worth evaluated with the *instantaneous* production rate held at nominal (rods hold k)
    return {"t": sol.t, "I135": I, "Xe135": X, "Pm149": Pm, "Sm149": Sm,
            "rho_Xe_pcm": worth_estimate(p, X, "Xe"), "rho_Sm_pcm": worth_estimate(p, Sm, "Sm"),
            "power": np.array([power(x) for x in sol.t])}


def sparging_study(p: PoisonParams, removal_rates) -> dict:
    """Equilibrium xenon worth versus sparging removal constant."""
    out = []
    for lam in removal_rates:
        out.append(equilibrium(replace(p, lam_sparge=float(lam)))["rho_Xe_pcm"])
    return {"lam_sparge": np.asarray(removal_rates), "rho_Xe_pcm": np.array(out)}


def direct_worth(design, nuclide: str, N_atoms_per_bcm: float, few_groups: str = "6g",
                 mesh_size: float = 8.0) -> float:
    """Exact (within the model) reactivity worth of adding a poison to the salt (pcm)."""
    from .core import solve_core
    salt = design.fuel_salt()
    k0 = solve_core(design, few_groups, mesh_size, salt=salt, adjoint=False, n_modes=1).k_eff
    s2 = salt.copy()
    s2.N[nuclide] = s2.N.get(nuclide, 0.0) + N_atoms_per_bcm
    k1 = solve_core(design, few_groups, mesh_size, salt=s2, adjoint=False, n_modes=1).k_eff
    return (1.0 / k0 - 1.0 / k1) * 1e5


def calibrate_worth(p: PoisonParams, design, few_groups: str = "6g",
                    mesh_size: float = 8.0) -> PoisonParams:
    """Scale the first-order worth so it reproduces the direct transport worth of
    equilibrium xenon (captures importance/spectral weighting)."""
    eq = equilibrium(replace(p, worth_scale=1.0))
    direct = direct_worth(design, "Xe135", eq["Xe135"] * 1e-24, few_groups, mesh_size)
    return replace(p, worth_scale=direct / eq["rho_Xe_pcm"])
