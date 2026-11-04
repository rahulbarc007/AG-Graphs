"""Energy-transfer kernels and their group-to-group collapse.

* Asymptotic elastic kernel (target at rest, isotropic CM scattering):
  E' uniform in [alpha E, E].
* Free-gas thermal kernel (Maxwellian target at temperature T, isotropic CM
  scattering, constant free-atom cross section), evaluated in a numerically
  stable form with the scaled complementary error function.
* Evaporation spectrum for inelastic and (n,2n) secondary neutrons.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf, erfcx

from ..constants import K_BOLTZMANN_EV
from .energy import GroupStructure


def free_gas_kernel(E_in, E_out, awr: float, T: float) -> np.ndarray:
    """sigma(E->E') / sigma_free in 1/eV for a free gas of mass ``awr``.

    ``E_in`` and ``E_out`` broadcast against each other.
    """
    kT = K_BOLTZMANN_EV * T
    A = awr
    eta = (A + 1.0) / (2.0 * np.sqrt(A))
    rho = (A - 1.0) / (2.0 * np.sqrt(A))
    e, ep = np.broadcast_arrays(np.asarray(E_in, float) / kT, np.asarray(E_out, float) / kT)
    se, sep = np.sqrt(e), np.sqrt(ep)
    out = np.zeros(e.shape)
    pref = eta ** 2 / (2.0 * e * kT)
    down = ep <= e
    up = ~down
    # --- down-scatter (stable via erfcx) ---
    if np.any(down):
        s, sp, pf = se[down], sep[down], pref[down]
        t1 = erf(eta * sp - rho * s) + erf(eta * sp + rho * s)
        b = eta * s - rho * sp
        c = eta * s + rho * sp
        t2 = (erfcx(c) * np.exp(-(eta * sp + rho * s) ** 2)
              - erfcx(b) * np.exp(-(eta * sp - rho * s) ** 2))
        out[down] = pf * (t1 + t2)
    # --- up-scatter (exp(e-e') <= 1, safe) ---
    if np.any(up):
        s, sp, pf = se[up], sep[up], pref[up]
        t1 = erf(eta * sp - rho * s) - erf(eta * sp + rho * s)
        t2 = np.exp(e[up] - ep[up]) * (erf(eta * s - rho * sp) + erf(eta * s + rho * sp))
        out[up] = pf * (t1 + t2)
    return np.maximum(out, 0.0)


def free_gas_total_factor(E, awr: float, T: float) -> np.ndarray:
    """Effective free-gas scattering cross section / free-atom cross section."""
    a = np.sqrt(awr * np.asarray(E, float) / (K_BOLTZMANN_EV * T))
    a = np.maximum(a, 1e-8)
    return (1.0 + 0.5 / a ** 2) * erf(a) + np.exp(-a ** 2) / (a * np.sqrt(np.pi))


def _segment_groups(grid: np.ndarray, groups: GroupStructure) -> np.ndarray:
    mids = np.sqrt(grid[:-1] * grid[1:])
    idx = np.searchsorted(-groups.bounds, -mids, side="right") - 1
    return np.clip(idx, 0, groups.G - 1)


def elastic_asymptotic_matrix(groups: GroupStructure, awr: float, weight,
                              n_quad: int = 48) -> np.ndarray:
    """Normalized P(g->g') for elastic scattering off a target at rest."""
    G = groups.G
    alpha = ((awr - 1.0) / (awr + 1.0)) ** 2
    lo, hi = groups.lower, groups.upper
    P = np.zeros((G, G))
    for g in range(G):
        u = np.linspace(np.log(lo[g]), np.log(hi[g]), n_quad + 1)
        uc = 0.5 * (u[1:] + u[:-1])
        E = np.exp(uc)
        w = weight(E) * E * np.diff(u)
        Elo = alpha * E
        # overlap of [alpha E, E] with each group, per incoming point
        ov = (np.minimum(E[:, None], hi[None, :]) - np.maximum(Elo[:, None], lo[None, :]))
        ov = np.maximum(ov, 0.0) / ((1.0 - alpha) * E[:, None])
        P[g] = (w[:, None] * ov).sum(axis=0) / w.sum()
    P /= P.sum(axis=1, keepdims=True)
    return P


def free_gas_group_matrix(groups: GroupStructure, awr: float, T: float, weight,
                          sigma_free: float, cutoff: float = 3.06, n_fine: int = 700):
    """Thermal free-gas group transfer data.

    Returns ``(P_thermal_rows, sigma_el_thermal, thermal_idx)`` where
    ``P_thermal_rows`` has shape (n_thermal, G) (row-normalized) and
    ``sigma_el_thermal`` is the group-averaged effective scattering XS.
    """
    th = groups.thermal_groups(cutoff)
    Emax_out = 25.0
    fine = np.unique(np.concatenate([
        np.geomspace(groups.bounds[-1], Emax_out, n_fine),
        groups.bounds[(groups.bounds >= groups.bounds[-1]) & (groups.bounds <= Emax_out)]]))
    seg_g = _segment_groups(fine, groups)
    dE = np.diff(fine)
    Etop = groups.upper[th].max()
    inc = fine[fine <= Etop * (1 + 1e-12)]
    # incoming point weights (trapezoid, per group)
    K = free_gas_kernel(inc[:, None], fine[None, :], awr, T)        # (n_in, n_fine)
    seg_int = 0.5 * (K[:, :-1] + K[:, 1:]) * dE[None, :]            # (n_in, n_seg)
    tot_num = seg_int.sum(axis=1)
    tot_ana = free_gas_total_factor(inc, awr, T)
    seg_int *= (tot_ana / np.maximum(tot_num, 1e-300))[:, None]
    # group-collapse outgoing
    out_g = np.zeros((len(inc), groups.G))
    np.add.at(out_g.T, seg_g, seg_int.T)
    # incoming group collapse with trapezoid weights
    w_in = np.zeros(len(inc))
    dEi = np.diff(inc)
    W = weight(inc)
    w_in[:-1] += 0.5 * dEi
    w_in[1:] += 0.5 * dEi
    gin = np.clip(np.searchsorted(-groups.bounds, -inc, side="right") - 1, 0, groups.G - 1)
    P_rows = np.zeros((len(th), groups.G))
    sig_th = np.zeros(len(th))
    for k, g in enumerate(th):
        m = gin == g
        # points exactly on the boundary belong to both groups: handle by segment midpoints
        if not np.any(m):
            continue
        ww = W[m] * w_in[m]
        rows = out_g[m]
        tot = (ww * tot_ana[m]).sum()
        P_rows[k] = (ww[:, None] * rows).sum(axis=0) / tot
        sig_th[k] = sigma_free * tot / ww.sum()
    P_rows /= np.maximum(P_rows.sum(axis=1, keepdims=True), 1e-300)
    return P_rows, sig_th, th


def evaporation_matrix(groups: GroupStructure, awr: float, spec, weight,
                       n_quad: int = 24) -> np.ndarray:
    """Normalized P(g->g') for an evaporation (Maxwellian) secondary spectrum."""
    G = groups.G
    P = np.zeros((G, G))
    if spec is None:
        return P
    Ethr, smax, Ew = spec
    lo, hi = groups.lower, groups.upper
    a_ld = max(awr / 10.0, 0.5)  # level density parameter (1/MeV)
    for g in range(G):
        if hi[g] <= Ethr:
            continue
        u = np.linspace(np.log(max(lo[g], Ethr * 1.0001)), np.log(hi[g]), n_quad + 1)
        uc = 0.5 * (u[1:] + u[:-1])
        E = np.exp(uc)
        sig = smax * (1.0 - np.exp(-(E - Ethr) / Ew))
        w = weight(E) * E * np.diff(u) * sig
        theta = 1.0e6 * np.sqrt((E / 1.0e6) / a_ld)
        Emax = E - Ethr

        def cdf(x, th):
            x = np.maximum(x, 0.0)
            return 1.0 - np.exp(-x / th) * (1.0 + x / th)

        norm = cdf(Emax, theta)
        frac = (cdf(np.minimum(hi[None, :], Emax[:, None]), theta[:, None])
                - cdf(np.minimum(lo[None, :], Emax[:, None]), theta[:, None]))
        frac = np.maximum(frac, 0.0) / np.maximum(norm[:, None], 1e-300)
        if w.sum() > 0:
            P[g] = (w[:, None] * frac).sum(axis=0) / w.sum()
    s = P.sum(axis=1, keepdims=True)
    P = np.where(s > 0, P / np.maximum(s, 1e-300), 0.0)
    return P
