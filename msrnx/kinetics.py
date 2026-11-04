"""Reactor kinetics for circulating-fuel molten salt reactors.

* Delayed-neutron precursor drift: effective delayed fractions with fuel
  circulation from (a) the analytic uniform slug-flow model and (b) an
  importance-weighted axial advection-decay model using the core's fission
  and adjoint shapes.
* Multi-node circulating-fuel point kinetics (plug-flow core nodes + loop
  nodes) with fuel-salt / graphite thermal feedback, time-dependent pump
  flow, external reactivity and heat-exchanger models.
* Inhour (period) analysis for static and circulating fuel, prompt jump and
  decay heat.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Precursor drift
# ---------------------------------------------------------------------------
def uniform_loss_factor(lam, tau_c: float, tau_L: float) -> np.ndarray:
    """Fraction of precursors decaying in-core, uniform slug flow (analytic)."""
    lam = np.asarray(lam, float)
    a = lam * tau_c
    b = lam * tau_L
    return 1.0 - (1.0 - np.exp(-a)) * (1.0 - np.exp(-b)) / (a * (1.0 - np.exp(-(a + b))))


def axial_loss_factor(lam, tau_c: float, tau_L: float, fission_shape=None,
                      importance_shape=None, n: int = 400) -> np.ndarray:
    """Importance-weighted effective delayed fraction ratio for slug flow.

    Fuel enters at the bottom (z=0) and leaves at the top (z=H).  Shapes are
    callables of the normalized height x in [0, 1] or (x, value) arrays;
    defaults are the bare-core sine shapes.
    """
    x = (np.arange(n) + 0.5) / n

    def _shape(s, default):
        if s is None:
            return default(x)
        if callable(s):
            return np.asarray(s(x), float)
        xs, vs = np.asarray(s[0], float), np.asarray(s[1], float)
        xs = (xs - xs.min()) / max(xs.max() - xs.min(), 1e-30)
        return np.interp(x, xs, vs)

    f = np.maximum(_shape(fission_shape, lambda t: np.sin(np.pi * t)), 0.0)
    w = np.maximum(_shape(importance_shape, lambda t: np.sin(np.pi * t)), 0.0)
    f = f / f.sum()
    dt = tau_c / n
    out = []
    for l in np.atleast_1d(lam):
        S = f / dt                      # production per unit residence time
        e = np.exp(-l * dt)
        # march C through the core for C_in = 0 to get C_out = a*C_in + b
        a = np.exp(-l * tau_c)
        Ck = np.zeros(n + 1)
        for k in range(n):
            Ck[k + 1] = Ck[k] * e + S[k] / l * (1.0 - e)
        b = Ck[-1]
        Cin = b * np.exp(-l * tau_L) / (1.0 - a * np.exp(-l * tau_L))
        Ck = np.zeros(n + 1)
        Ck[0] = Cin
        Cbar = np.zeros(n)
        for k in range(n):
            Ck[k + 1] = Ck[k] * e + S[k] / l * (1.0 - e)
            Cbar[k] = S[k] / l + (Ck[k] - S[k] / l) * (1.0 - e) / (l * dt)
        emitted = (l * Cbar * w * dt).sum()
        static = (S * w * dt).sum()
        out.append(emitted / static)
    return np.array(out)


@dataclass
class CirculationResult:
    beta_i_static: np.ndarray
    beta_i_circ: np.ndarray
    lam: np.ndarray
    tau_c: float
    tau_L: float
    model: str

    @property
    def beta_static(self) -> float:
        return float(self.beta_i_static.sum())

    @property
    def beta_circ(self) -> float:
        return float(self.beta_i_circ.sum())

    @property
    def loss_pcm(self) -> float:
        return (self.beta_static - self.beta_circ) * 1e5


def circulating_beta(beta_i, lam_i, tau_c: float, tau_L: float, model: str = "axial",
                     fission_shape=None, importance_shape=None) -> CirculationResult:
    beta_i = np.asarray(beta_i, float)
    lam_i = np.asarray(lam_i, float)
    if model == "uniform":
        F = uniform_loss_factor(lam_i, tau_c, tau_L)
    else:
        F = axial_loss_factor(lam_i, tau_c, tau_L, fission_shape, importance_shape)
    return CirculationResult(beta_i, beta_i * F, lam_i, tau_c, tau_L, model)


# ---------------------------------------------------------------------------
# Point kinetics with circulating fuel
# ---------------------------------------------------------------------------
@dataclass
class ThermalModel:
    """Two-node core thermal model + advected loop with a heat exchanger."""
    power_W: float = 250e6
    T_in0: float = 850.0             # K
    T_out0: float = 950.0            # K (sets nominal flow heat capacity)
    m_fuel_core_kg: float = 2.0e4    # salt mass in core
    m_graphite_kg: float = 1.0e5
    cp_fuel: float = 2386.0          # J/kg/K (FLiBe)
    cp_graphite: float = 1700.0      # J/kg/K at ~900 K
    gamma_graphite: float = 0.07     # fraction of power deposited in graphite
    hA: float = 5.0e5                # W/K fuel->graphite
    hx_mode: str = "ua"              # 'ua' | 'constant_inlet' | 'constant_power'
    T_secondary: float = 800.0       # secondary-side temperature for 'ua'


@dataclass
class PKResult:
    t: np.ndarray
    n: np.ndarray
    rho: np.ndarray
    rho_feedback: np.ndarray
    rho_external: np.ndarray
    T_fuel: np.ndarray
    T_graphite: np.ndarray
    T_in: np.ndarray
    T_out: np.ndarray
    flow: np.ndarray
    precursors_in_core: np.ndarray
    rho0: float
    rho_drift: np.ndarray = None   # equivalent reactivity of precursor redistribution

    @property
    def power_fraction(self) -> np.ndarray:
        return self.n


class CirculatingPointKinetics:
    """Multi-node circulating-fuel point kinetics with thermal feedback."""

    def __init__(self, beta_i, lam_i, Lambda: float, tau_c: float, tau_L: float,
                 alpha_fuel: float = -3.0e-5, alpha_graphite: float = -1.0e-5,
                 n_core: int = 10, n_loop: int = 20, fission_shape=None,
                 importance_shape=None, thermal: Optional[ThermalModel] = None,
                 circulating: bool = True):
        self.beta_i = np.asarray(beta_i, float)
        self.lam = np.asarray(lam_i, float)
        self.beta = self.beta_i.sum()
        self.Lambda = Lambda
        self.tau_c0, self.tau_L0 = tau_c, tau_L
        self.af, self.ag = alpha_fuel, alpha_graphite
        self.nc, self.nl = n_core, n_loop
        self.circ = circulating
        x = (np.arange(n_core) + 0.5) / n_core
        f = np.sin(np.pi * x) if fission_shape is None else np.interp(
            x, *self._norm_shape(fission_shape))
        w = np.sin(np.pi * x) if importance_shape is None else np.interp(
            x, *self._norm_shape(importance_shape))
        self.f = f / f.sum()
        self.w = w / (self.f * w).sum()    # so that sum f*w = 1 (static importance)
        self.th = thermal or ThermalModel()
        th = self.th
        self.W0cp = th.power_W / (th.T_out0 - th.T_in0)   # W/K nominal flow heat capacity
        self.Tf0 = 0.5 * (th.T_in0 + th.T_out0)
        self.Tg0 = self.Tf0 + th.gamma_graphite * th.power_W / th.hA
        # 'ua' heat exchanger sized so the loop returns T_in0 at nominal power
        self.loop_T_mean0 = None
        self._ng = len(self.lam)

    @staticmethod
    def _norm_shape(s):
        xs, vs = np.asarray(s[0], float), np.asarray(s[1], float)
        xs = (xs - xs.min()) / max(xs.max() - xs.min(), 1e-30)
        return xs, np.maximum(vs, 0.0)

    # state layout: [n, C (ng x nc), CL (ng x nl), Tf, Tg, TL (nl)]
    def _unpack(self, y):
        ng, nc, nl = self._ng, self.nc, self.nl
        n = y[0]
        C = y[1:1 + ng * nc].reshape(ng, nc)
        o = 1 + ng * nc
        CL = y[o:o + ng * nl].reshape(ng, nl)
        o += ng * nl
        Tf, Tg = y[o], y[o + 1]
        TL = y[o + 2:o + 2 + nl]
        return n, C, CL, Tf, Tg, TL

    def _rhs(self, t, y, rho_ext, flow):
        ng, nc, nl = self._ng, self.nc, self.nl
        n, C, CL, Tf, Tg, TL = self._unpack(y)
        # flow always carries heat; ``circulating`` only controls precursor drift
        fr = max(flow(t), 1e-6)
        tc = self.tau_c0 / fr
        tl = self.tau_L0 / fr
        lam = self.lam[:, None]
        rho_fb = self.af * (Tf - self.Tf0) + self.ag * (Tg - self.Tg0)
        rho = self.rho0 + rho_ext(t) + rho_fb
        emis = (self.lam[:, None] * C * self.w[None, :]).sum()
        dn = (rho - self.beta) / self.Lambda * n + emis
        prod = self.beta_i[:, None] * self.f[None, :] * n / self.Lambda
        if self.circ:
            inflow_c = np.concatenate([CL[:, -1:], C[:, :-1]], axis=1)
            dC = prod - lam * C + (inflow_c - C) * (nc / tc)
            inflow_l = np.concatenate([C[:, -1:], CL[:, :-1]], axis=1)
            dCL = -lam * CL + (inflow_l - CL) * (nl / tl)
        else:
            dC = prod - lam * C
            dCL = -lam * CL
        th = self.th
        P = th.power_W * n
        Wcp = self.W0cp * fr
        T_in = TL[-1] if th.hx_mode != "constant_inlet" else th.T_in0
        Mf = th.m_fuel_core_kg * th.cp_fuel
        Mg = th.m_graphite_kg * th.cp_graphite
        dTf = ((1 - th.gamma_graphite) * P - 2.0 * Wcp * (Tf - T_in) - th.hA * (Tf - Tg)) / Mf
        dTg = (th.gamma_graphite * P + th.hA * (Tf - Tg)) / Mg
        T_out = 2.0 * Tf - T_in
        # loop temperature nodes: advection + heat exchanger removal
        inflow_T = np.concatenate([[T_out], TL[:-1]])
        Mloop = self.W0cp * self.tau_L0          # loop salt heat capacity (J/K), flow-independent
        if th.hx_mode == "constant_power":
            q_hx = th.power_W / nl * np.ones(nl)
        else:
            q_hx = self.UA_node * (TL - self._Tsec(t))
        dTL = (inflow_T - TL) * (nl / tl) - q_hx / (Mloop / nl)
        return np.concatenate([[dn], dC.ravel(), dCL.ravel(), [dTf, dTg], dTL])

    def steady_state(self, flow0: float = 1.0):
        """Steady state at n=1: precursor distribution, rho0 and HX sizing."""
        ng, nc, nl = self._ng, self.nc, self.nl
        th = self.th
        # precursors: linear solve per group
        C = np.zeros((ng, nc))
        CL = np.zeros((ng, nl))
        for i in range(ng):
            l = self.lam[i]
            m = nc + nl
            A = np.zeros((m, m))
            b = np.zeros(m)
            if self.circ:
                tc = self.tau_c0 / flow0
                tl = self.tau_L0 / flow0
                for j in range(nc):
                    A[j, j] = -l - nc / tc
                    A[j, (j - 1) if j > 0 else m - 1] += nc / tc
                    b[j] = -self.beta_i[i] * self.f[j] / self.Lambda
                for k in range(nl):
                    r = nc + k
                    A[r, r] = -l - nl / tl
                    A[r, r - 1] += nl / tl
            else:
                for j in range(nc):
                    A[j, j] = -l
                    b[j] = -self.beta_i[i] * self.f[j] / self.Lambda
                for k in range(nl):
                    A[nc + k, nc + k] = 1.0
            sol = np.linalg.solve(A, b)
            C[i], CL[i] = sol[:nc], sol[nc:]
        emis = (self.lam[:, None] * C * self.w[None, :]).sum()
        self.rho0 = self.beta - emis * self.Lambda
        # loop temperatures: linear drop from T_out0 to T_in0 across the loop
        TL = np.linspace(th.T_out0, th.T_in0, nl + 1)[1:]
        Wcp = self.W0cp * flow0
        if th.hx_mode == "ua":
            # UA per node so that steady state holds: removal per node = Wcp*dT_node
            dT_node = (th.T_out0 - th.T_in0) / nl
            self.UA_node = Wcp * dT_node / np.maximum(TL - th.T_secondary, 1e-6)
        else:
            self.UA_node = np.zeros(nl)
        y0 = np.concatenate([[1.0], C.ravel(), CL.ravel(), [self.Tf0, self.Tg0], TL])
        return y0

    def run(self, t_end: float, rho_ext: Callable = lambda t: 0.0,
            flow: Callable = lambda t: 1.0, max_step: float = 0.05, t_eval=None,
            rtol: float = 1e-7, atol: float = 1e-10,
            T_secondary: Optional[Callable] = None) -> PKResult:
        """Integrate a transient.

        ``rho_ext(t)`` external reactivity, ``flow(t)`` pump flow fraction and
        ``T_secondary(t)`` heat-exchanger secondary temperature (K, 'ua' mode).
        """
        self._Tsec = T_secondary or (lambda t: self.th.T_secondary)
        y0 = self.steady_state(flow(0.0))
        if t_eval is None:
            t_eval = np.linspace(0.0, t_end, 1001)
        sol = solve_ivp(self._rhs, (0.0, t_end), y0, method="BDF", t_eval=t_eval,
                        args=(rho_ext, flow), max_step=max_step, rtol=rtol, atol=atol)
        Y = sol.y
        n = Y[0]
        ng, nc, nl = self._ng, self.nc, self.nl
        o = 1 + ng * nc + ng * nl
        Tf, Tg = Y[o], Y[o + 1]
        TL = Y[o + 2:o + 2 + nl]
        T_in = TL[-1] if self.th.hx_mode != "constant_inlet" else np.full_like(n, self.th.T_in0)
        rfb = self.af * (Tf - self.Tf0) + self.ag * (Tg - self.Tg0)
        rex = np.array([rho_ext(t) for t in sol.t])
        Ccore = Y[1:1 + ng * nc].reshape(ng, nc, -1)
        Cl = Y[1 + ng * nc:o].reshape(ng, nl, -1)
        frac = Ccore.sum(axis=(0, 1)) / (Ccore.sum(axis=(0, 1)) + Cl.sum(axis=(0, 1)) *
                                        (self.tau_L0 / self.nl) / (self.tau_c0 / self.nc))
        # delayed-neutron source expressed as quasi-static reactivity:
        # rho_drift = Lambda * S_d / n - (beta - rho0)  (0 at the initial steady state)
        emis = (self.lam[:, None, None] * Ccore * self.w[None, :, None]).sum(axis=(0, 1))
        rho_drift = self.Lambda * emis / n - (self.beta - self.rho0)
        return PKResult(sol.t, n, self.rho0 + rex + rfb, rfb, rex, Tf, Tg, T_in,
                        2 * Tf - T_in, np.array([flow(t) for t in sol.t]), frac, self.rho0,
                        rho_drift)

    # ------------------------------------------------------------------
    def stable_period(self, rho_step: float, flow0: float = 1.0) -> float:
        """Asymptotic period (s) for a step (no feedback) from the linear system."""
        ng, nc, nl = self._ng, self.nc, self.nl
        self.steady_state(flow0)
        m = 1 + ng * (nc + nl)
        J = np.zeros((m, m))
        rho = self.rho0 + rho_step
        J[0, 0] = (rho - self.beta) / self.Lambda
        tc, tl = self.tau_c0 / flow0, self.tau_L0 / flow0
        for i in range(ng):
            l = self.lam[i]
            for j in range(nc):
                r = 1 + i * nc + j
                J[0, r] = l * self.w[j]
                J[r, 0] = self.beta_i[i] * self.f[j] / self.Lambda
                J[r, r] = -l - (nc / tc if self.circ else 0.0)
                if self.circ:
                    src = (1 + ng * nc + i * nl + nl - 1) if j == 0 else r - 1
                    J[r, src] += nc / tc
            for k in range(nl):
                r = 1 + ng * nc + i * nl + k
                J[r, r] = -l - (nl / tl if self.circ else 0.0)
                if self.circ:
                    src = (1 + i * nc + nc - 1) if k == 0 else r - 1
                    J[r, src] += nl / tl
                else:
                    J[r, r] = -1.0
        ev = np.linalg.eigvals(J)
        w = ev.real.max()
        return 1.0 / w if w > 0 else np.inf


def inhour_static(period: float, beta_i, lam_i, Lambda: float) -> float:
    """Reactivity for a stable period T (stationary fuel)."""
    w = 1.0 / period
    return float(w * Lambda + np.sum(np.asarray(beta_i) * w / (w + np.asarray(lam_i))))


def prompt_jump(rho: float, beta: float) -> float:
    return beta / (beta - rho)


def decay_heat_fraction(t_after: np.ndarray, T_operation: float) -> np.ndarray:
    """Way-Wigner decay heat P_d/P_0 (t, T in seconds)."""
    t = np.maximum(np.asarray(t_after, float), 1e-3)
    return 0.0622 * (t ** -0.2 - (t + T_operation) ** -0.2)


def thermal_model_for(design, alpha_mass_fuel_core: Optional[float] = None) -> ThermalModel:
    """Build a ThermalModel from an MSRDesign (approximate inventories)."""
    from .materials import graphite as _g
    v = design.salt_volumes_m3()
    salt = design.fuel_salt()
    m_salt_core = v["core_zones"] * 1e6 * salt.density / 1000.0
    Vg = sum(vol * (1 - z.salt_fraction) for vol, z in zip(design.zone_volumes_cm3(), design.zones))
    m_g = Vg * design.graphite_density / 1000.0
    cp = 2386.0 if "LiF" in design.salt_composition else 1100.0
    dT = 100.0
    return ThermalModel(power_W=design.power_MW * 1e6, T_in0=design.T_fuel - dT / 2,
                        T_out0=design.T_fuel + dT / 2, m_fuel_core_kg=max(m_salt_core, 1.0),
                        m_graphite_kg=max(m_g, 1.0), cp_fuel=cp,
                        gamma_graphite=0.07 if m_g > 1 else 0.0,
                        hA=max(2.0e3 * design.power_MW, 1e3),
                        T_secondary=design.T_fuel - dT / 2 - 60.0)
