"""Analytic verification of the diffusion solver, lattice code and kinetics."""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import j0, j1

from msrnx.data.delayed_neutrons import DELAYED
from msrnx.diffusion import DiffusionSolver, Mesh
from msrnx.kinetics import (CirculatingPointKinetics, circulating_beta, inhour_static,
                            uniform_loss_factor)
from msrnx.lattice import CellGeometry, run_lattice
from msrnx.macro import MacroXS
from msrnx.materials import graphite, msbr_fuel_salt
from msrnx.xs.energy import GroupStructure


def _xs(tot, tr, nsf, S, chi):
    G = len(tot)
    gs = GroupStructure(f"{G}g", np.geomspace(2e7, 1e-5, G + 1))
    z = np.zeros(G)
    return MacroXS(gs, total=np.array(tot, float), transport=np.array(tr, float),
                   absorption=np.array(tot) - np.array(S).sum(1), capture=z,
                   fission=np.array(nsf) / 2.4, nu_fission=np.array(nsf, float),
                   kappa_fission=np.array(nsf) * 80, nalpha=z, n2n=z, tritium=z,
                   scatter=np.array(S, float), chi=np.array(chi, float), chi_d=np.array(chi, float),
                   invv=np.ones(G) * 1e-6)


def test_two_group_sphere_zero_flux_exact():
    xs = _xs([0.026, 0.08], [1 / (3 * 1.4), 1 / (3 * 0.4)], [0.005, 0.12], [[0, 0.016], [0, 0]], [1, 0])
    D1, D2 = xs.D
    R = 50.0
    res = DiffusionSolver(Mesh("sphere", np.linspace(0, R, 801), bc={"r_max": "zero"}), [xs]).solve()
    B2 = (np.pi / R) ** 2
    k = (0.005 * (D2 * B2 + 0.08) + 0.12 * 0.016) / ((0.026 + D1 * B2) * (0.08 + D2 * B2))
    assert (res.k - k) * 1e5 == pytest.approx(0.0, abs=0.1)


def test_one_group_cylinder_marshak_exact():
    xs = _xs([0.30], [0.30], [0.025], [[0.28]], [1.0])
    D, R, H = xs.D[0], 60.0, 140.0
    Br = brentq(lambda b: D * b * j1(b * R) - 0.5 * j0(b * R), 1e-6, 2.40 / R)
    Bz = brentq(lambda b: D * b * np.tan(b * H / 2) - 0.5, 1e-6, np.pi / H - 1e-9)
    k_exact = 0.025 / (0.02 + D * (Br ** 2 + Bz ** 2))
    res = DiffusionSolver(Mesh("rz", np.linspace(0, R, 161), np.linspace(0, H, 241)), [xs]).solve()
    assert (res.k - k_exact) * 1e5 == pytest.approx(0.0, abs=0.5)


def test_adjoint_eigenvalue_equals_forward():
    xs = _xs([0.026, 0.08], [1 / (3 * 1.4), 1 / (3 * 0.4)], [0.005, 0.12], [[0, 0.016], [0.001, 0]], [1, 0])
    s = DiffusionSolver(Mesh("rz", np.linspace(0, 60, 31), np.linspace(0, 140, 41)), [xs])
    assert s.solve().k == pytest.approx(s.solve(adjoint=True).k, rel=1e-10)


def test_lattice_six_factor_and_lumping():
    salt, mod = msbr_fuel_salt(900.0), graphite(1.86, 920.0)
    cell = CellGeometry(1.0, 0.13, "hex")
    het = run_lattice(salt, mod, cell, heterogeneous=True)
    hom = run_lattice(salt, mod, cell, heterogeneous=False)
    assert het.six_factor["k"] == pytest.approx(het.k_inf, rel=1e-8)
    assert het.k_inf > hom.k_inf                       # resonance lumping
    assert het.six_factor["p"] > hom.six_factor["p"]


def test_infinite_medium_maxwellian_thermalization():
    """Weakly absorbing graphite: the thermal spectrum approaches a Maxwellian
    at the moderator temperature (detailed balance of the multigroup kernel)."""
    from msrnx.macro import build_macro
    T = 900.0
    m = build_macro(graphite(1.86, T))
    th = m.groups.thermal_groups()
    S = m.scatter[np.ix_(th, th)]
    A = np.diag(m.total[th]) - S.T
    src = np.zeros(len(th))
    src[0] = 1.0
    phi = np.linalg.solve(A, src)
    E = m.groups.midpoints[th]
    from msrnx.constants import K_BOLTZMANN_EV
    peak = E[np.argmax(phi / m.groups.lethargy_width[th])]
    assert 1.0 < peak / (K_BOLTZMANN_EV * T) < 3.0     # flux per lethargy peaks near 2kT


def test_kinetics_inhour_and_circulation_limits():
    d = DELAYED["U235"]
    pk = CirculatingPointKinetics(d.beta_i, d.lam_i, 2.4e-4, 8.46, 16.73, circulating=False)
    for rho in (50e-5, 150e-5):
        T = pk.stable_period(rho)
        assert inhour_static(T, d.beta_i, d.lam_i, 2.4e-4) == pytest.approx(rho, rel=1e-6)
    # infinitely long loop: only in-core decay survives
    F = uniform_loss_factor(d.lam_i, 8.0, 1e9)
    a = d.lam_i * 8.0
    assert np.allclose(F, 1 - (1 - np.exp(-a)) / a)
    # MSRE benchmark (uniform slug flow, U-235): ~283 pcm
    assert circulating_beta(d.beta_i, d.lam_i, 8.46, 16.73, "uniform").loss_pcm == pytest.approx(283, abs=3)
