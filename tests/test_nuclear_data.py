"""Verification of the nuclear-data layer."""
import numpy as np
import pytest

from msrnx.constants import K_BOLTZMANN_EV
from msrnx.data.nuclides import NUCLIDES
from msrnx.xs.kernels import free_gas_kernel, free_gas_total_factor
from msrnx.xs.library import load_library
from msrnx.xs.pointwise import PointwiseNuclide


@pytest.mark.parametrize("name", sorted(NUCLIDES))
def test_calibration_reproduces_targets(name):
    nuc = NUCLIDES[name]
    r = PointwiseNuclide(nuc).cal.report
    assert r["capture_th"] == pytest.approx(nuc.capture_th, rel=1e-3, abs=1e-6)
    if nuc.fission_th:
        assert r["fission_th"] == pytest.approx(nuc.fission_th, rel=1e-3)
    if nuc.RI_capture:
        assert r["RI_capture"] == pytest.approx(nuc.RI_capture, rel=0.02, abs=0.05)
    if nuc.RI_fission:
        assert r["RI_fission"] == pytest.approx(nuc.RI_fission, rel=0.02)


@pytest.mark.parametrize("A,T", [(1.0, 300.0), (11.9, 900.0), (18.84, 1200.0)])
def test_free_gas_kernel_normalization(A, T):
    kT = K_BOLTZMANN_EV * T
    for E in (0.01, 0.0253, 0.3):
        Ep = np.unique(np.concatenate([np.geomspace(1e-7, E * 40 + 20 * kT, 200000), [E]]))
        num = np.trapezoid(free_gas_kernel(E, Ep, A, T), Ep)
        assert num == pytest.approx(free_gas_total_factor(E, A, T), rel=1e-5)


def test_free_gas_detailed_balance():
    A, T = 11.9, 900.0
    kT = K_BOLTZMANN_EV * T
    M = lambda e: e * np.exp(-e / kT)
    for E, Ep in [(0.02, 0.05), (0.05, 0.3), (0.1, 0.01)]:
        lhs = M(E) * free_gas_kernel(E, Ep, A, T)
        rhs = M(Ep) * free_gas_kernel(Ep, E, A, T)
        assert lhs == pytest.approx(rhs, rel=1e-8)


def test_doppler_increases_shielded_absorption():
    lib = load_library()
    for n in ("Th232", "U238"):
        g = lib.groups.group_of(21.0 if n == "Th232" else 6.67)
        cold = lib[n].micro(293.6, 50.0)["capture"][g]
        hot = lib[n].micro(1200.0, 50.0)["capture"][g]
        inf = lib[n].micro(900.0, 1e10)["capture"][g]
        assert hot > cold            # Doppler broadening reduces self-shielding
        assert cold < 0.5 * inf      # strong shielding at sigma0 = 50 b


def test_transfer_matrices_are_normalized():
    lib = load_library()
    for n in ("C12", "F19", "Li7", "U238"):
        P = lib[n].micro(900.0)["P_el"]
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-10)
