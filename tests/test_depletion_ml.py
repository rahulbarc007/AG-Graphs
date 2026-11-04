"""Verification of CRAM depletion and the ML layer."""
import numpy as np
import pytest
from scipy.linalg import expm

from msrnx.depletion.cram import augment_with_feed, cram16
from msrnx.ml import (DesignSpace, MultiFidelitySurrogate, Parameter, PODModel, ishigami,
                      ishigami_analytic, make_surrogate, regression_metrics, sobol_indices)


def test_cram_scalar_and_stiff_chain():
    for x in (0.0, 0.5, 5.0, 50.0):
        assert cram16(np.array([[-x]]), np.array([1.0]), 1.0)[0] == pytest.approx(np.exp(-x), abs=1e-14)
    lam = np.log(2) / np.array([1e30, 21.83 * 60, 26.975 * 86400, 1e30])
    A = np.zeros((4, 4))
    A[0, 0] = -3e-10
    A[1, 0] = 3e-10
    A[1, 1] = -lam[1]
    A[2, 1] = lam[1]
    A[2, 2] = -lam[2] - 1e-9
    A[3, 2] = lam[2]
    A[3, 3] = -2e-9
    n0 = np.array([1.0, 0, 0, 0])
    dt = 30 * 86400
    assert np.allclose(cram16(A, n0, dt), expm(A * dt) @ n0, rtol=1e-10, atol=1e-18)
    B = augment_with_feed(A, np.array([0, 0, 0, 1e-12]))
    v = np.append(n0, 1.0)
    assert np.allclose(cram16(B, v, dt), expm(B * dt) @ v, rtol=1e-10, atol=1e-18)


def test_sobol_matches_ishigami():
    sp = DesignSpace([Parameter(f"x{i}", -np.pi, np.pi) for i in range(3)])
    r = sobol_indices(ishigami, sp, n=8192, seed=1, n_boot=20)
    a = ishigami_analytic()
    assert np.allclose(r.ST, a["ST"], atol=0.02)
    assert np.allclose(r.S1, a["S1"], atol=0.02)


def test_gp_surrogate_accuracy_and_uncertainty():
    sp = DesignSpace([Parameter("x", 0.0, 1.0), Parameter("y", 0.0, 1.0)])
    f = lambda X: np.sin(4 * X[:, 0]) * np.exp(-X[:, 1]) + X[:, 1] ** 2
    X = sp.sample(60, "lhs", 0)
    Xt = sp.sample(200, "random", 1)
    gp = make_surrogate("gp", sp).fit(X, f(X))
    mu, sd = gp.predict(Xt, return_std=True)
    assert regression_metrics(f(Xt), mu)["r2"] > 0.995
    assert np.mean(np.abs(mu - f(Xt)) < 3 * sd + 1e-3) > 0.9


def test_multifidelity_beats_single_fidelity():
    sp = DesignSpace([Parameter("x", 0.0, 1.0), Parameter("y", 0.0, 1.0)])
    hf = lambda X: np.sin(6 * X[:, 0]) + 0.3 * X[:, 1] + 0.1 * np.cos(9 * X[:, 1])
    lf = lambda X: np.sin(6 * X[:, 0]) * 0.9 + 0.2
    X = sp.sample(12, "lhs", 0)
    Xt = sp.sample(300, "random", 2)
    sf = make_surrogate("gp", sp).fit(X, hf(X))
    mf = MultiFidelitySurrogate(sp).fit(X, lf(X), hf(X))
    e_sf = regression_metrics(hf(Xt), sf.predict(Xt))["rmse"]
    e_mf = regression_metrics(hf(Xt), mf.predict(Xt, lf(Xt)))["rmse"]
    assert e_mf < e_sf


def test_pod_reconstruction():
    sp = DesignSpace([Parameter("a", 0.5, 1.5), Parameter("b", -0.5, 0.5)])
    x = np.linspace(0, 1, 100)
    X = sp.sample(40, "lhs", 0)
    S = np.array([np.sin(np.pi * x) ** a * (1 + b * x) for a, b in X])
    pod = PODModel(energy=0.99999, space=sp).fit(X, S)
    assert pod.rank < 10
    assert np.abs(pod.project(S) - S).max() < 1e-2
