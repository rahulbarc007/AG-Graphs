"""Chebyshev Rational Approximation Method (CRAM) for exp(A t) n0.

Incomplete partial-fraction (IPF) form of order 16 (Pusa 2016), accurate
to ~1e-15 for burnup matrices whose eigenvalues lie near the negative real
axis, regardless of stiffness (short half-lives).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

C16_ALPHA = np.array([
    +5.464930576870210e+3 - 3.797983575308356e+4j,
    +9.045112476907548e+1 - 1.115537522430261e+3j,
    +2.344818070467641e+2 - 4.228020157070496e+2j,
    +9.453304067358312e+1 - 2.951294291446048e+2j,
    +7.283792954673409e+2 - 1.205646080220011e+5j,
    +3.648229059594851e+1 - 1.155509621409682e+2j,
    +2.547321630156819e+1 - 2.639500283021502e+1j,
    +2.394538338734709e+1 - 5.650522971778156e+0j], dtype=np.complex128)

C16_THETA = np.array([
    +3.509103608414918 + 8.436198985884374j,
    +5.948152268951177 + 3.587457362018322j,
    -5.264971343442647 + 16.22022147316793j,
    +1.419375897185666 + 10.92536348449672j,
    +6.416177699099435 + 1.194122393370139j,
    +4.993174737717997 + 5.996881713603942j,
    -1.413928462488886 + 13.49772569889275j,
    -10.84391707869699 + 19.27744616718165j], dtype=np.complex128)

C16_ALPHA0 = 2.124853710495224e-16


def cram16(A, n0: np.ndarray, dt: float) -> np.ndarray:
    """Return exp(A dt) n0 using IPF CRAM-16 (A dense or sparse)."""
    As = sp.csc_matrix(A, dtype=np.float64) * dt
    ident = sp.identity(As.shape[0], format="csc", dtype=np.complex128)
    y = np.asarray(n0, dtype=np.float64).copy()
    for alpha, theta in zip(C16_ALPHA, C16_THETA):
        y = y + 2.0 * np.real(alpha * spla.spsolve(As - theta * ident, y))
    return y * C16_ALPHA0


def augment_with_feed(A: np.ndarray, feed: np.ndarray) -> np.ndarray:
    """Augmented matrix [[A, f], [0, 0]] so that a constant feed is exact."""
    n = A.shape[0]
    B = np.zeros((n + 1, n + 1))
    B[:n, :n] = A
    B[:n, n] = feed
    return B
