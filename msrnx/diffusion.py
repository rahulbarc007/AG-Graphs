"""Multigroup neutron diffusion solver (finite volume).

Geometries: 1D slab / cylinder / sphere and 2D cylindrical (R-Z).  Cell-
centred finite volumes with harmonic-mean interface diffusion coefficients
and Marshak (vacuum), zero-flux or reflective boundaries.  The k-eigenvalue
problem  M phi = (1/k) F phi  (and its adjoint) is solved with ARPACK on the
operator M^-1 F using a sparse LU factorization of M (power-iteration
fallback); several modes give the dominance ratio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .constants import MEV_TO_J
from .macro import MacroXS


@dataclass
class Mesh:
    geometry: str                       # 'slab' | 'cylinder' | 'sphere' | 'rz'
    r_edges: np.ndarray                 # x for slab
    z_edges: Optional[np.ndarray] = None
    region: Optional[np.ndarray] = None # int array (nr, nz)
    bc: dict = field(default_factory=lambda: {"r_min": "reflective", "r_max": "vacuum",
                                              "z_min": "vacuum", "z_max": "vacuum"})
    transverse_B2: float = 0.0          # buckling for 1D problems

    def __post_init__(self):
        self.r_edges = np.asarray(self.r_edges, float)
        if self.geometry == "rz":
            self.z_edges = np.asarray(self.z_edges, float)
        else:
            self.z_edges = np.array([0.0, 1.0])
        if self.region is None:
            self.region = np.zeros((self.nr, self.nz), dtype=int)
        self.region = np.asarray(self.region, dtype=int)

    @property
    def nr(self) -> int:
        return len(self.r_edges) - 1

    @property
    def nz(self) -> int:
        return len(self.z_edges) - 1

    @property
    def N(self) -> int:
        return self.nr * self.nz

    @property
    def rc(self) -> np.ndarray:
        return 0.5 * (self.r_edges[1:] + self.r_edges[:-1])

    @property
    def zc(self) -> np.ndarray:
        return 0.5 * (self.z_edges[1:] + self.z_edges[:-1])

    @property
    def dr(self) -> np.ndarray:
        return np.diff(self.r_edges)

    @property
    def dz(self) -> np.ndarray:
        return np.diff(self.z_edges)

    def volumes(self) -> np.ndarray:
        """Cell volumes, shape (nr, nz)."""
        r = self.r_edges
        if self.geometry == "slab":
            v = np.diff(r)
        elif self.geometry == "cylinder":
            v = np.pi * (r[1:] ** 2 - r[:-1] ** 2)
        elif self.geometry == "sphere":
            v = 4.0 / 3.0 * np.pi * (r[1:] ** 3 - r[:-1] ** 3)
        else:
            v = np.pi * (r[1:] ** 2 - r[:-1] ** 2)
            return v[:, None] * self.dz[None, :]
        return v[:, None] * np.ones((1, 1))

    def radial_face_areas(self) -> np.ndarray:
        """Areas of the nr+1 radial faces, shape (nr+1, nz)."""
        r = self.r_edges
        if self.geometry == "slab":
            a = np.ones_like(r)
        elif self.geometry in ("cylinder",):
            a = 2.0 * np.pi * r
        elif self.geometry == "sphere":
            a = 4.0 * np.pi * r ** 2
        else:
            return 2.0 * np.pi * r[:, None] * self.dz[None, :]
        return a[:, None] * np.ones((1, 1))

    def axial_face_areas(self) -> np.ndarray:
        r = self.r_edges
        return np.pi * (r[1:] ** 2 - r[:-1] ** 2)


def uniform_edges(segments) -> np.ndarray:
    """Build mesh edges from [(length, n_cells), ...] segments."""
    edges = [0.0]
    for length, n in segments:
        step = length / n
        for _ in range(n):
            edges.append(edges[-1] + step)
    return np.array(edges)


@dataclass
class DiffusionResult:
    k: float
    flux: np.ndarray                 # (G, nr, nz)
    mesh: Mesh
    xs: list
    adjoint: Optional[np.ndarray] = None
    k_modes: Optional[np.ndarray] = None
    flux_modes: Optional[np.ndarray] = None
    power_W: Optional[float] = None
    iterations: int = 0

    @property
    def G(self) -> int:
        return self.flux.shape[0]

    @property
    def dominance_ratio(self) -> float:
        if self.k_modes is None or len(self.k_modes) < 2:
            return float("nan")
        return float(abs(self.k_modes[1]) / abs(self.k_modes[0]))

    def field(self, attr: str) -> np.ndarray:
        """Per-cell array (G, nr, nz) of a MacroXS attribute."""
        G = self.G
        out = np.zeros((G, self.mesh.nr, self.mesh.nz))
        for rid, xs in enumerate(self.xs):
            m = self.mesh.region == rid
            out[:, m] = getattr(xs, attr)[:, None]
        return out

    def power_density(self) -> np.ndarray:
        """Power density (W/cm3), shape (nr, nz)."""
        kf = self.field("kappa_fission")
        return (kf * self.flux).sum(axis=0) * MEV_TO_J

    def normalize(self, power_W: float) -> "DiffusionResult":
        V = self.mesh.volumes()
        P = float((self.power_density() * V).sum())
        self.flux *= power_W / P
        if self.flux_modes is not None:
            self.flux_modes = self.flux_modes * (power_W / P)
        self.power_W = power_W
        return self

    def total_power(self) -> float:
        return float((self.power_density() * self.mesh.volumes()).sum())

    def region_integral(self, attr: str, region_ids=None) -> np.ndarray:
        """Group-wise integral of Sigma_attr * phi * V over the selected regions."""
        V = self.mesh.volumes()
        F = self.field(attr) * self.flux * V[None]
        if region_ids is None:
            return F.reshape(self.G, -1).sum(axis=1)
        m = np.isin(self.mesh.region, np.atleast_1d(region_ids))
        return F[:, m].sum(axis=1)

    def region_flux(self, region_ids=None) -> np.ndarray:
        """Volume-integrated group flux (n cm / s) over regions."""
        V = self.mesh.volumes()
        F = self.flux * V[None]
        if region_ids is None:
            return F.reshape(self.G, -1).sum(axis=1)
        m = np.isin(self.mesh.region, np.atleast_1d(region_ids))
        return F[:, m].sum(axis=1)

    def peaking(self, region_ids=None) -> dict:
        q = self.power_density()
        V = self.mesh.volumes()
        m = q > 0 if region_ids is None else np.isin(self.mesh.region, np.atleast_1d(region_ids))
        qa = float((q[m] * V[m]).sum() / V[m].sum())
        qmax = float(q[m].max())
        out = {"q_avg": qa, "q_max": qmax, "total_peaking": qmax / qa}
        if self.mesh.geometry == "rz":
            qr = (q * V).sum(axis=1) / np.maximum(V.sum(axis=1), 1e-30)
            qz = (q * V).sum(axis=0) / np.maximum(V.sum(axis=0), 1e-30)
            rm = qr > 0
            zm = qz > 0
            Vr = V.sum(axis=1)[rm]
            Vz = V.sum(axis=0)[zm]
            out["radial_peaking"] = float(qr[rm].max() / ((qr[rm] * Vr).sum() / Vr.sum()))
            out["axial_peaking"] = float(qz[zm].max() / ((qz[zm] * Vz).sum() / Vz.sum()))
        return out

    def leakage(self) -> np.ndarray:
        """Group-wise net leakage = production/k - absorption balance (n/s)."""
        prod = (self.field("chi") * (self.field("nu_fission") * self.flux).sum(axis=0)[None]
                * self.mesh.volumes()[None]).reshape(self.G, -1).sum(axis=1) / self.k
        V = self.mesh.volumes()
        S = np.zeros(self.G)
        rem = np.zeros(self.G)
        for rid, xs in enumerate(self.xs):
            m = self.mesh.region == rid
            phi_r = (self.flux[:, m] * V[m][None]).sum(axis=1)
            S += xs.scatter.T @ phi_r
            rem += xs.total * phi_r
        return prod + S - rem


class DiffusionSolver:
    """Assembles and solves the multigroup diffusion eigenvalue problem."""

    def __init__(self, mesh: Mesh, xs_regions: list):
        self.mesh = mesh
        self.xs = list(xs_regions)
        self.G = self.xs[0].G
        if self.mesh.region.max() >= len(self.xs):
            raise ValueError("region index exceeds number of cross-section sets")
        self._assemble()

    # ------------------------------------------------------------------
    def _cell_field(self, attr):
        out = np.zeros((self.G, self.mesh.nr, self.mesh.nz))
        for rid, xs in enumerate(self.xs):
            m = self.mesh.region == rid
            out[:, m] = getattr(xs, attr)[:, None]
        return out

    def _assemble(self):
        mesh, G = self.mesh, self.G
        nr, nz, N = mesh.nr, mesh.nz, mesh.N
        V = mesh.volumes()
        D = self._cell_field("D")
        tot = self._cell_field("total")
        Ar = mesh.radial_face_areas()
        dr = mesh.dr
        rows, cols, vals = [], [], []
        cid = (np.arange(nr)[:, None] + nr * np.arange(nz)[None, :])  # (nr, nz)

        def bc_coeff(kind, Dc, delta, area):
            if kind == "reflective":
                return np.zeros_like(Dc)
            if kind == "zero":
                return area * 2.0 * Dc / delta
            return area * 2.0 * Dc / (delta + 4.0 * Dc)   # Marshak vacuum

        for g in range(G):
            off = g * N
            diag = (tot[g] + D[g] * mesh.transverse_B2) * V
            # radial interior faces
            if nr > 1:
                Da, Db = D[g, :-1, :], D[g, 1:, :]
                da, db = dr[:-1, None], dr[1:, None]
                a = Ar[1:-1] * 2.0 * Da * Db / (Da * db + Db * da)
                ia, ib = cid[:-1, :].ravel() + off, cid[1:, :].ravel() + off
                rows += [ia, ib]
                cols += [ib, ia]
                vals += [-a.ravel(), -a.ravel()]
                diag[:-1, :] += a
                diag[1:, :] += a
            # radial boundaries
            inner = mesh.bc.get("r_min", "reflective")
            outer = mesh.bc.get("r_max", "vacuum")
            if mesh.r_edges[0] == 0.0 and mesh.geometry in ("cylinder", "sphere", "rz"):
                inner = "reflective"
            diag[0, :] += bc_coeff(inner, D[g, 0, :], dr[0], Ar[0])
            diag[-1, :] += bc_coeff(outer, D[g, -1, :], dr[-1], Ar[-1])
            # axial
            if mesh.geometry == "rz":
                Az = mesh.axial_face_areas()[:, None]
                dz = mesh.dz
                if nz > 1:
                    Da, Db = D[g, :, :-1], D[g, :, 1:]
                    da, db = dz[None, :-1], dz[None, 1:]
                    a = Az * 2.0 * Da * Db / (Da * db + Db * da)
                    ia, ib = cid[:, :-1].ravel() + off, cid[:, 1:].ravel() + off
                    rows += [ia, ib]
                    cols += [ib, ia]
                    vals += [-a.ravel(), -a.ravel()]
                    diag[:, :-1] += a
                    diag[:, 1:] += a
                diag[:, 0] += bc_coeff(mesh.bc.get("z_min", "vacuum"), D[g, :, 0], dz[0], Az[:, 0])
                diag[:, -1] += bc_coeff(mesh.bc.get("z_max", "vacuum"), D[g, :, -1], dz[-1], Az[:, 0])
            rows.append(cid.ravel() + off)
            cols.append(cid.ravel() + off)
            vals.append(diag.ravel())
        # scattering coupling: equation g, unknown g' -> -S[g', g] V
        for rid, xs in enumerate(self.xs):
            m = (mesh.region == rid)
            cells = cid[m]
            vol = V[m]
            S = xs.scatter
            for gp in range(G):
                for g in range(G):
                    s = S[gp, g]
                    if s == 0.0:
                        continue
                    rows.append(cells + g * N)
                    cols.append(cells + gp * N)
                    vals.append(-s * vol)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        self.M = sp.csc_matrix((vals, (rows, cols)), shape=(G * N, G * N))
        self.V = V.ravel(order="F")  # cell order matches cid (i fastest)
        # fission operator factors
        chi = self._cell_field("chi")
        nsf = self._cell_field("nu_fission")
        self.chi = np.stack([chi[g].ravel(order="F") for g in range(G)])
        self.nsf = np.stack([nsf[g].ravel(order="F") for g in range(G)])
        self._lu = None

    # ------------------------------------------------------------------
    def _F(self, x):
        X = x.reshape(self.G, -1)
        s = (self.nsf * X).sum(axis=0) * self.V
        return (self.chi * s[None, :]).ravel()

    def _FT(self, x):
        X = x.reshape(self.G, -1)
        s = (self.chi * X).sum(axis=0)
        return (self.nsf * (s * self.V)[None, :]).ravel()

    def lu(self):
        if self._lu is None:
            self._lu = spla.splu(self.M, permc_spec="MMD_AT_PLUS_A")
        return self._lu

    def solve(self, adjoint: bool = False, n_modes: int = 1, tol: float = 1e-10,
              method: str = "arpack") -> DiffusionResult:
        lu = self.lu()
        n = self.M.shape[0]
        if adjoint:
            op = lambda x: lu.solve(self._FT(x), trans="T")
        else:
            op = lambda x: lu.solve(self._F(x))
        v0 = np.ones(n)
        k_modes = None
        vecs = None
        if method == "arpack":
            try:
                A = spla.LinearOperator((n, n), matvec=op, dtype=float)
                nm = max(n_modes, 1)
                vals, vecs = spla.eigs(A, k=nm, which="LM", v0=v0, tol=tol,
                                       ncv=min(n - 1, max(2 * nm + 8, 20)), maxiter=5000)
                order = np.argsort(-np.abs(vals))
                vals, vecs = vals[order], vecs[:, order]
                k = float(vals[0].real)
                phi = vecs[:, 0].real
                k_modes = vals.real
                iterations = -1
            except Exception:  # pragma: no cover - fallback
                method = "power"
        if method == "power":
            phi = v0 / np.linalg.norm(v0)
            k = 1.0
            iterations = 0
            for iterations in range(1, 20000):
                new = op(phi)
                k_new = float(np.linalg.norm(new))
                new /= k_new
                if np.linalg.norm(new - phi) < tol * 10 and abs(k_new - k) < tol:
                    phi, k = new, k_new
                    break
                phi, k = new, k_new
        if phi.sum() < 0:
            phi = -phi
        phi = np.maximum(phi, 0.0) if not adjoint else phi
        G, nr, nz = self.G, self.mesh.nr, self.mesh.nz
        shaped = np.stack([phi.reshape(G, -1)[g].reshape(nz, nr).T for g in range(G)])
        res = DiffusionResult(k, shaped, self.mesh, self.xs, k_modes=k_modes,
                              iterations=iterations)
        if vecs is not None and n_modes > 1:
            modes = []
            for j in range(vecs.shape[1]):
                v = vecs[:, j].real
                modes.append(np.stack([v.reshape(G, -1)[g].reshape(nz, nr).T for g in range(G)]))
            res.flux_modes = np.array(modes)
        return res

    def solve_forward_adjoint(self, n_modes: int = 1) -> DiffusionResult:
        fwd = self.solve(adjoint=False, n_modes=n_modes)
        adj = self.solve(adjoint=True)
        fwd.adjoint = adj.flux
        fwd.__dict__["k_adjoint"] = adj.k
        return fwd


def solve_diffusion(mesh: Mesh, xs_regions: list, adjoint: bool = False,
                    n_modes: int = 1) -> DiffusionResult:
    solver = DiffusionSolver(mesh, xs_regions)
    if adjoint:
        return solver.solve_forward_adjoint(n_modes=n_modes)
    return solver.solve(n_modes=n_modes)
