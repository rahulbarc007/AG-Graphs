"""Physics evaluators that feed the AI/ML layer.

``evaluate_hf``  high fidelity: lattice -> spectral zones -> 2D R-Z (6 groups)
``evaluate_lf``  low fidelity : lattice + B1 with bare-core buckling (no reflector
                 transport), ~20x cheaper but biased -> target of ML correction.

The default design space is a single-zone graphite-moderated thorium breeder
(LiF-BeF2-ThF4-UF4 with U-233) with nine design / operating parameters.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Optional

import numpy as np

from ..core import CoreZone, MSRDesign, solve_core
from ..lattice import CellGeometry, conversion_ratio, fundamental_mode, run_lattice
from ..materials import graphite, salt_at_temperature
from .design_space import DesignSpace, Parameter

MSR_SPACE = DesignSpace([
    Parameter("salt_fraction", 0.08, 0.40, 0.15, label="Salt volume fraction"),
    Parameter("uf4_pct", 0.15, 0.60, 0.25, unit="mol%", label="UF4 [mol%]"),
    Parameter("thf4_pct", 6.0, 14.0, 12.0, unit="mol%", label="ThF4 [mol%]"),
    Parameter("core_radius", 120.0, 260.0, 190.0, unit="cm", label="Core radius [cm]"),
    Parameter("h_over_d", 0.7, 1.5, 1.0, label="H/D"),
    Parameter("reflector", 20.0, 80.0, 50.0, unit="cm", label="Reflector [cm]"),
    Parameter("channel_radius", 0.5, 3.0, 1.2, unit="cm", label="Channel radius [cm]"),
    Parameter("T_fuel", 850.0, 1000.0, 900.0, unit="K", label="Fuel temperature [K]"),
    Parameter("li6_ppm", 5.0, 500.0, 50.0, log=True, unit="ppm", label="Li-6 [ppm]"),
])

HF_OUTPUTS = ("k_eff", "conversion_ratio", "alpha_fuel", "power_peaking",
              "graphite_life", "fissile_kg", "leakage")


def build_design(p: dict) -> MSRDesign:
    uf4 = p.get("uf4_pct", 0.25) / 100.0
    thf4 = p.get("thf4_pct", 12.0) / 100.0
    bef2 = 0.16
    lif = 1.0 - uf4 - thf4 - bef2
    R = p.get("core_radius", 190.0)
    H = 2 * R * p.get("h_over_d", 1.0)
    li7 = 1.0 - p.get("li6_ppm", 50.0) * 1e-6
    T = p.get("T_fuel", 900.0)
    return MSRDesign(
        name="ML design", salt_composition={"LiF": lif, "BeF2": bef2, "ThF4": thf4, "UF4": uf4},
        uranium={"U233": 1.0}, li7_enrichment=li7,
        zones=[CoreZone("core", R, p.get("salt_fraction", 0.15), p.get("channel_radius", 1.2),
                        "hex")],
        core_height=H, radial_reflector=p.get("reflector", 50.0),
        axial_reflector=p.get("reflector", 50.0), downcomer=5.0, vessel=5.0,
        lower_plenum=15.0, upper_plenum=15.0, power_MW=1000.0, T_fuel=T, T_graphite=T + 20.0,
        flow_rate_m3s=2.0, external_loop_volume_m3=10.0)


def _lattice_alpha(design: MSRDesign, dT: float = 25.0, B2: Optional[float] = None) -> float:
    z = design.zones[0]
    cell = CellGeometry(z.channel_radius, z.salt_fraction, z.lattice)
    mod = graphite(design.graphite_density, design.T_graphite)
    s0 = design.fuel_salt()
    B2 = B2 if B2 is not None else _bare_B2(design)
    ks = []
    for sgn in (-1, 1):
        s = salt_at_temperature(s0, design.T_fuel + sgn * dT)
        r = run_lattice(s, mod, cell)
        ks.append(fundamental_mode(r.hom, B2)[0])
    return (1.0 / ks[0] - 1.0 / ks[1]) * 1e5 / (2 * dT)


def _bare_B2(design: MSRDesign, savings: float = 10.0) -> float:
    R = design.core_radius + savings
    H = design.core_height + 2 * savings
    return (2.405 / R) ** 2 + (np.pi / H) ** 2


def evaluate_hf(p: dict, return_field: bool = False, few_groups: str = "6g",
                mesh_size: float = 10.0) -> dict:
    d = build_design(p)
    r = solve_core(d, few_groups, mesh_size, adjoint=False, n_modes=1)
    s = r.summary()
    inv = d.inventory_kg()
    out = {"k_eff": r.k_eff, "conversion_ratio": r.conversion_ratio,
           "alpha_fuel": _lattice_alpha(d), "power_peaking": s["power_peaking"],
           "graphite_life": s["graphite_lifetime_years"],
           "fissile_kg": inv.get("U233", 0.0), "leakage": s["core_leakage_fraction"],
           "k_inf": r.k_inf_zones[0]}
    if return_field:
        out["power_map"] = normalized_power_map(r)
    return out


def evaluate_lf(p: dict) -> dict:
    d = build_design(p)
    z = d.zones[0]
    r = run_lattice(d.fuel_salt(), graphite(d.graphite_density, d.T_graphite),
                    CellGeometry(z.channel_radius, z.salt_fraction, z.lattice))
    k = fundamental_mode(r.hom, _bare_B2(d))[0]
    return {"k_eff": k, "k_inf": r.k_inf,
            "conversion_ratio": conversion_ratio(r.fuel_macro, r.phi_fuel * r.V_fuel)}


def normalized_power_map(core_result, nr: int = 20, nz: int = 30) -> np.ndarray:
    """Core-zone power density on a fixed (r/R, z/H) grid, normalized to mean 1."""
    d = core_result.diffusion
    mesh = d.mesh
    q = d.power_density()
    des = core_result.design
    z0 = des.axial_reflector + des.lower_plenum
    rr = (np.arange(nr) + 0.5) / nr * des.core_radius
    zz = z0 + (np.arange(nz) + 0.5) / nz * des.core_height
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((mesh.rc, mesh.zc), q, bounds_error=False, fill_value=None)
    R, Z = np.meshgrid(rr, zz, indexing="ij")
    m = f(np.column_stack([R.ravel(), Z.ravel()])).reshape(nr, nz)
    w = (2 * np.arange(nr) + 1)[:, None] * np.ones((1, nz))   # annulus volume weights
    return (m / ((m * w).sum() / w.sum())).ravel()


def _safe(fn, p):
    try:
        return fn(p)
    except Exception as exc:  # pragma: no cover - keep batch running
        return {"error": str(exc)}


def run_batch(fn: Callable, params: list, n_jobs: Optional[int] = None) -> list:
    """Evaluate ``fn`` over a list of parameter dicts (process-parallel)."""
    n_jobs = n_jobs or min(4, os.cpu_count() or 1)
    if n_jobs <= 1 or len(params) < 4:
        return [_safe(fn, p) for p in params]
    from functools import partial
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(partial(_safe, fn), params, chunksize=max(1, len(params) // (4 * n_jobs))))


def to_arrays(results: list, outputs) -> dict:
    return {o: np.array([r.get(o, np.nan) if "error" not in r else np.nan for r in results])
            for o in outputs}


def evaluate_hf_field(p: dict) -> dict:
    """High-fidelity evaluation that also returns the normalized power map."""
    return evaluate_hf(p, return_field=True)


def hf_k_oracle(X) -> np.ndarray:
    """Vectorized k_eff oracle (parallel physics) for active learning / BO."""
    res = run_batch(evaluate_hf, MSR_SPACE.as_dicts(X))
    return np.array([r.get("k_eff", np.nan) for r in res])


def lf_k_oracle(X) -> np.ndarray:
    """Vectorized low-fidelity k_eff (serial; ~10 ms per design)."""
    return np.array([evaluate_lf(p)["k_eff"] for p in MSR_SPACE.as_dicts(X)])
