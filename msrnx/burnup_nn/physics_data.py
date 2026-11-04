"""Built-in training data: MSR-NX coupled transport-depletion runs.

Each case is one design / operating point of a single-fluid, graphite-moderated
thorium MSR (LiF-BeF2-ThF4-UF4 with U-233 start-up fuel) depleted at constant
power with online processing (He sparging of noble gases, rare-earth
extraction, Pa-233 isolation in a decay tank that returns U-233).  The CE/CM
predictor-corrector with CRAM-16 of :mod:`msrnx.depletion` gives the burnup
trajectory; every time point becomes one row of a :class:`BurnupDataset`.

The generator is a *starting point*: replace or augment its output with data
from your own codes, or pass your own ``space`` and ``builder`` to generate
physics data for another design family (LEU, TRU, chloride fast, ...).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..constants import SECONDS_PER_DAY, SECONDS_PER_YEAR
from ..ml.design_space import DesignSpace, Parameter
from .dataset import BurnupDataset

BURNUP_SPACE = DesignSpace([
    Parameter("uf4_pct", 0.20, 0.60, 0.35, unit="mol%", label="UF4 (U-233) [mol%]"),
    Parameter("thf4_pct", 6.0, 14.0, 10.0, unit="mol%", label="ThF4 [mol%]"),
    Parameter("salt_fraction", 0.10, 0.30, 0.18, label="Salt volume fraction"),
    Parameter("power_MW", 400.0, 2000.0, 1000.0, unit="MW", label="Thermal power [MW]"),
    Parameter("re_removal_days", 20.0, 2000.0, 200.0, log=True, unit="d",
              label="Rare-earth removal time [d]"),
    Parameter("pa_removal_days", 3.0, 3000.0, 100.0, log=True, unit="d",
              label="Pa removal time [d]"),
    Parameter("xe_removal_s", 20.0, 20000.0, 200.0, log=True, unit="s",
              label="Xe sparging time [s]"),
])

# fine early steps resolve Xe-135 (hours), Pm/Sm-149 (days) and Pa-233 (weeks)
DEFAULT_STEPS_DAYS = (0.5, 0.5, 1, 2, 4, 7, 15, 30, 30, 60, 60, 90, 90, 120, 120,
                      180, 180, 180, 180, 240, 240, 300, 300, 360, 360)       # 3150 days
COARSE_STEPS_DAYS = (30, 60, 90, 180, 180, 300, 300, 360, 360, 360, 360, 360)  # 2940 days

TRACKED_NUCLIDES = ("Th232", "Pa233", "U232", "U233", "U234", "U235", "U236", "Np237",
                    "Pu238", "Pu239", "Xe135", "Sm149")

# fixed geometry of the default design family (MSR_SPACE nominal values)
FIXED = dict(core_radius=190.0, h_over_d=1.0, reflector=50.0, channel_radius=1.2,
             T_fuel=900.0, li6_ppm=50.0)


def build_case(p: dict):
    """Default design family: parameters -> (MSRDesign, Reprocessing)."""
    from ..depletion import Reprocessing
    from ..ml.physics import build_design
    q = dict(FIXED)
    q.update({k: v for k, v in p.items() if k in ("salt_fraction", "uf4_pct", "thf4_pct",
                                                    "core_radius", "h_over_d", "reflector",
                                                    "channel_radius", "T_fuel", "li6_ppm")})
    design = build_design(q).copy(power_MW=float(p.get("power_MW", 1000.0)))
    s = 1.0 / SECONDS_PER_DAY
    rep = Reprocessing({"noble_gas": float(p.get("xe_removal_s", 200.0)) * s,
                        "noble_metal": 50.0 * s,
                        "rare_earth": float(p.get("re_removal_days", 200.0)),
                        "Pa": float(p.get("pa_removal_days", 100.0)),
                        "halogen": 60.0, "tritium": 1.0, "salt_seeker": 3435.0},
                       pa_decay_tank=True, name="online processing")
    return design, rep


def depletion_frame(res, params: dict, case_id: str, with_feed: bool = False) -> pd.DataFrame:
    """Convert a :class:`DepletionResult` into long-format rows."""
    n = len(res.time_days)
    out = {"case_id": [case_id] * n, "time_days": res.time_days}
    for k, v in params.items():
        out[k] = np.full(n, float(v))
    out["burnup_MWd_kgHM"] = res.burnup_MWd_kgHM
    out["k_eff"] = res.k_eff
    if with_feed:
        out["k_bos"] = res.k_bos
        out["fissile_feed_kg"] = res.fissile_feed_kg
        out["fertile_feed_kg"] = res.fertile_feed_kg
    out["breeding_ratio"] = res.breeding_ratio
    out["flux_n_cm2_s"] = res.flux
    for nuc in TRACKED_NUCLIDES:
        if nuc in res.names:
            out[f"{nuc}_kg"] = res.inventory_kg(nuc)
    out["fissile_kg"] = res.fissile_inventory_kg()
    pf = res.poison_fraction
    out["Xe135_abs_pct"] = 100.0 * pf["Xe135"]
    out["Sm149_abs_pct"] = 100.0 * pf["Sm149"]
    out["Pa233_abs_pct"] = 100.0 * pf["Pa233"]
    out["FP_abs_pct"] = 100.0 * sum(pf[k] for k in ("Xe135", "Sm149", "LFP_RE", "LFP_NM",
                                                     "LFP_SS", "LFP_NG"))
    lam_t = np.log(2.0) / (12.32 * SECONDS_PER_YEAR)
    out["tritium_Ci_day"] = res.tritium_rate * 86400.0 * lam_t / 3.7e10
    return pd.DataFrame(out)


def run_case(p: dict, steps_days=DEFAULT_STEPS_DAYS, transport: str = "lattice",
             feed=None, case_id: str = "case", builder: Optional[Callable] = None) -> pd.DataFrame:
    """One depletion trajectory.

    transport  ``"lattice"`` (2D-calibrated lattice/B1, default, ~1 s per case),
               ``"lattice-uncalibrated"`` (cheapest) or ``"core"`` (full 2D R-Z
               diffusion at every evaluation, ~10x slower, higher fidelity)
    feed       optional :class:`msrnx.depletion.FeedControl` (criticality-
               controlled refuelling); default: fixed initial loading
    """
    from ..depletion import CoreTransport, DepletionSolver, LatticeTransport
    design, rep = (builder or build_case)(p)
    if transport == "core":
        tm = CoreTransport(design, "6g", 10.0)
    elif transport == "lattice-uncalibrated":
        tm = LatticeTransport(design, calibrate=False)
    elif transport == "lattice":
        tm = LatticeTransport(design, calibrate=True)
    else:
        raise ValueError(f"unknown transport '{transport}'")
    res = DepletionSolver(design, reprocessing=rep, feed=feed, transport_model=tm).run(
        list(steps_days))
    return depletion_frame(res, p, case_id, with_feed=feed is not None)


def _worker(args):
    case_id, p, steps, transport, feed, builder = args
    try:
        return run_case(p, steps, transport, feed, case_id, builder), None
    except Exception as exc:  # pragma: no cover - report and keep going
        return None, f"{case_id}: {type(exc).__name__}: {exc}"


def generate_physics_dataset(n_cases: int = 300, space: DesignSpace = BURNUP_SPACE,
                             steps_days=DEFAULT_STEPS_DAYS, seed: int = 0,
                             method: str = "lhs", transport: str = "lattice", feed=None,
                             builder: Optional[Callable] = None, n_jobs: Optional[int] = None,
                             X: Optional[np.ndarray] = None, case_prefix: str = "msrnx",
                             source: Optional[str] = None, out_csv: Optional[str] = None,
                             verbose: bool = True) -> BurnupDataset:
    """Sample ``n_cases`` designs from ``space`` and deplete each one.

    ``X`` (n, dim) overrides the sampling.  ``builder(p) -> (design, reprocessing)``
    must be an importable module-level function when ``n_jobs > 1``.
    Returns a :class:`BurnupDataset` (and writes ``out_csv`` if given).
    """
    if X is None:
        X = space.sample(n_cases, method, seed)
    params = space.as_dicts(X)
    n_jobs = n_jobs or min(4, os.cpu_count() or 1)
    jobs = [(f"{case_prefix}_{i:04d}", p, tuple(steps_days), transport, feed, builder)
            for i, p in enumerate(params)]
    t0 = time.time()
    if verbose:
        print(f"[burnup_nn] depleting {len(jobs)} cases ({transport} transport, "
              f"{len(steps_days)} steps to {sum(steps_days):.0f} d, {n_jobs} processes) ...")
    if n_jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_worker, jobs, chunksize=max(1, len(jobs) // (8 * n_jobs))))
    else:
        results = [_worker(j) for j in jobs]
    frames = [f for f, e in results if f is not None]
    errors = [e for f, e in results if e is not None]
    if errors and verbose:
        print(f"[burnup_nn] {len(errors)} case(s) failed, e.g. {errors[0]}")
    if not frames:
        raise RuntimeError("all depletion cases failed")
    ds = BurnupDataset(pd.concat(frames, ignore_index=True), "case_id", "time_days",
                       source or f"msrnx-{transport}")
    if verbose:
        print(f"[burnup_nn] {ds.n_cases} cases / {len(ds)} rows in {time.time() - t0:.1f} s")
    if out_csv:
        ds.to_csv(out_csv)
        if verbose:
            print(f"[burnup_nn] written {out_csv}")
    return ds
