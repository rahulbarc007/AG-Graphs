"""Synthetic continuous-energy (pointwise) cross sections.

For each nuclide the pointwise cross sections are assembled from

* a 1/v background (bound levels) fixing the 2200 m/s values,
* explicit single-level Breit-Wigner resonances,
* a statistical resonance ladder (Wigner spacings, Porter-Thomas widths)
  continued by its smooth average above the explicit cut-off,
* a smooth resonance-integral band for nuclides without resonance data,
* smooth fast-range shapes (capture, fission, threshold reactions).

Doppler broadening is exact within the psi-chi formalism using the Faddeeva
function ``w(z)`` (``scipy.special.wofz``).  The amplitudes are calibrated
so that the 0.0253 eV cross sections and the infinite-dilution resonance
integrals (0.5 eV cut-off) reproduce the database targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import wofz

from ..constants import FOUR_PI_LAMBDA2_E, K_BOLTZMANN_EV, E_THERMAL_EV, T_ROOM_K
from ..data.nuclides import Nuclide, Resonance

E_MIN, E_MAX = 1.0e-5, 2.0e7
E_FAST = 24788.0          # start of smooth fast range (= a FINE45 boundary)
RI_CUT = 0.5              # cadmium cut-off for resonance integrals

_OFFSETS_SPARSE = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.5, 7.0,
                            10.0, 15.0, 25.0, 40.0, 70.0, 120.0])
_OFFSETS_DENSE = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])


def psi_chi(E, E0: float, Gamma: float, awr: float, T: float):
    """Doppler-broadened symmetric (psi) and asymmetric (chi) line shapes."""
    x = 2.0 * (E - E0) / Gamma
    if T <= 0.0:
        d = 1.0 / (1.0 + x * x)
        return d, x * d
    Delta = np.sqrt(4.0 * E0 * K_BOLTZMANN_EV * T / awr)
    xi = Gamma / Delta
    w = wofz(0.5 * xi * (x + 1j))
    fac = 0.5 * xi * np.sqrt(np.pi)
    return fac * w.real, fac * w.imag


def make_ladder(nuc: Nuclide):
    """Generate the statistical ladder resonances and the smooth-region data."""
    L = nuc.ladder
    if L is None:
        return [], None
    rng = np.random.default_rng(L.seed)
    I = L.target_spin
    spins = [0.5] if I == 0 else [I - 0.5, I + 0.5]
    weights = np.array([2 * J + 1 for J in spins], dtype=float)
    max_res = 1600 if L.D > 5.0 else 600
    per_spin = (max_res * weights / weights.sum()).astype(int)
    res = []
    spin_info = []
    E_cut = L.E_hi
    for J, w, nmax in zip(spins, weights, per_spin):
        g = (2 * J + 1) / (2 * (2 * I + 1))
        DJ = L.D * weights.sum() / w
        spin_info.append((g, DJ))
        E = L.E_lo + rng.uniform(0.0, DJ)
        n = 0
        while E < L.E_hi and n < nmax:
            gn0 = L.S0 * DJ * rng.chisquare(1)
            gf = L.gf * rng.chisquare(L.nu_f) / L.nu_f if L.gf > 0 else 0.0
            res.append(Resonance(E, gn0 * np.sqrt(E), L.gg, gf, g))
            E += DJ * np.sqrt(-4.0 / np.pi * np.log(rng.uniform(1e-12, 1.0)))
            n += 1
        E_cut = min(E_cut, E)
    res.sort(key=lambda r: r.E0)
    if E_cut >= L.E_hi * 0.999:
        smooth = None
    else:
        # smooth statistical average from E_cut to E_hi (width-fluctuation averaged)
        rng2 = np.random.default_rng(L.seed + 7)
        samples = []
        for (g, DJ) in spin_info:
            gn0 = L.S0 * DJ * rng2.chisquare(1, 4000)
            gf = (L.gf * rng2.chisquare(L.nu_f, 4000) / L.nu_f) if L.gf > 0 else np.zeros(4000)
            samples.append((g, DJ, gn0, gf))
        smooth = dict(E_cut=E_cut, E_hi=L.E_hi, samples=samples, gg=L.gg)
        res = [r for r in res if r.E0 < E_cut]
    return res, smooth


def _smooth_ladder_xs(E, smooth, awr):
    """Average capture / fission cross sections of the statistical region."""
    cap = np.zeros_like(E)
    fis = np.zeros_like(E)
    if smooth is None:
        return cap, fis
    m = (E >= smooth["E_cut"]) & (E < smooth["E_hi"])
    if not np.any(m):
        return cap, fis
    Em = E[m]
    # evaluate width averages on a coarse grid and interpolate
    Ec = np.geomspace(smooth["E_cut"], smooth["E_hi"], 40)
    cap_c = np.zeros_like(Ec)
    fis_c = np.zeros_like(Ec)
    pref = 0.5 * np.pi * FOUR_PI_LAMBDA2_E * ((awr + 1.0) / awr) ** 2
    for g, DJ, gn0, gf in smooth["samples"]:
        for k, e in enumerate(Ec):
            gn = gn0 * np.sqrt(e)
            Gam = gn + smooth["gg"] + gf
            cap_c[k] += pref * g / DJ * np.mean(gn * smooth["gg"] / Gam) / e
            fis_c[k] += pref * g / DJ * np.mean(gn * gf / Gam) / e
    cap[m] = np.exp(np.interp(np.log(Em), np.log(Ec), np.log(cap_c + 1e-300)))
    if np.any(fis_c > 0):
        fis[m] = np.exp(np.interp(np.log(Em), np.log(Ec), np.log(fis_c + 1e-300)))
    return cap, fis


def energy_grid(nuc: Nuclide, ladder, extra_points=None) -> np.ndarray:
    pts = [np.geomspace(E_MIN, E_MAX, int(120 * np.log10(E_MAX / E_MIN)) + 1),
           np.geomspace(E_MIN, 10.0, 500)]
    for r in nuc.resonances:
        hw = 0.5 * (r.gn + r.gg + r.gf) + np.sqrt(4 * r.E0 * K_BOLTZMANN_EV * T_ROOM_K / nuc.awr)
        off = np.concatenate([-_OFFSETS_SPARSE[::-1], _OFFSETS_SPARSE[1:]])
        pts.append(r.E0 + hw * off)
    dense = nuc.ladder is not None and nuc.ladder.D < 5.0
    base_off = _OFFSETS_DENSE if dense else _OFFSETS_SPARSE
    off = np.concatenate([-base_off[::-1], base_off[1:]])
    for r in ladder:
        hw = 0.5 * (r.gn + r.gg + r.gf) + np.sqrt(4 * r.E0 * K_BOLTZMANN_EV * T_ROOM_K / nuc.awr)
        pts.append(r.E0 + hw * off)
    if extra_points is not None:
        pts.append(np.asarray(extra_points))
    E = np.concatenate(pts)
    E = E[(E >= E_MIN) & (E <= E_MAX)]
    E = np.unique(np.round(E, decimals=12))
    # remove points that are too close (relative 1e-9)
    keep = np.concatenate([[True], np.diff(E) / E[1:] > 1e-9])
    return E[keep]


def _resonance_sum(E, res_list, awr, T, sigma_p, windowed):
    cap = np.zeros_like(E)
    fis = np.zeros_like(E)
    el = np.zeros_like(E)
    kT = K_BOLTZMANN_EV * T
    for r in res_list:
        Gam = r.gn + r.gg + r.gf
        s0 = FOUR_PI_LAMBDA2_E * ((awr + 1.0) / awr) ** 2 * r.g * r.gn / (Gam * r.E0)
        if windowed:
            Delta = np.sqrt(4.0 * r.E0 * kT / awr)
            W = max(60.0 * Gam, 12.0 * Delta)
            lo = np.searchsorted(E, r.E0 - W)
            hi = np.searchsorted(E, r.E0 + W)
            sl = slice(lo, hi)
        else:
            sl = slice(0, len(E))
        Es = E[sl]
        if Es.size == 0:
            continue
        psi, chi = psi_chi(Es, r.E0, Gam, awr, T)
        amp = s0 * np.sqrt(r.E0 / Es) * psi
        cap[sl] += amp * (r.gg / Gam)
        if r.gf > 0:
            fis[sl] += amp * (r.gf / Gam)
        el[sl] += s0 * (r.gn / Gam) * psi + 2.0 * np.sqrt(s0 * sigma_p * r.g * r.gn / Gam) * chi
    return cap, fis, el


def _threshold(E, spec):
    if spec is None:
        return np.zeros_like(E)
    Ethr, smax, Ew = spec
    return np.where(E > Ethr, smax * (1.0 - np.exp(-(E - Ethr) / Ew)), 0.0)


def _potential(E, nuc: Nuclide):
    knots = [(E_MIN, nuc.sigma_p)] + [k for k in nuc.scatter_knots if k[0] > E_MIN]
    knots.append((E_MAX * 1.01, knots[-1][1]))
    kE = np.log([k[0] for k in knots])
    kS = np.array([k[1] for k in knots])
    order = np.argsort(kE)
    return np.interp(np.log(E), kE[order], kS[order])


def _fast_powerlaw(E, spec, flat_above_1MeV: bool):
    out = np.zeros_like(E)
    if spec is None:
        return out
    s25, s1 = spec
    p = np.log(s25 / s1) / np.log(1.0e6 / E_FAST)
    m = E >= E_FAST
    vals = s25 * (E_FAST / E[m]) ** p
    if flat_above_1MeV:
        vals = np.where(E[m] > 1.0e6, s1, vals)
    out[m] = vals
    return out


@dataclass
class Components:
    """Unit-amplitude pointwise components of one nuclide at one temperature."""
    E: np.ndarray
    one_over_v: np.ndarray
    expl: tuple
    lad: tuple
    lad_smooth: tuple
    smooth_band: np.ndarray
    fast_cap: np.ndarray
    fast_fis: np.ndarray
    thr_fis: np.ndarray
    potential: np.ndarray
    inelastic: np.ndarray
    n2n: np.ndarray
    nalpha_fast: np.ndarray
    tritium: np.ndarray


def components(nuc: Nuclide, T: float, E: np.ndarray, ladder, smooth) -> Components:
    awr = nuc.awr
    one_v = np.sqrt(E_THERMAL_EV / E)
    if nuc.bg_cutoff:
        Eb = nuc.bg_cutoff
        one_v = one_v * ((Eb + E_THERMAL_EV) / (E + Eb)) ** 2
    expl = _resonance_sum(E, nuc.resonances, awr, T, nuc.sigma_p, windowed=False)
    lad = _resonance_sum(E, ladder, awr, T, nuc.sigma_p, windowed=True)
    lad_s = _smooth_ladder_xs(E, smooth, awr)
    lo, hi = nuc.smooth_res
    band = np.where((E >= lo) & (E < hi), np.sqrt(1.0 / E), 0.0)
    norm = 2.0 * (lo ** -0.5 - hi ** -0.5)
    band = band / norm  # unit resonance integral
    fast_cap = _fast_powerlaw(E, nuc.fast_capture, flat_above_1MeV=False)
    fast_fis = _fast_powerlaw(E, nuc.fast_fission, flat_above_1MeV=True)
    if nuc.fission_threshold is not None:
        Em, plateau, w = nuc.fission_threshold
        f0 = 1.0 / (1.0 + np.exp(Em / w))          # logistic value at E = 0
        thr = plateau * np.maximum(1.0 / (1.0 + np.exp(-(E - Em) / w)) - f0, 0.0) / (1.0 - f0)
    else:
        thr = np.zeros_like(E)
    return Components(E, one_v, expl, lad, lad_s, band, fast_cap, fast_fis, thr,
                      _potential(E, nuc), _threshold(E, nuc.inelastic), _threshold(E, nuc.n2n),
                      _threshold(E, nuc.nalpha_fast), _threshold(E, nuc.tritium_fast))


@dataclass
class Calibration:
    bg_capture: float = 0.0
    bg_fission: float = 0.0
    s_explicit: float = 1.0
    c_capture: float = 1.0
    c_fission: float = 1.0
    c_band_capture: float = 0.0
    c_band_fission: float = 0.0
    report: dict = field(default_factory=dict)


def _ri(E, sigma):
    m = E >= RI_CUT
    return float(np.trapezoid(sigma[m], np.log(E[m])))


def _value_at(E, sigma, E0=E_THERMAL_EV):
    return float(np.interp(np.log(E0), np.log(E), sigma))


def calibrate(nuc: Nuclide, comp: Components) -> Calibration:
    """Fit amplitudes so thermal values and resonance integrals match targets."""
    E = comp.E
    cal = Calibration()
    eg, ef, _ = comp.expl
    lg = comp.lad[0] + comp.lad_smooth[0]
    lf = comp.lad[1] + comp.lad_smooth[1]
    has_lad = nuc.ladder is not None
    eg_th, ef_th = _value_at(E, eg), _value_at(E, ef)
    lg_th, lf_th = _value_at(E, lg), _value_at(E, lf)
    RI_1v = _ri(E, comp.one_over_v)
    RI_eg, RI_ef = _ri(E, eg), _ri(E, ef)
    RI_lg, RI_lf = _ri(E, lg), _ri(E, lf)
    RI_fast_g = _ri(E, comp.fast_cap)
    RI_fast_f = _ri(E, comp.fast_fis + comp.thr_fis)
    s_e, c_g, c_f = 1.0, (1.0 if has_lad else 0.0), (1.0 if has_lad else 0.0)
    bg_g = bg_f = 0.0
    cb_g = cb_f = 0.0
    for _ in range(50):
        # thermal overshoot: shrink explicit amplitude
        th_g = s_e * eg_th + c_g * lg_th
        th_f = s_e * ef_th + c_f * lf_th
        scale = 1.0
        if eg_th > 0 and th_g > nuc.capture_th:
            scale = min(scale, max((nuc.capture_th - c_g * lg_th) / (s_e * eg_th), 0.05))
        if ef_th > 0 and th_f > nuc.fission_th:
            scale = min(scale, max((nuc.fission_th - c_f * lf_th) / (s_e * ef_th), 0.05))
        s_e *= scale
        bg_g = max(nuc.capture_th - (s_e * eg_th + c_g * lg_th), 0.0)
        bg_f = max(nuc.fission_th - (s_e * ef_th + c_f * lf_th), 0.0)
        known_g = s_e * RI_eg + bg_g * RI_1v + RI_fast_g
        known_f = s_e * RI_ef + bg_f * RI_1v + RI_fast_f
        if nuc.RI_capture is not None:
            if has_lad and RI_lg > 0:
                c_g = max((nuc.RI_capture - known_g) / RI_lg, 0.0)
            elif not has_lad:
                cb_g = max(nuc.RI_capture - known_g, 0.0)
        if nuc.RI_fission is not None:
            if has_lad and RI_lf > 0:
                c_f = max((nuc.RI_fission - known_f) / RI_lf, 0.0)
            elif not has_lad:
                cb_f = max(nuc.RI_fission - known_f, 0.0)
    cal.bg_capture, cal.bg_fission, cal.s_explicit = bg_g, bg_f, s_e
    cal.c_capture, cal.c_fission = c_g, c_f
    cal.c_band_capture, cal.c_band_fission = cb_g, cb_f
    return cal


@dataclass
class PointwiseXS:
    name: str
    T: float
    E: np.ndarray
    capture: np.ndarray
    fission: np.ndarray
    elastic: np.ndarray
    nalpha: np.ndarray
    inelastic: np.ndarray
    n2n: np.ndarray
    tritium: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.elastic + self.capture + self.fission + self.nalpha + self.inelastic + self.n2n

    def thermal(self, reaction: str = "capture") -> float:
        return _value_at(self.E, getattr(self, reaction))

    def resonance_integral(self, reaction: str = "capture") -> float:
        return _ri(self.E, getattr(self, reaction))


def assemble(nuc: Nuclide, comp: Components, cal: Calibration, T: float) -> PointwiseXS:
    eg, ef, eel = comp.expl
    lg = comp.lad[0] + comp.lad_smooth[0]
    lf = comp.lad[1] + comp.lad_smooth[1]
    capture = (cal.bg_capture * comp.one_over_v + cal.s_explicit * eg + cal.c_capture * lg
               + cal.c_band_capture * comp.smooth_band + comp.fast_cap)
    fission = (cal.bg_fission * comp.one_over_v + cal.s_explicit * ef + cal.c_fission * lf
               + cal.c_band_fission * comp.smooth_band + comp.fast_fis + comp.thr_fis)
    elastic = comp.potential + cal.s_explicit * eel + comp.lad[2]
    elastic = np.maximum(elastic, 0.02 * nuc.sigma_p)
    nalpha = nuc.nalpha_th * comp.one_over_v + comp.nalpha_fast
    return PointwiseXS(nuc.name, T, comp.E, np.maximum(capture, 0.0), np.maximum(fission, 0.0),
                       elastic, nalpha, comp.inelastic, comp.n2n, comp.tritium)


class PointwiseNuclide:
    """Pointwise generator for one nuclide (grid, ladder and calibration cached)."""

    def __init__(self, nuc: Nuclide, extra_points=None):
        self.nuc = nuc
        self.ladder, self.smooth = make_ladder(nuc)
        self.E = energy_grid(nuc, self.ladder, extra_points)
        comp_ref = components(nuc, T_ROOM_K, self.E, self.ladder, self.smooth)
        self.cal = calibrate(nuc, comp_ref)
        ref = assemble(nuc, comp_ref, self.cal, T_ROOM_K)
        self.cal.report = {
            "capture_th": ref.thermal("capture"), "fission_th": ref.thermal("fission"),
            "RI_capture": ref.resonance_integral("capture"),
            "RI_fission": ref.resonance_integral("fission"),
            "n_ladder": len(self.ladder),
        }
        self._cache = {T_ROOM_K: ref}

    @property
    def has_resonances(self) -> bool:
        return bool(self.nuc.resonances) or bool(self.ladder)

    def at(self, T: float) -> PointwiseXS:
        if T not in self._cache:
            comp = components(self.nuc, T, self.E, self.ladder, self.smooth)
            self._cache[T] = assemble(self.nuc, comp, self.cal, T)
        return self._cache[T]
