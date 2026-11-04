"""Built-in nuclide database for MSR-NX.

The values below are *representative* evaluated-library-like parameters
(2200 m/s cross sections, infinite-dilution resonance integrals above the
0.5 eV cadmium cut-off, low-lying SLBW resonance parameters, average
resonance statistics, fission and decay data).  They are compiled from
standard references (BNL Atlas of Neutron Resonances / Chart of Nuclides
level of accuracy) and are good enough to reproduce the *physics trends*
of molten salt reactor neutronics: spectrum, self-shielding, Doppler
broadening, breeding, poisoning and kinetics.

They are NOT a substitute for a licensed, validated nuclear data library.
For licensing-grade work, replace the multigroup library with one
generated from ENDF/B-VIII.0 or JEFF-3.3 (see ``msrnx.xs.library``, which
accepts externally supplied group constants).

Units: energies in eV, cross sections in barns, half-lives in seconds,
widths in eV.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..constants import SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_YEAR, awr_from_mass

MINUTE = 60.0


@dataclass(frozen=True)
class Resonance:
    """Single-level Breit-Wigner resonance (s-wave)."""
    E0: float          # resonance energy (eV)
    gn: float          # neutron width at E0 (eV)
    gg: float          # radiative width (eV)
    gf: float = 0.0    # fission width (eV)
    g: float = 1.0     # statistical spin factor (2J+1)/(2(2I+1))


@dataclass(frozen=True)
class Ladder:
    """Average resonance parameters used to synthesise a statistical ladder.

    Resonance energies follow a Wigner spacing distribution, reduced neutron
    widths a Porter-Thomas distribution and fission widths a chi-squared
    distribution with ``nu_f`` degrees of freedom.  The ladder amplitude is
    later calibrated so that the infinite-dilution resonance integrals match
    the database targets.
    """
    D: float                 # mean observed s-wave level spacing (eV)
    gg: float                # mean radiative width (eV)
    S0: float                # s-wave neutron strength function
    gf: float = 0.0          # mean fission width (eV)
    nu_f: int = 2            # fission-width degrees of freedom
    target_spin: float = 0.0 # spin I of target nucleus
    E_lo: float = 1.0        # ladder start (eV)
    E_hi: float = 24788.0    # ladder end (eV), top of resolved region
    seed: int = 1


@dataclass(frozen=True)
class Nuclide:
    name: str
    Z: int
    A: int
    mass: float                 # atomic mass (u)
    element: str
    chem: str                   # chemistry/processing class
    sigma_p: float              # potential / free-atom elastic (b)
    capture_th: float = 0.0     # (n,gamma) at 0.0253 eV (b)
    fission_th: float = 0.0     # (n,f) at 0.0253 eV (b)
    nalpha_th: float = 0.0      # (n,alpha)/(n,t)/(n,p) at 0.0253 eV, 1/v (b)
    RI_capture: Optional[float] = None
    RI_fission: Optional[float] = None
    resonances: tuple = ()
    ladder: Optional[Ladder] = None
    fast_capture: Optional[tuple] = None       # (sigma @24.8 keV, sigma @1 MeV)
    fast_fission: Optional[tuple] = None       # fissile: (sigma @24.8 keV, sigma @1 MeV)
    fission_threshold: Optional[tuple] = None  # (E_mid eV, plateau b, width eV)
    inelastic: Optional[tuple] = None          # (E_thr eV, sigma_max b, E_w eV)
    n2n: Optional[tuple] = None                # (E_thr eV, sigma_max b, E_w eV)
    nalpha_fast: Optional[tuple] = None        # (E_thr eV, sigma_max b, E_w eV)
    tritium_fast: Optional[tuple] = None       # tally only, e.g. Li-7(n,n't)
    scatter_knots: tuple = ()                  # ((E, sigma), ...) elastic shape
    nu: tuple = (0.0, 0.0)                     # (nu at thermal, d nu/dE per MeV)
    watt: tuple = (0.988, 2.249)               # Watt a (MeV), b (1/MeV)
    kappa: float = 200.0                       # recoverable MeV per fission
    half_life: Optional[float] = None
    decay: tuple = ()                          # ((daughter, branching), ...)
    capture_product: Optional[str] = None      # 'self' = lumped (no transmutation)
    n2n_product: Optional[str] = None
    nalpha_product: Optional[str] = None
    smooth_res: tuple = (1.0, 24788.0)         # band for smooth RI when no ladder
    bg_cutoff: Optional[float] = None          # bound-level background fall-off (eV)

    @property
    def awr(self) -> float:
        return awr_from_mass(self.mass)

    @property
    def fissionable(self) -> bool:
        return (self.fission_th > 0.0 or self.fast_fission is not None
                or self.fission_threshold is not None)

    @property
    def is_actinide(self) -> bool:
        return self.Z >= 89

    @property
    def free_gas(self) -> bool:
        """Use the free-gas thermal scattering kernel (light nuclides)."""
        return self.awr < 40.0

    @property
    def decay_constant(self) -> float:
        import math
        return 0.0 if not self.half_life else math.log(2.0) / self.half_life


def _N(**kw) -> Nuclide:
    return Nuclide(**kw)


# --------------------------------------------------------------------------
# Light, carrier-salt, moderator and structural nuclides
# --------------------------------------------------------------------------
_LIGHT = [
    _N(name="Li6", Z=3, A=6, mass=6.0151223, element="Li", chem="carrier",
       sigma_p=0.75, capture_th=0.039, nalpha_th=940.0, nalpha_product="H3",
       scatter_knots=((1.0e5, 1.0), (1.0e7, 1.0))),
    _N(name="Li7", Z=3, A=7, mass=7.0160034, element="Li", chem="carrier",
       sigma_p=0.97, capture_th=0.0454, inelastic=(5.5e5, 0.35, 1.0e6),
       tritium_fast=(2.8e6, 0.35, 1.5e6),
       scatter_knots=((1.0e4, 0.97), (1.0e6, 1.5), (1.0e7, 1.0))),
    _N(name="Be9", Z=4, A=9, mass=9.0121831, element="Be", chem="carrier",
       sigma_p=6.15, capture_th=0.0076, n2n=(1.85e6, 0.55, 1.5e6),
       nalpha_fast=(0.67e6, 0.10, 1.0e6), nalpha_product="Li6",
       inelastic=(2.43e6, 0.20, 1.0e6),
       scatter_knots=((5.0e5, 6.0), (1.0e7, 1.6))),
    _N(name="B10", Z=5, A=10, mass=10.0129370, element="B", chem="absorber",
       sigma_p=2.23, capture_th=0.5, nalpha_th=3840.0, nalpha_product="Li7",
       scatter_knots=((1.0e5, 2.2), (1.0e7, 1.5))),
    _N(name="B11", Z=5, A=11, mass=11.0093054, element="B", chem="absorber",
       sigma_p=5.0, capture_th=0.0055,
       scatter_knots=((1.0e5, 4.8), (1.0e7, 1.4))),
    _N(name="C12", Z=6, A=12, mass=12.011, element="C", chem="moderator",
       sigma_p=4.74, capture_th=0.00353, inelastic=(4.8e6, 0.4, 1.0e6),
       scatter_knots=((1.0e4, 4.74), (1.0e7, 1.4))),
    _N(name="O16", Z=8, A=16, mass=15.9949146, element="O", chem="structural",
       sigma_p=3.76, capture_th=0.00019, inelastic=(6.4e6, 0.4, 1.0e6),
       scatter_knots=((1.0e5, 3.76), (1.0e7, 1.5))),
    _N(name="F19", Z=9, A=19, mass=18.9984032, element="F", chem="carrier",
       sigma_p=3.64, capture_th=0.0096, inelastic=(1.1e5, 0.7, 1.0e6),
       scatter_knots=((1.0e5, 3.8), (1.0e7, 2.0))),
    _N(name="Na23", Z=11, A=23, mass=22.9897693, element="Na", chem="carrier",
       sigma_p=3.28, capture_th=0.53, RI_capture=0.31, inelastic=(4.4e5, 0.8, 1.0e6),
       scatter_knots=((1.0e5, 3.3), (1.0e7, 2.2))),
    _N(name="Mg24", Z=12, A=24, mass=24.305, element="Mg", chem="carrier",
       sigma_p=3.71, capture_th=0.063, inelastic=(1.37e6, 0.6, 1.0e6),
       scatter_knots=((1.0e5, 3.7), (1.0e7, 2.0))),
    _N(name="Cl35", Z=17, A=35, mass=34.9688527, element="Cl", chem="carrier",
       sigma_p=21.0, capture_th=43.6, nalpha_th=0.489, inelastic=(1.22e6, 0.8, 1.0e6),
       scatter_knots=((1.0, 21.0), (1.0e2, 5.0), (1.0e7, 2.0))),
    _N(name="Cl37", Z=17, A=37, mass=36.9659026, element="Cl", chem="carrier",
       sigma_p=1.19, capture_th=0.433, inelastic=(1.73e6, 0.8, 1.0e6),
       scatter_knots=((1.0e3, 1.2), (1.0e5, 3.0), (1.0e7, 2.0))),
    _N(name="K39", Z=19, A=39, mass=39.0983, element="K", chem="carrier",
       sigma_p=1.96, capture_th=2.1, RI_capture=1.0, inelastic=(2.5e6, 0.6, 1.0e6),
       scatter_knots=((1.0e5, 2.0), (1.0e7, 2.5))),
    _N(name="Cr52", Z=24, A=52, mass=51.9961, element="Cr", chem="structural",
       sigma_p=3.49, capture_th=3.05, RI_capture=1.6, inelastic=(1.43e6, 1.0, 1.0e6),
       scatter_knots=((1.0e4, 3.5), (1.0e7, 2.5))),
    _N(name="Fe56", Z=26, A=56, mass=55.845, element="Fe", chem="structural",
       sigma_p=11.62, capture_th=2.56, RI_capture=1.4, inelastic=(0.85e6, 1.3, 1.0e6),
       scatter_knots=((1.0e4, 11.6), (1.0e6, 3.5), (1.0e7, 2.5))),
    _N(name="Ni58", Z=28, A=58, mass=58.6934, element="Ni", chem="structural",
       sigma_p=18.5, capture_th=4.49, RI_capture=2.1, inelastic=(1.45e6, 1.0, 1.0e6),
       scatter_knots=((1.0e4, 18.5), (1.0e6, 3.5), (1.0e7, 2.5))),
    _N(name="Zr90", Z=40, A=91, mass=91.224, element="Zr", chem="carrier",
       sigma_p=6.4, capture_th=0.185, RI_capture=1.0, inelastic=(0.92e6, 1.0, 1.0e6),
       scatter_knots=((1.0e5, 6.4), (1.0e7, 3.5))),
    _N(name="Mo96", Z=42, A=96, mass=95.95, element="Mo", chem="structural",
       sigma_p=5.71, capture_th=2.48, RI_capture=24.0, inelastic=(2.0e5, 1.5, 1.0e6),
       scatter_knots=((1.0e5, 5.7), (1.0e7, 3.5))),
    _N(name="H3", Z=1, A=3, mass=3.0160493, element="H", chem="tritium",
       sigma_p=1.0, capture_th=6e-6, half_life=12.32 * SECONDS_PER_YEAR),
]

# --------------------------------------------------------------------------
# Actinides
# --------------------------------------------------------------------------
_ACTINIDES = [
    _N(name="Th232", bg_cutoff=5.0, Z=90, A=232, mass=232.0380553, element="Th", chem="Th",
       sigma_p=12.0, capture_th=7.35, RI_capture=85.0,
       resonances=(Resonance(21.78, 2.04e-3, 26.0e-3), Resonance(23.45, 3.88e-3, 24.5e-3),
                   Resonance(59.50, 3.9e-3, 24.5e-3), Resonance(69.20, 43.0e-3, 24.5e-3),
                   Resonance(113.0, 13.0e-3, 24.5e-3), Resonance(120.8, 21.0e-3, 24.5e-3),
                   Resonance(128.2, 2.0e-3, 24.5e-3), Resonance(170.0, 51.0e-3, 24.5e-3)),
       ladder=Ladder(D=16.8, gg=0.0245, S0=0.84e-4, E_lo=180.0, seed=232),
       fast_capture=(0.55, 0.13), fission_threshold=(1.5e6, 0.14, 0.15e6),
       inelastic=(5.0e4, 2.5, 6.0e5), n2n=(6.44e6, 1.7, 1.2e6), n2n_product="Pa231",
       scatter_knots=((1.0e5, 12.0), (3.0e6, 4.8), (1.0e7, 4.5)),
       nu=(1.95, 0.135), watt=(1.0, 2.2), kappa=196.2,
       capture_product="Th233"),
    _N(name="Th233", Z=90, A=233, mass=233.0415818, element="Th", chem="Th",
       sigma_p=12.0, capture_th=1500.0,
       half_life=21.83 * MINUTE, decay=(("Pa233", 1.0),)),
    _N(name="Pa231", bg_cutoff=1.0, Z=91, A=231, mass=231.0358842, element="Pa", chem="Pa",
       sigma_p=10.0, capture_th=200.6, fission_th=0.020, RI_capture=480.0,
       ladder=Ladder(D=0.45, gg=0.040, S0=0.9e-4, target_spin=1.5, E_lo=0.6, seed=231),
       fast_capture=(1.0, 0.15), fission_threshold=(0.6e6, 1.1, 0.12e6),
       inelastic=(2.0e4, 2.0, 6.0e5), scatter_knots=((1.0e5, 10.0), (1.0e7, 4.5)),
       nu=(2.3, 0.13), capture_product="U232"),
    _N(name="Pa233", bg_cutoff=1.0, Z=91, A=233, mass=233.0402473, element="Pa", chem="Pa",
       sigma_p=10.0, capture_th=39.5, RI_capture=860.0,
       ladder=Ladder(D=0.45, gg=0.040, S0=0.9e-4, target_spin=1.5, E_lo=0.6, seed=233),
       fast_capture=(1.0, 0.15), fission_threshold=(0.8e6, 1.0, 0.12e6),
       inelastic=(2.0e4, 2.0, 6.0e5), scatter_knots=((1.0e5, 10.0), (1.0e7, 4.5)),
       nu=(2.3, 0.13), half_life=26.975 * SECONDS_PER_DAY, decay=(("U233", 1.0),),
       capture_product="U234"),
    _N(name="U232", bg_cutoff=2.0, Z=92, A=232, mass=232.0371562, element="U", chem="U",
       sigma_p=10.5, capture_th=73.1, fission_th=76.7, RI_capture=280.0, RI_fission=350.0,
       ladder=Ladder(D=4.0, gg=0.035, S0=1.0e-4, gf=0.4, nu_f=3, E_lo=4.0, seed=2232),
       fast_capture=(0.6, 0.06), fast_fission=(2.8, 2.0),
       inelastic=(5.0e4, 1.5, 6.0e5), scatter_knots=((1.0e5, 10.5), (1.0e7, 4.5)),
       nu=(3.13, 0.12), half_life=68.9 * SECONDS_PER_YEAR, decay=(("HA", 1.0),),
       capture_product="U233"),
    _N(name="U233", bg_cutoff=2.0, Z=92, A=233, mass=233.0396352, element="U", chem="U",
       sigma_p=10.5, capture_th=45.5, fission_th=531.2, RI_capture=140.0, RI_fission=764.0,
       ladder=Ladder(D=0.55, gg=0.040, S0=1.0e-4, gf=0.40, nu_f=3, target_spin=2.5,
                     E_lo=1.0, seed=233233),
       fast_capture=(0.45, 0.05), fast_fission=(3.3, 1.9),
       inelastic=(4.0e4, 1.4, 6.0e5), n2n=(5.8e6, 0.4, 1.0e6), n2n_product="U232",
       scatter_knots=((1.0e5, 10.5), (1.0e7, 4.5)),
       nu=(2.497, 0.105), watt=(0.977, 2.546), kappa=198.0, capture_product="U234"),
    _N(name="U234", bg_cutoff=5.0, Z=92, A=234, mass=234.0409521, element="U", chem="U",
       sigma_p=10.5, capture_th=100.1, fission_th=0.067, RI_capture=660.0,
       resonances=(Resonance(5.16, 3.7e-3, 26.0e-3),),
       ladder=Ladder(D=11.0, gg=0.027, S0=0.9e-4, E_lo=20.0, seed=234),
       fast_capture=(0.6, 0.12), fission_threshold=(0.5e6, 1.2, 0.10e6),
       inelastic=(4.0e4, 2.2, 6.0e5), scatter_knots=((1.0e5, 10.5), (1.0e7, 4.5)),
       nu=(2.36, 0.13), capture_product="U235"),
    _N(name="U235", bg_cutoff=2.0, Z=92, A=235, mass=235.0439299, element="U", chem="U",
       sigma_p=10.5, capture_th=98.7, fission_th=585.1, RI_capture=144.0, RI_fission=275.0,
       ladder=Ladder(D=0.45, gg=0.035, S0=1.0e-4, gf=0.12, nu_f=3, target_spin=3.5,
                     E_lo=1.0, seed=235235),
       fast_capture=(0.85, 0.11), fast_fission=(2.45, 1.22),
       inelastic=(1.3e4, 1.8, 6.0e5), n2n=(5.3e6, 0.6, 1.0e6), n2n_product="U234",
       scatter_knots=((1.0e5, 10.5), (1.0e7, 4.5)),
       nu=(2.4355, 0.120), watt=(0.988, 2.249), kappa=202.5, capture_product="U236"),
    _N(name="U236", bg_cutoff=5.0, Z=92, A=236, mass=236.0455680, element="U", chem="U",
       sigma_p=10.5, capture_th=5.1, fission_th=0.05, RI_capture=360.0,
       resonances=(Resonance(5.45, 2.0e-3, 23.0e-3),),
       ladder=Ladder(D=15.0, gg=0.023, S0=1.0e-4, E_lo=20.0, seed=236),
       fast_capture=(0.5, 0.12), fission_threshold=(0.8e6, 0.75, 0.12e6),
       inelastic=(4.5e4, 2.3, 6.0e5), scatter_knots=((1.0e5, 10.5), (1.0e7, 4.5)),
       nu=(2.3, 0.14), capture_product="U237"),
    _N(name="U237", Z=92, A=237, mass=237.0487302, element="U", chem="U",
       sigma_p=10.5, capture_th=450.0, fission_th=1.7, RI_capture=1200.0,
       fission_threshold=(0.5e6, 0.6, 0.12e6), nu=(2.4, 0.13),
       half_life=6.752 * SECONDS_PER_DAY, decay=(("Np237", 1.0),), capture_product="U238"),
    _N(name="U238", bg_cutoff=10.0, Z=92, A=238, mass=238.0507882, element="U", chem="U",
       sigma_p=11.29, capture_th=2.683, RI_capture=280.0,
       resonances=(Resonance(6.674, 1.493e-3, 23.0e-3), Resonance(20.87, 10.26e-3, 22.9e-3),
                   Resonance(36.68, 34.13e-3, 22.9e-3), Resonance(66.03, 24.6e-3, 23.3e-3),
                   Resonance(80.75, 1.87e-3, 23.0e-3), Resonance(102.56, 71.7e-3, 25.0e-3),
                   Resonance(116.9, 25.5e-3, 22.0e-3), Resonance(165.3, 3.0e-3, 23.0e-3),
                   Resonance(189.7, 167.0e-3, 23.0e-3), Resonance(208.5, 52.0e-3, 23.0e-3)),
       ladder=Ladder(D=20.9, gg=0.0233, S0=1.03e-4, E_lo=220.0, seed=238),
       fast_capture=(0.48, 0.11), fission_threshold=(1.6e6, 0.55, 0.15e6),
       inelastic=(4.5e4, 2.6, 5.0e5), n2n=(6.18e6, 1.5, 1.0e6), n2n_product="U237",
       scatter_knots=((1.0e5, 11.29), (3.0e6, 4.8), (1.0e7, 4.5)),
       nu=(2.30, 0.150), watt=(0.881, 3.400), kappa=205.9, capture_product="U239"),
    _N(name="U239", Z=92, A=239, mass=239.0542933, element="U", chem="U",
       sigma_p=10.5, capture_th=22.0,
       half_life=23.45 * MINUTE, decay=(("Np239", 1.0),)),
    _N(name="Np237", bg_cutoff=1.0, Z=93, A=237, mass=237.0481734, element="Np", chem="Np",
       sigma_p=10.5, capture_th=175.9, fission_th=0.020, RI_capture=660.0,
       ladder=Ladder(D=0.6, gg=0.040, S0=1.0e-4, target_spin=2.5, E_lo=0.4, seed=237),
       fast_capture=(1.5, 0.15), fission_threshold=(0.5e6, 1.5, 0.10e6),
       inelastic=(3.0e4, 1.8, 6.0e5), scatter_knots=((1.0e5, 10.5), (1.0e7, 4.5)),
       nu=(2.6, 0.14), capture_product="Pu238"),
    _N(name="Np239", Z=93, A=239, mass=239.0529390, element="Np", chem="Np",
       sigma_p=10.5, capture_th=32.0, RI_capture=400.0,
       half_life=2.356 * SECONDS_PER_DAY, decay=(("Pu239", 1.0),), capture_product="Pu240"),
    _N(name="Pu238", bg_cutoff=1.0, Z=94, A=238, mass=238.0495599, element="Pu", chem="Pu",
       sigma_p=10.5, capture_th=412.0, fission_th=17.9, RI_capture=150.0, RI_fission=30.0,
       fast_capture=(0.8, 0.12), fast_fission=(1.2, 2.0), nu=(2.89, 0.15),
       half_life=87.7 * SECONDS_PER_YEAR, decay=(("U234", 1.0),), capture_product="Pu239"),
    _N(name="Pu239", bg_cutoff=1.0, Z=94, A=239, mass=239.0521634, element="Pu", chem="Pu",
       sigma_p=10.0, capture_th=270.7, fission_th=747.4, RI_capture=200.0, RI_fission=303.0,
       resonances=(Resonance(0.2956, 0.0789e-3, 39.5e-3, 59.6e-3, 0.75),),
       ladder=Ladder(D=2.3, gg=0.040, S0=1.2e-4, gf=0.15, nu_f=1, target_spin=0.5,
                     E_lo=1.0, seed=239239),
       fast_capture=(1.4, 0.05), fast_fission=(1.75, 1.73),
       inelastic=(8.0e3, 1.5, 6.0e5), n2n=(5.7e6, 0.3, 1.0e6), n2n_product="Pu238",
       scatter_knots=((1.0e5, 10.0), (1.0e7, 4.5)),
       nu=(2.879, 0.148), watt=(0.966, 2.842), kappa=207.0, capture_product="Pu240"),
    _N(name="Pu240", Z=94, A=240, mass=240.0538135, element="Pu", chem="Pu",
       sigma_p=10.0, capture_th=289.5, fission_th=0.056, RI_capture=8500.0,
       resonances=(Resonance(1.056, 2.46e-3, 31.6e-3),),
       ladder=Ladder(D=13.0, gg=0.031, S0=1.0e-4, E_lo=20.0, seed=240),
       fast_capture=(0.9, 0.18), fission_threshold=(0.6e6, 1.5, 0.10e6),
       inelastic=(4.3e4, 2.0, 6.0e5), scatter_knots=((1.0e5, 10.0), (1.0e7, 4.5)),
       nu=(2.80, 0.15), capture_product="Pu241"),
    _N(name="Pu241", bg_cutoff=1.0, Z=94, A=241, mass=241.0568515, element="Pu", chem="Pu",
       sigma_p=10.0, capture_th=361.1, fission_th=1012.0, RI_capture=162.0, RI_fission=570.0,
       ladder=Ladder(D=0.9, gg=0.040, S0=1.0e-4, gf=0.30, nu_f=3, target_spin=2.5,
                     E_lo=0.3, seed=241241),
       fast_capture=(0.9, 0.08), fast_fission=(2.9, 1.65),
       inelastic=(1.0e4, 1.6, 6.0e5), scatter_knots=((1.0e5, 10.0), (1.0e7, 4.5)),
       nu=(2.945, 0.150), watt=(0.966, 2.842), kappa=210.0,
       half_life=14.29 * SECONDS_PER_YEAR, decay=(("Am241", 1.0),), capture_product="Pu242"),
    _N(name="Pu242", bg_cutoff=2.0, Z=94, A=242, mass=242.0587426, element="Pu", chem="Pu",
       sigma_p=10.0, capture_th=18.5, RI_capture=1130.0,
       resonances=(Resonance(2.677, 2.0e-3, 26.0e-3),),
       ladder=Ladder(D=15.0, gg=0.023, S0=0.9e-4, E_lo=20.0, seed=242),
       fast_capture=(0.7, 0.12), fission_threshold=(0.85e6, 1.35, 0.12e6),
       inelastic=(4.5e4, 2.0, 6.0e5), scatter_knots=((1.0e5, 10.0), (1.0e7, 4.5)),
       nu=(2.75, 0.15), capture_product="Am243"),
    _N(name="Am241", bg_cutoff=0.5, Z=95, A=241, mass=241.0568291, element="Am", chem="Am",
       sigma_p=10.0, capture_th=684.0, fission_th=3.1, RI_capture=1500.0,
       ladder=Ladder(D=0.6, gg=0.045, S0=1.0e-4, target_spin=2.5, E_lo=0.3, seed=95241),
       fast_capture=(1.8, 0.2), fission_threshold=(0.45e6, 1.8, 0.10e6),
       scatter_knots=((1.0e5, 10.0), (1.0e7, 4.5)), nu=(3.08, 0.13),
       half_life=432.6 * SECONDS_PER_YEAR, decay=(("Np237", 1.0),), capture_product="HA"),
    _N(name="Am243", bg_cutoff=1.0, Z=95, A=243, mass=243.0613811, element="Am", chem="Am",
       sigma_p=10.0, capture_th=75.1, fission_th=0.07, RI_capture=1800.0,
       ladder=Ladder(D=0.6, gg=0.040, S0=1.0e-4, target_spin=2.5, E_lo=0.5, seed=95243),
       fast_capture=(1.5, 0.2), fission_threshold=(0.85e6, 1.1, 0.12e6),
       scatter_knots=((1.0e5, 10.0), (1.0e7, 4.5)), nu=(3.06, 0.13), capture_product="HA"),
    _N(name="HA", Z=96, A=244, mass=244.0, element="Cm", chem="Cm", sigma_p=10.0),
]

# --------------------------------------------------------------------------
# Fission products (explicit poisons + chemistry-class lumps)
# --------------------------------------------------------------------------
_FISSION_PRODUCTS = [
    _N(name="I135", Z=53, A=135, mass=134.910048, element="I", chem="halogen",
       sigma_p=4.0, half_life=6.57 * SECONDS_PER_HOUR, decay=(("Xe135", 1.0),)),
    _N(name="Xe135", Z=54, A=135, mass=134.907227, element="Xe", chem="noble_gas",
       sigma_p=4.3, capture_th=2.65e6,
       resonances=(Resonance(0.0841, 25.4e-3, 92.0e-3, 0.0, 0.625),),
       half_life=9.14 * SECONDS_PER_HOUR, decay=(("LFP_SS", 1.0),), capture_product="LFP_NG"),
    _N(name="Pm149", Z=61, A=149, mass=148.918334, element="Pm", chem="rare_earth",
       sigma_p=6.0, capture_th=1400.0, RI_capture=1000.0,
       half_life=53.08 * SECONDS_PER_HOUR, decay=(("Sm149", 1.0),), capture_product="LFP_RE"),
    _N(name="Sm149", Z=62, A=149, mass=148.917185, element="Sm", chem="rare_earth",
       sigma_p=6.0, capture_th=40140.0, RI_capture=3400.0,
       resonances=(Resonance(0.0973, 0.53e-3, 60.5e-3, 0.0, 0.5625),),
       capture_product="LFP_RE"),
    _N(name="Sm151", Z=62, A=151, mass=150.919932, element="Sm", chem="rare_earth",
       sigma_p=6.0, capture_th=15170.0, RI_capture=3500.0, bg_cutoff=0.3, capture_product="LFP_RE"),
    _N(name="LFP_NG", Z=54, A=133, mass=133.0, element="Xe", chem="noble_gas",
       sigma_p=4.0, capture_th=14.0, RI_capture=110.0, capture_product="self"),
    _N(name="LFP_NM", Z=44, A=101, mass=101.0, element="Ru", chem="noble_metal",
       sigma_p=6.0, capture_th=14.0, RI_capture=140.0, capture_product="self"),
    _N(name="LFP_RE", Z=60, A=145, mass=145.0, element="Nd", chem="rare_earth",
       sigma_p=6.0, capture_th=54.0, RI_capture=150.0, capture_product="self"),
    _N(name="LFP_SS", Z=55, A=110, mass=110.0, element="Cs", chem="salt_seeker",
       sigma_p=5.0, capture_th=4.7, RI_capture=54.0, capture_product="self"),
]

NUCLIDES: dict[str, Nuclide] = {n.name: n for n in (_LIGHT + _ACTINIDES + _FISSION_PRODUCTS)}

FISSILE = ("U233", "U235", "Pu239", "Pu241")
FERTILE = ("Th232", "U238", "Pu240", "U234")
FISSION_PRODUCT_NAMES = tuple(n.name for n in _FISSION_PRODUCTS)
ACTINIDE_NAMES = tuple(n.name for n in _ACTINIDES)


def get(name: str) -> Nuclide:
    try:
        return NUCLIDES[name]
    except KeyError as exc:  # pragma: no cover - user error path
        raise KeyError(f"Unknown nuclide '{name}'. Known: {sorted(NUCLIDES)}") from exc


# Natural / reference isotopic compositions (atom fractions)
NATURAL_ABUNDANCE = {
    "Li": {"Li6": 0.0759, "Li7": 0.9241},
    "B": {"B10": 0.199, "B11": 0.801},
    "Cl": {"Cl35": 0.7576, "Cl37": 0.2424},
    "Be": {"Be9": 1.0}, "F": {"F19": 1.0}, "C": {"C12": 1.0}, "O": {"O16": 1.0},
    "Na": {"Na23": 1.0}, "Mg": {"Mg24": 1.0}, "K": {"K39": 1.0}, "Zr": {"Zr90": 1.0},
    "Cr": {"Cr52": 1.0}, "Fe": {"Fe56": 1.0}, "Ni": {"Ni58": 1.0}, "Mo": {"Mo96": 1.0},
    "Th": {"Th232": 1.0},
    "U": {"U234": 0.000054, "U235": 0.007204, "U238": 0.992742},
}

# Reactor-grade plutonium vector (atom fractions, typical LWR discharge)
REACTOR_GRADE_PU = {"Pu238": 0.025, "Pu239": 0.54, "Pu240": 0.25, "Pu241": 0.115, "Pu242": 0.07}
