"""Physical constants and unit helpers used throughout MSR-NX."""
from __future__ import annotations

import numpy as np

AVOGADRO = 6.02214076e23            # 1/mol
BARN = 1.0e-24                      # cm^2
K_BOLTZMANN_EV = 8.617333262e-5     # eV/K
NEUTRON_MASS_AMU = 1.00866491595    # u
NEUTRON_MASS_KG = 1.67492749804e-27 # kg
EV_TO_J = 1.602176634e-19
MEV_TO_J = 1.602176634e-13
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 365.25 * 86400.0
E_THERMAL_EV = 0.0253               # 2200 m/s reference energy
T_ROOM_K = 293.6
PCM = 1.0e5                         # 1 pcm = 1e-5 (dk/k)

# 4*pi*lambda-bar^2 * E  (barn * eV) for a neutron, laboratory frame
FOUR_PI_LAMBDA2_E = 2.6040e6


def kT(T_kelvin: float) -> float:
    """Thermal energy kT in eV."""
    return K_BOLTZMANN_EV * T_kelvin


def neutron_speed_cm_s(E_eV):
    """Non-relativistic neutron speed (cm/s) for kinetic energy E (eV)."""
    E = np.asarray(E_eV, dtype=float)
    return np.sqrt(2.0 * E * EV_TO_J / NEUTRON_MASS_KG) * 100.0


def awr_from_mass(mass_amu: float) -> float:
    """Atomic weight ratio (target mass / neutron mass)."""
    return mass_amu / NEUTRON_MASS_AMU


def rho_from_k(k):
    """Static reactivity rho = (k-1)/k."""
    k = np.asarray(k, dtype=float)
    return (k - 1.0) / k


def pcm(drho):
    """Convert a reactivity difference to pcm."""
    return np.asarray(drho) * PCM
