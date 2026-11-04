"""MSR-NX: an AI/ML-augmented neutronics design framework for molten salt reactors.

Physics: nuclear data synthesis and Bondarenko self-shielding, lattice
(collision probability + B1), multigroup R-Z diffusion (forward/adjoint),
circulating-fuel kinetics, reactivity coefficients, control rods, Xe/Sm
poisoning, and depletion with CRAM and online reprocessing.

AI/ML (``msrnx.ml``): surrogates with uncertainty, active learning,
physics-guarded prediction, multi-fidelity correction, Sobol / Morris /
active-subspace parameter reduction, POD reduced-order models, XS
functionalization, UQ and Bayesian design optimization.
"""
__version__ = "1.0.0"

from .xs.library import load_library  # noqa: F401
from .xs.energy import FINE45, few_group  # noqa: F401
from .materials import (Material, molten_salt, flibe, flinak, msre_fuel_salt,  # noqa: F401
                        msbr_fuel_salt, graphite, hastelloy_n, b4c, mix, salt_at_temperature)
from .macro import MacroXS, build_macro, condense  # noqa: F401
from .lattice import (CellGeometry, run_lattice, fundamental_mode, critical_buckling,  # noqa: F401
                      six_factor, conversion_ratio)
from .diffusion import Mesh, DiffusionSolver, solve_diffusion  # noqa: F401
from .core import (MSRDesign, CoreZone, ControlRod, solve_core, PRESETS, msre_design,  # noqa: F401
                   msbr_design, tmsr_leu_design, fast_chloride_design)
