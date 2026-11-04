"""Fission-product yields (atoms per fission) for the MSR-NX product set.

Explicit poisons carry cumulative/independent chain yields; the remaining
fission products are grouped into four chemistry-class lumps that matter
for online fuel-salt processing:

* ``LFP_NG``  noble gases (Kr, Xe except Xe-135)   -> removed by He sparging
* ``LFP_NM``  noble / semi-noble metals (Mo..Te)   -> plate-out / filtration
* ``LFP_RE``  rare earths (Y, La..Gd except Sm/Pm) -> reductive extraction
* ``LFP_SS``  salt seekers (Rb, Sr, Cs, Ba, Zr...) -> salt discard only

Each row sums to 2.0 atoms per fission (two fragments).  Values are
thermal-fission (fast for U-238/Th-232) approximations.
"""
from __future__ import annotations

PRODUCTS = ("I135", "Xe135", "Pm149", "Sm151", "LFP_NG", "LFP_NM", "LFP_RE", "LFP_SS")

_RAW = {
    #         I135    Xe135   Pm149   Sm151   NG     NM     RE
    "U233":  (0.0488, 0.0136, 0.0077, 0.0033, 0.260, 0.380, 0.500),
    "U235":  (0.0628, 0.0025, 0.0108, 0.0042, 0.235, 0.480, 0.525),
    "Pu239": (0.0645, 0.0115, 0.0124, 0.0077, 0.230, 0.620, 0.480),
    "Pu241": (0.0703, 0.0023, 0.0154, 0.0091, 0.220, 0.650, 0.500),
    "U238":  (0.0696, 0.0003, 0.0162, 0.0080, 0.240, 0.550, 0.550),
    "Th232": (0.0545, 0.0012, 0.0060, 0.0028, 0.270, 0.300, 0.480),
}


def _complete(row):
    total = sum(row)
    return dict(zip(PRODUCTS, list(row) + [2.0 - total]))


YIELDS = {fis: _complete(row) for fis, row in _RAW.items()}

# Fissionable nuclides without their own entry borrow a surrogate's yields.
SURROGATE = {
    "U232": "U233", "U234": "U235", "U236": "U235", "U237": "U235",
    "Pa231": "Th232", "Pa233": "Th232", "Np237": "U235", "Pu238": "Pu239",
    "Pu240": "Pu239", "Pu242": "Pu241", "Am241": "Pu241", "Am243": "Pu241",
}


def yields_for(nuclide: str) -> dict:
    """Return the product-yield dictionary for a fissioning nuclide."""
    if nuclide in YIELDS:
        return YIELDS[nuclide]
    return YIELDS[SURROGATE.get(nuclide, "U235")]
