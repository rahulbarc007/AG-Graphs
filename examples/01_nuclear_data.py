"""Example 01 - Nuclear data: pointwise synthesis, Doppler broadening,
Bondarenko self-shielding and thermal scattering."""
import numpy as np
from _common import study

from msrnx import plotting as pl
from msrnx.data.nuclides import NUCLIDES
from msrnx.xs.energy import FINE45
from msrnx.xs.kernels import free_gas_kernel
from msrnx.xs.library import load_library
from msrnx.xs.pointwise import PointwiseNuclide

S = study("01_nuclear_data", "1. Nuclear data and the 45-group library",
          "Synthetic pointwise cross sections (SLBW resonances, exact Doppler broadening, "
          "statistical ladders) calibrated to 2200 m/s values and resonance integrals, "
          "collapsed to 45 groups with narrow-resonance Bondarenko self-shielding.")
lib = load_library()

# ---- calibration table --------------------------------------------------
rows = []
key = ["Th232", "Pa233", "U233", "U234", "U235", "U238", "Pu239", "Pu240", "Pu241", "Xe135",
       "Sm149", "Li6", "Li7", "Be9", "F19", "C12"]
for n in key:
    nuc = NUCLIDES[n]
    r = PointwiseNuclide(nuc).cal.report
    rows.append([n, nuc.capture_th, round(r["capture_th"], 4), nuc.fission_th,
                 round(r["fission_th"], 4), nuc.RI_capture if nuc.RI_capture else "-",
                 round(r["RI_capture"], 2), nuc.RI_fission if nuc.RI_fission else "-",
                 round(r["RI_fission"], 2)])
S.add_table("Calibration of synthetic cross sections (barns): target vs generated",
            ["nuclide", "sig_c target", "sig_c", "sig_f target", "sig_f", "RI_c target",
             "RI_c", "RI_f target", "RI_f"], rows)

# ---- Doppler broadening ---------------------------------------------------
pw = PointwiseNuclide(NUCLIDES["U238"])
fig, ax = pl.figure(1, 2, (11, 4.2), "Doppler broadening of resonances",
                    "Capture cross section from SLBW parameters with exact psi-chi (Faddeeva) broadening")
for (lo, hi, name, pwn, a) in [(6.0, 7.4, "U-238  6.67 eV", pw, ax[0]),
                               (20.8, 22.8, "Th-232  21.8 eV", PointwiseNuclide(NUCLIDES["Th232"]), ax[1])]:
    for i, T in enumerate([293.6, 900.0, 1500.0]):
        px = pwn.at(T)
        m = (px.E > lo) & (px.E < hi)
        a.semilogy(px.E[m], px.capture[m], color=pl.SERIES[i],
                   label=f"{T:.0f} K  (peak {px.capture[m].max():,.0f} b)")
    a.set_title(name, loc="left")
    a.set_xlabel("neutron energy [eV]")
    a.set_ylabel("capture cross section [b]")
    a.legend(loc="upper right")
S.add_figure(pl.save(fig, S.path("doppler_broadening.png")),
             "Peak capture falls and the wings rise as temperature increases (area is preserved).")

# ---- Bondarenko effective resonance integrals --------------------------
sig0 = np.logspace(0, 6, 40)
res_groups = np.where((FINE45.lower >= 0.5) & (FINE45.upper <= 2.5e4))[0]
du = FINE45.lethargy_width[res_groups]
fig, ax = pl.figure(1, 2, (11, 4.2), "Resonance self-shielding (Bondarenko)",
                    "Effective capture resonance integral vs background cross section sigma_0")
RIeff = {}
for a, n in zip(ax, ["Th232", "U238"]):
    mg = lib[n]
    for i, T in enumerate([293.6, 900.0, 1500.0]):
        ri = np.array([np.sum(mg.micro(T, s)["capture"][res_groups] * du) for s in sig0])
        RIeff[(n, T)] = ri
        a.semilogx(sig0, ri, color=pl.SERIES[i], label=f"{T:.0f} K")
    pl.end_label(a, sig0[-1], RIeff[(n, 1500.0)][-1], f"inf. dil. {RIeff[(n, 1500.0)][-1]:.0f} b")
    a.set_title(n.replace("Th", "Th-").replace("U2", "U-2"), loc="left")
    a.set_xlabel("sigma_0 [b]")
    a.set_ylabel("effective RI [b]")
    a.legend(loc="upper left")
S.add_figure(pl.save(fig, S.path("bondarenko_RI.png")),
             "Strong self-shielding at low dilution; higher temperature raises absorption (negative Doppler feedback).")
rows = []
for n in ["Th232", "U238"]:
    for s0 in [50.0, 200.0, 1000.0]:
        i = np.argmin(np.abs(sig0 - s0))
        rows.append([n, int(sig0[i]), round(RIeff[(n, 293.6)][i], 2), round(RIeff[(n, 900.0)][i], 2),
                     round(100 * (RIeff[(n, 900.0)][i] / RIeff[(n, 293.6)][i] - 1), 2)])
S.add_table("Effective resonance integral (b) and Doppler increase", ["nuclide", "sigma0 [b]",
            "RI 294 K", "RI 900 K", "Doppler +%"], rows)

# ---- multigroup cross sections -------------------------------------------
fig, ax = pl.figure(1, 2, (11, 4.2), "45-group microscopic cross sections (900 K, infinite dilution)")
for i, (n, rx, lab) in enumerate([("U233", "fission", "U-233 fission"), ("U235", "fission", "U-235 fission"),
                                  ("Pu239", "fission", "Pu-239 fission")]):
    pl.step_groups(ax[0], FINE45.bounds, lib[n].micro(900.0)[rx], pl.SERIES[i], lab)
for i, (n, rx, lab) in enumerate([("Th232", "capture", "Th-232 capture"), ("U238", "capture", "U-238 capture"),
                                  ("Pa233", "capture", "Pa-233 capture")]):
    pl.step_groups(ax[1], FINE45.bounds, lib[n].micro(900.0)[rx], pl.SERIES[i], lab)
for a in ax:
    a.set_yscale("log")
    a.set_xlabel("energy [eV]")
    a.set_ylabel("cross section [b]")
    a.legend()
ax[0].set_title("fissile nuclides", loc="left")
ax[1].set_title("fertile nuclides and Pa-233", loc="left")
S.add_figure(pl.save(fig, S.path("multigroup_xs.png")), "Group-averaged cross sections used by the lattice code.")

# ---- free-gas thermal kernel for graphite -------------------------------
fig, ax = pl.figure(1, 1, (7.2, 4.2), "Free-gas thermal scattering kernel, carbon at 900 K",
                    "Energy-transfer probability density for three incident energies (up- and down-scatter)")
Ep = np.geomspace(1e-4, 2.0, 2000)
for i, E in enumerate([0.0253, 0.1, 0.5]):
    K = free_gas_kernel(E, Ep, NUCLIDES["C12"].awr, 900.0)
    ax.semilogx(Ep, K * Ep, color=pl.SERIES[i], label=f"E = {E:g} eV")
    ax.axvline(E, color=pl.GRID, lw=1)
ax.set_xlabel("outgoing energy E' [eV]")
ax.set_ylabel("E' sigma(E->E')/sigma_free")
ax.legend()
S.add_figure(pl.save(fig, S.path("free_gas_kernel.png")),
             "Thermal neutrons gain energy from the hot graphite lattice (up-scatter) - origin of the moderator temperature effect.")

S.set(n_nuclides=len(lib.nuclides), n_groups=lib.G, library_hash=lib.meta["hash"])
S.save()
