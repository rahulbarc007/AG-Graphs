"""Example 03 - Full-core analysis: two-step lattice -> spectral zones -> 2D R-Z
diffusion for the MSBR-like core; flux/power maps, adjoint, peaking, kinetics
parameters, graphite lifetime, design comparison and group-condensation study."""
import numpy as np
from _common import study

from msrnx import plotting as pl
from msrnx.core import PRESETS, msbr_design, msre_design, solve_core
from msrnx.xs.energy import FINE45, few_group

S = study("03_core_analysis", "3. Full-core neutronics (2D R-Z multigroup diffusion)",
          "Lattice calculations per zone, 1D radial/axial 45-group spectral-zone calculations, "
          "condensation to 8 groups and 2D forward/adjoint diffusion.")
d = msbr_design()
r = solve_core(d, "8g", mesh_size=5.0, n_modes=3)
s = r.summary()
S.add_table("MSBR-like core: key results", ["quantity", "value"], [
    ["k_eff", round(r.k_eff, 5)], ["excess reactivity [pcm]", round(s["reactivity_pcm"], 0)],
    ["k_inf zone I / zone II", f"{r.k_inf_zones[0]:.4f} / {r.k_inf_zones[1]:.4f}"],
    ["dominance ratio k1/k0", round(s["dominance_ratio"], 4)],
    ["beta_eff static [pcm]", round(s["beta_eff_static_pcm"], 1)],
    ["prompt generation time [us]", round(s["generation_time_us"], 1)],
    ["conversion ratio (clean)", round(s["conversion_ratio"], 4)],
    ["power peaking total / radial / axial",
     f"{s['power_peaking']:.2f} / {s['radial_peaking']:.2f} / {s['axial_peaking']:.2f}"],
    ["average / peak power density [W/cc]", f"{s['q_avg_W_cm3']:.1f} / {s['q_max_W_cm3']:.1f}"],
    ["power outside graphite zones", f"{100 * s['power_outside_zones']:.1f} %"],
    ["neutron leakage from core zones", f"{100 * s['core_leakage_fraction']:.2f} %"],
    ["peak graphite fast flux (>50 keV) [n/cm2/s]", f"{s['graphite_fast_flux_peak']:.3e}"],
    ["graphite lifetime [full-power years at CF]", round(s["graphite_lifetime_years"], 2)],
    ["tritium production [Ci/day]", round(s["tritium_Ci_per_day"], 1)],
    ["solve time [s]", round(r.timings["total"], 2)]])
S.set(**{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in s.items()})

mesh = r.diffusion.mesh
phi = r.diffusion.flux
G = r.groups
th = G.upper <= 0.42
fa = G.lower >= 6.7e4
outline = [(0, d.zones[0].outer_radius, d.axial_reflector + d.lower_plenum,
            d.axial_reflector + d.lower_plenum + d.core_height),
           (d.zones[0].outer_radius, d.zones[1].outer_radius, d.axial_reflector + d.lower_plenum,
            d.axial_reflector + d.lower_plenum + d.core_height)]
fig, ax = pl.figure(1, 3, (13, 5.6), "MSBR-like core: flux and power distributions (R-Z)",
                    "Zone outlines in white; normalized to 2250 MWth")
pl.rz_map(ax[0], mesh.r_edges, mesh.z_edges, phi[fa].sum(0), "fast flux >67 keV [n/cm2/s]", outline=outline)
ax[0].set_title("fast flux", loc="left")
pl.rz_map(ax[1], mesh.r_edges, mesh.z_edges, phi[th].sum(0), "thermal flux <0.41 eV [n/cm2/s]", outline=outline)
ax[1].set_title("thermal flux (reflector peak)", loc="left")
pl.rz_map(ax[2], mesh.r_edges, mesh.z_edges, r.diffusion.power_density(), "power density [W/cm3]", outline=outline)
ax[2].set_title("power density", loc="left")
S.add_figure(pl.save(fig, S.path("rz_maps.png")),
             "Thermal flux peaks in the graphite reflector; power peaks in the well-moderated zone I.")

# adjoint
fig, ax = pl.figure(1, 2, (10, 5.6), "Neutron importance (adjoint flux)",
                    "Adjoint solution of the same operator; weights kinetics parameters and perturbations")
pl.rz_map(ax[0], mesh.r_edges, mesh.z_edges, r.diffusion.adjoint[th].sum(0) / r.diffusion.adjoint.max(),
          "thermal importance (normalized)", outline=outline)
ax[0].set_title("thermal importance", loc="left")
pl.rz_map(ax[1], mesh.r_edges, mesh.z_edges, r.diffusion.adjoint[fa].sum(0) / r.diffusion.adjoint.max(),
          "fast importance (normalized)", outline=outline)
ax[1].set_title("fast importance", loc="left")
S.add_figure(pl.save(fig, S.path("adjoint_maps.png")), "Adjoint (importance) maps.")

# radial / axial power profiles
q = r.diffusion.power_density()
V = mesh.volumes()
zone_mask = mesh.region < len(d.zones)
z0 = d.axial_reflector + d.lower_plenum
iz = np.argmin(np.abs(mesh.zc - (z0 + d.core_height / 2)))
fig, ax = pl.figure(1, 2, (11, 4.0), "Power profiles")
ax[0].plot(mesh.rc, q[:, iz], color=pl.SERIES[0])
for zz in d.zones:
    ax[0].axvline(zz.outer_radius, color=pl.AXIS, lw=1)
    pl.value_label(ax[0], zz.outer_radius, q[:, iz].max() * 0.95, zz.name, dx=-4, ha="right")
ax[0].set_xlabel("r [cm]")
ax[0].set_ylabel("power density at mid-plane [W/cm3]")
ax[0].set_title("radial (mid-plane)", loc="left")
qz = (q * V).sum(0) / V.sum(0)
ax[1].plot(qz, mesh.zc, color=pl.SERIES[1])
ax[1].axhspan(z0, z0 + d.core_height, color=pl.GRID, alpha=0.35, lw=0)
ax[1].set_ylabel("z [cm]")
ax[1].set_xlabel("radially averaged power density [W/cm3]")
ax[1].set_title("axial (fission also in the plena)", loc="left")
S.add_figure(pl.save(fig, S.path("power_profiles.png")), "Mid-plane radial and radially averaged axial power shapes.")

# region spectra
fig, ax = pl.figure(1, 1, (7.6, 4.3), "Region-wise spectra from the 1D spectral-zone calculations",
                    "Used to condense each region's 45-group constants to the core group structure")
for i, (rid, lab) in enumerate([(0, "zone I"), (1, "zone II"), (2, "radial reflector"), (6, "upper plenum")]):
    spec = r.fine_spectra["used"][rid] / FINE45.lethargy_width
    ax.semilogx(FINE45.midpoints, spec / spec.max(), drawstyle="steps-mid", color=pl.SERIES[i], label=lab)
ax.set_xlabel("energy [eV]")
ax.set_ylabel("flux per lethargy (normalized)")
ax.legend()
S.add_figure(pl.save(fig, S.path("region_spectra.png")),
             "The under-moderated zone II and the salt plena have harder spectra than zone I.")

# ---- design comparison -----------------------------------------------------
rows, vals = [], {"k_eff": [], "beta [pcm]": [], "Lambda [us]": [], "conv. ratio": []}
names = []
for key, fn in PRESETS.items():
    dd = fn()
    rr = solve_core(dd, "6g", 8.0)
    ss = rr.summary()
    names.append(dd.name.replace(" MSR", ""))
    rows.append([dd.name, round(rr.k_eff, 4), round(ss["beta_eff_static_pcm"], 1),
                 round(ss["generation_time_us"], 1), round(ss["conversion_ratio"], 3),
                 round(ss["power_peaking"], 2), f"{ss['graphite_fast_flux_peak']:.2e}",
                 round(rr.timings["total"], 2)])
    vals["k_eff"].append(rr.k_eff)
    vals["beta [pcm]"].append(ss["beta_eff_static_pcm"])
    vals["Lambda [us]"].append(ss["generation_time_us"])
    vals["conv. ratio"].append(ss["conversion_ratio"])
S.add_table("Reference designs (clean, fresh fuel)", ["design", "k_eff", "beta_eff [pcm]", "Lambda [us]",
            "conv. ratio", "peaking", "fast flux peak", "time [s]"], rows)
fig, ax = pl.figure(1, 4, (14, 3.8), "Reference MSR designs compared")
for a, (lab, v) in zip(ax, vals.items()):
    xs = np.arange(len(v))
    a.bar(xs, v, width=0.6, color=pl.SERIES[0], edgecolor=pl.SURFACE, linewidth=2)
    a.set_xticks(xs, names, rotation=25, ha="right")
    a.set_title(lab, loc="left")
    a.grid(axis="x", visible=False)
    for x, y in zip(xs, v):
        pl.value_label(a, x, y, f"{y:.3g}", ha="center", va="bottom", dx=0, dy=2)
    if lab == "Lambda [us]":
        a.set_yscale("log")
S.add_figure(pl.save(fig, S.path("design_comparison.png")),
             "Graphite-moderated cores have ~100x longer generation times than the fast chloride core; U-233 has the smallest beta.")

# ---- group condensation study (energy parameter reduction) ----------------
dm = msre_design()
ref = solve_core(dm, FINE45, mesh_size=4.0, adjoint=False, n_modes=1)
rows = []
ng, err = [], []
for name in ["2g", "4g", "6g", "8g"]:
    rr = solve_core(dm, name, mesh_size=4.0, adjoint=False, n_modes=1)
    e = (rr.k_eff - ref.k_eff) * 1e5
    ng.append(few_group(name).G)
    err.append(e)
    rows.append([name, few_group(name).G, round(rr.k_eff, 5), round(e, 1), round(rr.timings["core_2d"], 3)])
rows.append(["45g (reference)", 45, round(ref.k_eff, 5), 0.0, round(ref.timings["core_2d"], 3)])
S.add_table("Energy-group condensation (MSRE-like core, 2D)", ["structure", "groups", "k_eff",
            "error vs 45g [pcm]", "2D solve [s]"], rows)
fig, ax = pl.figure(1, 1, (7.2, 4.0), "Energy-group condensation error",
                    "k_eff error of condensed 2D solutions vs the 45-group 2D reference (MSRE-like)")
ax.plot(ng, np.abs(err), "o-", color=pl.SERIES[0])
for x, y in zip(ng, np.abs(err)):
    pl.value_label(ax, x, y, f"{y:.0f} pcm", dy=8, ha="center")
ax.set_xlabel("number of energy groups in the core calculation")
ax.set_ylabel("|k error| [pcm]")
ax.set_xticks(ng)
S.add_figure(pl.save(fig, S.path("group_condensation.png")),
             "Spectral-zone condensation keeps few-group errors small - a physics parameter reduction from 45 to a handful of groups.")
S.save()
