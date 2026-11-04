"""Example 02 - Lattice physics of a graphite-moderated thorium MSR cell:
moderation, heterogeneity (resonance lumping), spectra, six-factor formula,
neutron balance and critical buckling."""
import numpy as np
from _common import study

from msrnx import plotting as pl
from msrnx.core import msbr_design
from msrnx.lattice import CellGeometry, conversion_ratio, critical_buckling, run_lattice
from msrnx.reactivity import moderation_curve
from msrnx.xs.energy import FINE45

S = study("02_lattice_physics", "2. Lattice physics: MSBR-type fuel channel in graphite",
          "Two-region collision-probability cell (Carlvik escape probability, Sauer Dancoff "
          "factor, Bell-Dancoff equivalence for self-shielding) with 45 energy groups.")
d = msbr_design()
salt, mod = d.fuel_salt(), d.moderator()
z = d.zones[0]
cell = CellGeometry(z.channel_radius, z.salt_fraction, z.lattice)
het = run_lattice(salt, mod, cell, heterogeneous=True)
hom = run_lattice(salt, mod, cell, heterogeneous=False)

rows = []
for lab, r in [("heterogeneous (CP)", het), ("homogenized mixture", hom)]:
    sf = r.six_factor
    rows.append([lab, round(r.k_inf, 5), round(sf["eta"], 4), round(sf["f"], 4), round(sf["p"], 4),
                 round(sf["epsilon"], 4), round(conversion_ratio(r.fuel_macro, r.phi_fuel * r.V_fuel), 4),
                 round(r.sig0["Th232"], 1), f"{critical_buckling(r.hom):.3e}"])
S.add_table("MSBR zone-I cell (salt fraction 0.13, channel radius 1.0 cm, hex)",
            ["model", "k_inf", "eta", "f", "p", "epsilon", "conv. ratio", "sigma0 Th [b]", "B2_crit [1/cm2]"], rows)
S.set(k_inf_het=het.k_inf, k_inf_hom=hom.k_inf, lumping_pcm=(1 / hom.k_inf - 1 / het.k_inf) * 1e5,
      thermal_disadvantage=het.thermal_disadvantage)

# ---- moderation curve ----------------------------------------------------
mc = moderation_curve(d, np.linspace(0.04, 0.6, 29))
fig, ax = pl.figure(1, 2, (11, 4.2), "Moderation ratio study",
                    "k_inf and conversion ratio versus fuel-salt volume fraction (channel radius 1.0 cm)")
ax[0].plot(mc["salt_fraction"], mc["k_inf"], color=pl.SERIES[0])
i = int(np.argmax(mc["k_inf"]))
ax[0].plot(mc["salt_fraction"][i], mc["k_inf"][i], "o", color=pl.SERIES[0])
pl.value_label(ax[0], mc["salt_fraction"][i], mc["k_inf"][i],
               f"optimum moderation: salt fraction {mc['salt_fraction'][i]:.2f}, k_inf={mc['k_inf'][i]:.3f}",
               dx=10, dy=-4, ha="left")
ax[0].axvspan(0.04, mc["salt_fraction"][i], color=pl.GRID, alpha=0.35, lw=0)
ax[0].text(0.06, ax[0].get_ylim()[0] + 0.02, "over-moderated", color=pl.INK2, fontsize=8.5)
ax[0].set_xlabel("salt volume fraction")
ax[0].set_ylabel("k_inf")
ax[0].set_title("multiplication", loc="left")
ax[1].plot(mc["salt_fraction"], mc["conversion_ratio"], color=pl.SERIES[1])
ax[1].axhline(1.0, color=pl.AXIS, lw=1)
ax[1].set_xlabel("salt volume fraction")
ax[1].set_ylabel("conversion ratio")
ax[1].set_title("breeding", loc="left")
S.add_figure(pl.save(fig, S.path("moderation_curve.png")),
             "Under-moderated lattices (high salt fraction) trade reactivity for breeding.")
S.set(optimum_salt_fraction=mc["optimum_salt_fraction"])

# ---- heterogeneity (lumping) --------------------------------------------
radii = np.linspace(0.3, 4.0, 16)
kh, s0, C = [], [], []
for r in radii:
    res = run_lattice(salt, mod, CellGeometry(float(r), z.salt_fraction, z.lattice))
    kh.append(res.k_inf)
    s0.append(res.sig0["Th232"])
    C.append(float(np.mean(res.dancoff[FINE45.lower > 1.0])))
fig, ax = pl.figure(1, 2, (11, 4.2), "Heterogeneity effect: resonance self-shielding by lumping",
                    "Same salt fraction (0.13); larger channels shield Th-232 resonances more strongly")
ax[0].plot(radii, kh, color=pl.SERIES[0], label="heterogeneous cell")
ax[0].axhline(hom.k_inf, color=pl.SERIES[1], label="homogenized mixture")
ax[0].set_xlabel("fuel channel radius [cm]")
ax[0].set_ylabel("k_inf")
ax[0].legend(loc="lower right")
ax[1].plot(radii, s0, color=pl.SERIES[2])
ax[1].set_xlabel("fuel channel radius [cm]")
ax[1].set_ylabel("Th-232 background sigma_0 [b]")
ax[1].set_yscale("log")
pl.end_label(ax[1], radii[-1], s0[-1], f"{s0[-1]:.0f} b", pl.SERIES[2])
S.add_figure(pl.save(fig, S.path("heterogeneity.png")),
             "Lumping the fuel lowers sigma_0 and raises the resonance escape probability.")
S.add_table("Lumping effect", ["channel radius [cm]", "k_inf", "sigma0 Th [b]", "Dancoff C"],
            [[round(r, 2), round(k, 5), round(s, 1), round(c, 4)] for r, k, s, c in
             list(zip(radii, kh, s0, C))[::3]])

# ---- spectra -------------------------------------------------------------
fig, ax = pl.figure(1, 1, (7.6, 4.3), "Neutron spectra in the cell",
                    "Flux per unit lethargy (normalized), fuel channel vs graphite moderator")
Em = FINE45.midpoints
for i, (lab, key) in enumerate([("fuel salt channel", "fuel"), ("graphite moderator", "mod")]):
    spec = het.lethargy_spectrum(key)
    ax.semilogx(Em, spec / spec.max(), drawstyle="steps-mid", color=pl.SERIES[i], label=lab)
ax.set_xlabel("energy [eV]")
ax.set_ylabel("flux per lethargy (normalized)")
ax.legend()
S.add_figure(pl.save(fig, S.path("cell_spectra.png")),
             f"Thermal disadvantage factor phi_mod/phi_fuel = {het.thermal_disadvantage:.3f}.")

# ---- six-factor and balance ---------------------------------------------
fig, ax = pl.figure(1, 2, (11, 4.0), "Neutron economy of the cell")
sf = het.six_factor
pl.hbar(ax[0], ["eta (HM)", "f (thermal util.)", "p (res. escape)", "epsilon (epithermal/fast)"],
        [sf["eta"], sf["f"], sf["p"], sf["epsilon"]], sort=False, fmt="{:.4f}")
ax[0].set_title(f"four-factor formula  (k_inf = {het.k_inf:.4f})", loc="left")
bal = {k: v for k, v in het.balance.items() if v > 0.001}
pl.hbar(ax[1], list(bal.keys()), [100 * v for v in bal.values()], color=pl.SERIES[0], fmt="{:.2f}%",
        xlabel="share of absorptions [%]")
ax[1].set_title("absorptions by nuclide", loc="left")
S.add_figure(pl.save(fig, S.path("neutron_economy.png")),
             "Th-232 and U-233 dominate absorption; Li-6/Li-7 are the main parasitic absorbers.")
S.save()
