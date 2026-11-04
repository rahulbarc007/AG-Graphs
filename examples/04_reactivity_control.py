"""Example 04 - Reactivity feedback and control: fuel (Doppler + density),
graphite and isothermal temperature coefficients, void coefficient, control
rod S-curve, shutdown margin and criticality search."""
import numpy as np
from _common import study

from msrnx import plotting as pl
from msrnx.core import ControlRod, msbr_design, msre_design, tmsr_leu_design
from msrnx.reactivity import (criticality_search, rod_worth_curve, set_fissile_fraction,
                              shutdown_margin, temperature_coefficients, temperature_scan,
                              void_coefficient)

S = study("04_reactivity_control", "4. Reactivity feedback, control rods and criticality",
          "All coefficients by central differences of full 2D core solutions; Doppler and "
          "density components separated by freezing either the cross sections or the density.")
designs = [msre_design(), msbr_design(), tmsr_leu_design()]
coef = {}
rows = []
for d in designs:
    c = temperature_coefficients(d, dT=50.0)
    coef[d.name] = c.coefficients
    rows.append([d.name] + [round(c.coefficients[k], 2) for k in
                            ("fuel_doppler", "fuel_density", "fuel_total", "graphite", "isothermal")])
S.add_table("Temperature coefficients of reactivity [pcm/K]",
            ["design", "fuel Doppler", "fuel density", "fuel total", "graphite", "isothermal"], rows)
S.add_table("Comparison with historical values [pcm/K]", ["quantity", "MSR-NX", "reported"], [
    ["MSRE fuel salt", round(coef["MSRE-like"]["fuel_total"], 2), "about -8.7"],
    ["MSRE graphite", round(coef["MSRE-like"]["graphite"], 2), "about -6.7"],
    ["MSBR fuel salt", round(coef["MSBR-like"]["fuel_total"], 2), "about -3.2"],
    ["MSBR graphite", round(coef["MSBR-like"]["graphite"], 2), "about +2.3 (free-gas graphite omits S(a,b))"]])
S.set(coefficients=coef)

fig, ax = pl.figure(1, 1, (9.5, 4.4), "Temperature coefficients of reactivity",
                    "Negative fuel-salt coefficients in all designs: prompt, inherent safety feedback")
cats = ["fuel Doppler", "fuel density", "fuel total", "graphite", "isothermal"]
keys = ["fuel_doppler", "fuel_density", "fuel_total", "graphite", "isothermal"]
pl.group_bars(ax, cats, {n: [coef[n][k] for k in keys] for n in coef}, fmt="{:.1f}",
              ylabel="pcm / K")
S.add_figure(pl.save(fig, S.path("temperature_coefficients.png")),
             "Doppler dominates for the thorium breeder; salt expansion dominates for the small MSRE core.")

# k vs temperature
d = msbr_design()
T = np.linspace(800, 1100, 7)
fig, ax = pl.figure(1, 2, (11, 4.0), "Reactivity versus temperature (MSBR-like)")
for i, which in enumerate(["fuel", "graphite", "isothermal"]):
    k = temperature_scan(d, T, which)
    rho = (k - 1) / k * 1e5
    ax[0].plot(T, rho - rho[np.argmin(np.abs(T - d.T_fuel))], "o-", color=pl.SERIES[i], label=which)
ax[0].set_xlabel("temperature [K]")
ax[0].set_ylabel("reactivity change [pcm]")
ax[0].legend()
ax[0].set_title("temperature scans", loc="left")
v = void_coefficient(d)
ax[1].plot(100 * v["void"], v["rho_pcm"] - v["rho_pcm"][0], "o-", color=pl.SERIES[0])
pl.value_label(ax[1], 100 * v["void"][-1], v["rho_pcm"][-1] - v["rho_pcm"][0],
               f"{v['pcm_per_percent_void']:.0f} pcm per % void", dx=-6, dy=10, ha="right")
ax[1].set_xlabel("helium void fraction in salt [%]")
ax[1].set_ylabel("reactivity change [pcm]")
ax[1].set_title("void coefficient", loc="left")
S.add_figure(pl.save(fig, S.path("reactivity_vs_temperature.png")),
             "Reactivity falls with fuel temperature and with helium voiding of the salt.")
S.set(void_coefficient_pcm_per_pct=v["pcm_per_percent_void"])

# control rod S-curve (MSRE-like, central rod)
dm = msre_design()
rod = ControlRod(radius=3.0, absorber_fraction=0.30)
rw = rod_worth_curve(dm, np.linspace(0, dm.core_height, 13), rod)
fig, ax = pl.figure(1, 2, (11, 4.0), "Central control rod worth (MSRE-like core)",
                    "B4C absorber channel, r = 3 cm (30% absorber), inserted from the top")
ax[0].plot(rw["insertion_cm"], -rw["worth_pcm"], "o-", color=pl.SERIES[0])
pl.value_label(ax[0], rw["insertion_cm"][-1], -rw["worth_pcm"][-1], f"{-rw['worth_pcm'][-1]:.0f} pcm",
               dx=-6, dy=-10, ha="right")
ax[0].set_xlabel("insertion depth [cm]")
ax[0].set_ylabel("integral worth [pcm]")
ax[0].set_title("integral worth (S-curve)", loc="left")
ax[1].plot(rw["z_mid"], -rw["differential_pcm_per_cm"], "o-", color=pl.SERIES[1])
ax[1].set_xlabel("insertion depth [cm]")
ax[1].set_ylabel("differential worth [pcm/cm]")
ax[1].set_title("differential worth", loc="left")
S.add_figure(pl.save(fig, S.path("rod_worth.png")), "Differential worth peaks where the flux-importance product is largest.")
sm = shutdown_margin(dm, [-rw["worth_pcm"][-1]] * 3)
S.add_table("Control rod and shutdown margin (MSRE-like, 3 identical rods, one stuck out)",
            ["quantity", "value"], [["single-rod worth [pcm]", round(-rw["worth_pcm"][-1], 1)],
                                    ["excess reactivity [pcm]", round(sm["excess_pcm"], 1)],
                                    ["available worth, stuck rod [pcm]", round(sm["available_worth_pcm"], 1)],
                                    ["shutdown margin [pcm]", round(sm["shutdown_margin_pcm"], 1)]])
S.set(rod_worth_pcm=-rw["worth_pcm"][-1], shutdown_margin_pcm=sm["shutdown_margin_pcm"])

# criticality search: UF4 for k_eff = 1.03 (TMSR-LEU)
dt = tmsr_leu_design()
cs = criticality_search(dt, lambda dd, x: set_fissile_fraction(dd, x), 0.004, 0.02, target_k=1.03)
h = cs["history"]
fig, ax = pl.figure(1, 1, (7.2, 4.0), "Criticality search (Brent's method)",
                    "UF4 mole fraction giving k_eff = 1.03 for the 5% LEU core")
ax.plot(100 * h[:, 0], h[:, 1], "o-", color=pl.SERIES[0])
ax.axhline(1.03, color=pl.AXIS, lw=1)
pl.value_label(ax, 100 * cs["value"], 1.03, f"UF4 = {100 * cs['value']:.3f} mol%", dy=12, ha="center")
ax.set_xlabel("UF4 [mol %]")
ax.set_ylabel("k_eff")
S.add_figure(pl.save(fig, S.path("criticality_search.png")), "Every point is a full 2D core solution.")
S.set(critical_uf4_molpct=100 * cs["value"])
S.save()
