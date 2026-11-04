"""Example 06 - Fission-product poisoning: equilibrium Xe-135 / Sm-149 worth,
helium-sparging effectiveness, post-shutdown iodine pit, load following."""
import numpy as np
from _common import study

from msrnx import plotting as pl
from msrnx.core import solve_core, tmsr_leu_design
from msrnx.poisons import calibrate_worth, equilibrium, poison_params_from_core, sparging_study, transient
from dataclasses import replace

S = study("06_xenon_samarium", "6. Xenon and samarium poisoning with off-gas sparging",
          "One-group I/Xe/Pm/Sm balance on the inventory-averaged flux; worths calibrated "
          "against direct 2D transport calculations with the poison added to the salt.")
d = tmsr_leu_design()
core = solve_core(d)
p0 = calibrate_worth(poison_params_from_core(core), d)
eq0 = equilibrium(p0)
S.add_table("Poison parameters (TMSR-LEU, 250 MWth)", ["quantity", "value"], [
    ["salt-averaged flux [n/cm2/s]", f"{p0.flux:.3e}"],
    ["one-group sigma_a Xe-135 [b]", f"{p0.sigma_Xe * 1e24:.3e}"],
    ["one-group sigma_a Sm-149 [b]", f"{p0.sigma_Sm * 1e24:.3e}"],
    ["yields I-135 / Xe-135 / Pm-149", f"{p0.gamma_I:.4f} / {p0.gamma_Xe:.4f} / {p0.gamma_Pm:.4f}"],
    ["transport calibration factor", round(p0.worth_scale, 3)],
    ["equilibrium Xe worth, no sparging [pcm]", round(eq0["rho_Xe_pcm"], 1)],
    ["equilibrium Sm worth [pcm]", round(eq0["rho_Sm_pcm"], 1)]])

# sparging effectiveness
lam = np.logspace(-6, -1, 40)
sp = sparging_study(p0, lam)
fig, ax = pl.figure(1, 1, (7.6, 4.2), "Xenon removal by helium sparging",
                    "Equilibrium Xe-135 worth versus off-gas removal rate constant")
ax.semilogx(1.0 / lam / 60.0, -sp["rho_Xe_pcm"], color=pl.SERIES[0])
for tau_min in [1.0, 60.0, 600.0]:
    v = -equilibrium(replace(p0, lam_sparge=1 / (tau_min * 60)))["rho_Xe_pcm"]
    ax.plot(tau_min, v, "o", color=pl.SERIES[0])
    pl.value_label(ax, tau_min, v, f"{v:.0f} pcm", dx=6, dy=-8)
ax.invert_xaxis()
ax.set_xlabel("mean Xe residence time before removal [min]  (faster removal ->)")
ax.set_ylabel("equilibrium xenon worth [pcm]")
S.add_figure(pl.save(fig, S.path("xenon_sparging.png")),
             "Removing Xe on a ~1 minute time scale nearly eliminates xenon poisoning - a unique MSR advantage.")

# post-shutdown iodine pit with/without sparging
days = 3.0
cases = [("no sparging", replace(p0, lam_sparge=0.0)),
         ("sparging, 60 min", replace(p0, lam_sparge=1 / 3600.0)),
         ("sparging, 5 min", replace(p0, lam_sparge=1 / 300.0))]
fig, ax = pl.figure(1, 2, (11, 4.2), "Xenon transients",
                    "Left: shutdown at t = 0 from equilibrium. Right: 100-50-100 % load follow")
rows = []
for i, (lab, p) in enumerate(cases):
    tr = transient(p, days * 86400, lambda t: 0.0, y0=equilibrium(p, 1.0))
    ax[0].plot(tr["t"] / 3600, -tr["rho_Xe_pcm"], color=pl.SERIES[i], label=lab)
    j = int(np.argmax(-tr["rho_Xe_pcm"]))
    rows.append([lab, round(-equilibrium(p)["rho_Xe_pcm"], 1), round(-tr["rho_Xe_pcm"][j], 1),
                 round(tr["t"][j] / 3600, 1)])
ax[0].set_xlabel("time after shutdown [h]")
ax[0].set_ylabel("xenon worth [pcm]")
ax[0].legend()
load = lambda t: 1.0 if t < 6 * 3600 else (0.5 if t < 18 * 3600 else 1.0)
for i, (lab, p) in enumerate(cases[:2]):
    tr = transient(p, 48 * 3600, load)
    ax[1].plot(tr["t"] / 3600, -tr["rho_Xe_pcm"], color=pl.SERIES[i], label=lab)
ax[1].set_xlabel("time [h]")
ax[1].set_ylabel("xenon worth [pcm]")
ax[1].axvspan(6, 18, color=pl.GRID, alpha=0.4, lw=0)
pl.value_label(ax[1], 12, ax[1].get_ylim()[0], "50 % power", ha="center", va="bottom", dy=4)
ax[1].legend()
S.add_figure(pl.save(fig, S.path("xenon_transients.png")),
             "The post-shutdown iodine pit that limits restart in solid-fuel reactors is suppressed by sparging.")
S.add_table("Xenon after shutdown", ["case", "equilibrium worth [pcm]", "peak worth [pcm]",
                                     "time of peak [h]"], rows)

# samarium after shutdown
tr = transient(p0, 30 * 86400, lambda t: 0.0, y0=equilibrium(p0, 1.0))
fig, ax = pl.figure(1, 1, (7.2, 4.0), "Samarium-149 after shutdown",
                    "Pm-149 (53 h) keeps decaying into stable Sm-149 once the flux is gone")
ax.plot(tr["t"] / 86400, -tr["rho_Sm_pcm"], color=pl.SERIES[0])
pl.end_label(ax, tr["t"][-1] / 86400, -tr["rho_Sm_pcm"][-1], f"{-tr['rho_Sm_pcm'][-1]:.0f} pcm")
ax.set_xlabel("time after shutdown [days]")
ax.set_ylabel("samarium worth [pcm]")
S.add_figure(pl.save(fig, S.path("samarium.png")), "Samarium worth rises to a new plateau after shutdown.")
S.set(xe_eq_pcm=-eq0["rho_Xe_pcm"], sm_eq_pcm=-eq0["rho_Sm_pcm"],
      sm_after_shutdown_pcm=-tr["rho_Sm_pcm"][-1])
S.save()
