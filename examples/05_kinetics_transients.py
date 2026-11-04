"""Example 05 - Circulating-fuel kinetics: delayed-neutron precursor drift,
MSRE benchmark, inhour curves and transients (reactivity step, unprotected
loss of flow, overcooling, scram) with thermal feedback."""
import numpy as np
from _common import study

from msrnx import plotting as pl
from msrnx.core import msbr_design, msre_design, solve_core
from msrnx.data.delayed_neutrons import DELAYED
from msrnx.kinetics import (CirculatingPointKinetics, circulating_beta, decay_heat_fraction,
                            inhour_static, thermal_model_for)
from msrnx.reactivity import temperature_coefficients

S = study("05_kinetics_transients", "5. Circulating-fuel kinetics and transients",
          "Precursor drift from importance-weighted slug flow; multi-node circulating point "
          "kinetics with fuel/graphite feedback and an advected heat-exchanger loop.")

# --- MSRE benchmark -------------------------------------------------------
dm = msre_design()
rm = solve_core(dm)
d235 = DELAYED["U235"]
uni = circulating_beta(d235.beta_i, d235.lam_i, 8.46, 16.73, "uniform")
tc, tl = dm.transit_times(include_plena=True)
ax_ = circulating_beta(rm.beta_i, rm.lambda_i, tc, tl, "axial", rm.axial_fission_shape,
                       rm.axial_importance_shape)
S.add_table("MSRE reactivity loss from fuel circulation (U-235 fuel)", ["model", "loss [pcm]"],
            [["uniform slug flow (tau_c 8.46 s, tau_L 16.73 s)", round(uni.loss_pcm, 1)],
             ["importance-weighted, core + plena (MSR-NX geometry)", round(ax_.loss_pcm, 1)],
             ["MSRE measurement", "about 212"]])
S.set(msre_loss_uniform=uni.loss_pcm, msre_loss_axial=ax_.loss_pcm)

# --- beta_eff vs pump flow ------------------------------------------------
d = msbr_design()
r = solve_core(d)
tcb, tlb = d.transit_times(include_plena=True)
flows = np.linspace(0.02, 1.5, 40)
fig, ax = pl.figure(1, 2, (11, 4.0), "Delayed-neutron loss by fuel circulation",
                    "Effective delayed fraction versus pump flow and per precursor group")
for i, (lab, res_, tc0, tl0) in enumerate([("MSRE-like (U-235)", rm, tc, tl), ("MSBR-like (U-233)", r, tcb, tlb)]):
    b = [circulating_beta(res_.beta_i, res_.lambda_i, tc0 / f, tl0 / f, "axial",
                          res_.axial_fission_shape, res_.axial_importance_shape).beta_circ * 1e5
         for f in flows]
    ax[0].plot(flows, b, color=pl.SERIES[i], label=lab)
    pl.end_label(ax[0], flows[-1], b[-1], f"{b[-1]:.0f} pcm")
ax[0].set_xlabel("pump flow / nominal")
ax[0].set_ylabel("beta_eff with circulation [pcm]")
ax[0].legend(loc="upper right")
cb = circulating_beta(r.beta_i, r.lambda_i, tcb, tlb, "axial", r.axial_fission_shape,
                      r.axial_importance_shape)
pl.group_bars(ax[1], [f"g{i + 1}" for i in range(6)],
              {"static": 1e5 * cb.beta_i_static, "circulating": 1e5 * cb.beta_i_circ},
              fmt="{:.0f}", ylabel="beta_i [pcm]")
ax[1].set_title("MSBR-like, per group", loc="left")
S.add_figure(pl.save(fig, S.path("beta_vs_flow.png")),
             "Long-lived precursor groups (1-3) decay mostly outside the core when the fuel circulates.")
S.set(msbr_beta_static=cb.beta_static * 1e5, msbr_beta_circ=cb.beta_circ * 1e5)

# --- inhour ----------------------------------------------------------------
coef = temperature_coefficients(d, dT=50.0, components=False).coefficients
th = thermal_model_for(d)
pk_c = CirculatingPointKinetics(r.beta_i, r.lambda_i, r.generation_time, tcb, tlb,
                                alpha_fuel=coef["fuel_total"] * 1e-5, alpha_graphite=coef["graphite"] * 1e-5,
                                fission_shape=r.axial_fission_shape,
                                importance_shape=r.axial_importance_shape, thermal=th)
pk_s = CirculatingPointKinetics(r.beta_i, r.lambda_i, r.generation_time, tcb, tlb,
                                alpha_fuel=coef["fuel_total"] * 1e-5, alpha_graphite=coef["graphite"] * 1e-5,
                                thermal=th, circulating=False)
rhos = np.linspace(10, 250, 25) * 1e-5
Tc = [pk_c.stable_period(x) for x in rhos]
Ts = [pk_s.stable_period(x) for x in rhos]
fig, ax = pl.figure(1, 1, (7.4, 4.2), "Inhour relation: stable period vs step reactivity",
                    "MSBR-like core; circulation lowers beta_eff so periods are shorter")
ax.semilogy(rhos * 1e5, Ts, color=pl.SERIES[0], label="stationary fuel")
ax.semilogy(rhos * 1e5, Tc, color=pl.SERIES[1], label="circulating fuel")
ax.set_xlabel("step reactivity [pcm]")
ax.set_ylabel("stable period [s]")
ax.legend()
S.add_figure(pl.save(fig, S.path("inhour.png")), "Period curves from the eigenvalues of the kinetics system.")

# --- transients -------------------------------------------------------------
t_end = 200.0
step = pk_c.run(t_end, rho_ext=lambda t: 30e-5 if t > 5 else 0.0, max_step=0.2)
step_s = pk_s.run(t_end, rho_ext=lambda t: 30e-5 if t > 5 else 0.0, max_step=0.2)
fig, ax = pl.figure(1, 2, (11, 4.0), "Transient 1: +30 pcm reactivity step with feedback")
ax[0].plot(step_s.t, step_s.n, color=pl.SERIES[0], label="stationary fuel")
ax[0].plot(step.t, step.n, color=pl.SERIES[1], label="circulating fuel")
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("relative power")
ax[0].legend()
ax[1].plot(step.t, step.T_fuel - step.T_fuel[0], color=pl.SERIES[1], label="fuel salt")
ax[1].plot(step.t, step.T_graphite - step.T_graphite[0], color=pl.SERIES[2], label="graphite")
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("temperature rise [K] (circulating)")
ax[1].legend()
S.add_figure(pl.save(fig, S.path("transient_step.png")),
             "Negative temperature feedback turns the excursion over and settles at a new power level.")

coast = lambda t: 1.0 if t < 5 else max(0.05, 1.0 / (1.0 + (t - 5) / 8.0))
ulof = pk_c.run(400.0, flow=coast, max_step=0.2)
fig, ax = pl.figure(2, 2, (11, 7.0), "Transient 2: unprotected loss of flow (pump coast-down)",
                    "No scram; flow coasts down to 5% natural circulation")
ax[0, 0].plot(ulof.t, ulof.flow, color=pl.SERIES[0])
ax[0, 0].set_ylabel("flow / nominal")
ax[0, 1].plot(ulof.t, ulof.n, color=pl.SERIES[1])
ax[0, 1].set_ylabel("relative power")
ax[1, 0].plot(ulof.t, ulof.T_fuel, color=pl.SERIES[1], label="fuel salt (core avg)")
ax[1, 0].plot(ulof.t, ulof.T_out, color=pl.SERIES[3], label="core outlet")
ax[1, 0].set_ylabel("temperature [K]")
ax[1, 0].legend()
ax[1, 1].plot(ulof.t, 1e5 * ulof.rho_drift, color=pl.SERIES[2],
              label="precursor drift (delayed source)")
ax[1, 1].plot(ulof.t, 1e5 * ulof.rho_feedback, color=pl.SERIES[1], label="temperature feedback")
ax[1, 1].plot(ulof.t, 1e5 * (ulof.rho_feedback + ulof.rho_drift), color=pl.SERIES[0], label="sum")
ax[1, 1].set_ylabel("reactivity [pcm]")
ax[1, 1].legend(loc="center right", bbox_to_anchor=(1.0, 0.64), fontsize=8)
for a in ax.ravel():
    a.set_xlabel("time [s]")
S.add_figure(pl.save(fig, S.path("transient_ulof.png")),
             "Slower flow keeps precursors in-core (positive reactivity); fuel heating compensates and power falls to match heat removal.")

T_sec0 = th.T_secondary
over = pk_c.run(300.0, max_step=0.2,
                T_secondary=lambda t: T_sec0 - 40.0 * min(max((t - 10.0) / 20.0, 0.0), 1.0))
fig, ax = pl.figure(1, 2, (11, 4.0), "Transient 3: overcooling (secondary temperature -40 K)")
ax[0].plot(over.t, over.n, color=pl.SERIES[1])
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("relative power")
ax[1].plot(over.t, over.T_in, color=pl.SERIES[0], label="core inlet")
ax[1].plot(over.t, over.T_fuel, color=pl.SERIES[1], label="fuel salt (core avg)")
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("temperature [K]")
ax[1].legend()
S.add_figure(pl.save(fig, S.path("transient_overcooling.png")),
             "The reactor follows the load: colder inlet salt adds reactivity and power rises until temperatures recover.")

scram = lambda t: -2000e-5 if t > 2 else 0.0
sc_c = pk_c.run(120.0, rho_ext=scram, max_step=0.05, t_eval=np.linspace(0, 120, 1500))
sc_s = pk_s.run(120.0, rho_ext=scram, max_step=0.05, t_eval=np.linspace(0, 120, 1500))
fig, ax = pl.figure(1, 2, (11, 4.0), "Transient 4: scram (-2000 pcm) and decay heat")
ax[0].semilogy(sc_s.t, sc_s.n, color=pl.SERIES[0], label="stationary fuel")
ax[0].semilogy(sc_c.t, sc_c.n, color=pl.SERIES[1], label="circulating fuel")
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("relative fission power")
ax[0].legend()
tt = np.logspace(0, 7, 200)
ax[1].loglog(tt, decay_heat_fraction(tt, 3.15e7), color=pl.SERIES[2])
ax[1].set_xlabel("time after shutdown [s]")
ax[1].set_ylabel("decay heat / operating power")
ax[1].set_title("decay heat (1 year operation)", loc="left")
S.add_figure(pl.save(fig, S.path("transient_scram.png")),
             "Circulation removes precursors from the core, so fission power falls faster after scram.")
def dT_ext(res):
    d = res.T_fuel - res.T_fuel[0]
    return round(float(d[np.argmax(np.abs(d))]), 1)


S.add_table("Transient summary (MSBR-like)", ["transient", "peak power", "final power",
                                              "extreme fuel dT [K]"], [
    ["+30 pcm step (circulating)", round(step.n.max(), 3), round(step.n[-1], 3), dT_ext(step)],
    ["+30 pcm step (stationary)", round(step_s.n.max(), 3), round(step_s.n[-1], 3), dT_ext(step_s)],
    ["unprotected loss of flow", round(ulof.n.max(), 3), round(ulof.n[-1], 3), dT_ext(ulof)],
    ["overcooling -40 K", round(over.n.max(), 3), round(over.n[-1], 3), dT_ext(over)],
    ["scram -2000 pcm (circulating)", round(sc_c.n.max(), 3), f"{sc_c.n[-1]:.2e}", dT_ext(sc_c)]])
S.set(rho0_circulation_pcm=pk_c.rho0 * 1e5, generation_time_s=r.generation_time)
S.save()
