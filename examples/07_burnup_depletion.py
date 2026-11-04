"""Example 07 - Burnup / depletion of a circulating fuel salt: online
reprocessing, criticality-controlled refuelling, breeding ratio, fissile
inventory, Pa-233, fission-product poisoning, actinides, tritium, and
ML-accelerated depletion."""
import numpy as np
from _common import study

from msrnx import plotting as pl
from msrnx.core import msbr_design, tmsr_leu_design
from msrnx.depletion import (CoreTransport, DepletionSolver, FeedControl, LatticeTransport,
                             Reprocessing)
from msrnx.ml import MLAcceleratedTransport

S = study("07_burnup_depletion", "7. Burnup and depletion with online fuel processing",
          "CE/CM predictor-corrector with CRAM-16, 2D-calibrated lattice transport, fissile "
          "feed/extraction to hold k_eff = 1 and fertile make-up.")
steps = [1, 2, 4, 8, 15] + [30] * 11 + [60] * 6 + [120] * 8
d = msbr_design()
runs = {}
for rep in [Reprocessing.none(), Reprocessing.sparging_only(), Reprocessing.msbr()]:
    runs[rep.name] = DepletionSolver(d, reprocessing=rep, feed=FeedControl(target_k=1.0)).run(steps)
rows = []
for name, r in runs.items():
    t = r.time_days
    i = len(t) - 1
    net = r.fissile_feed_kg[-1] - r.fissile_feed_kg[0]
    rows.append([name, round(t[i] / 365.25, 2), round(r.burnup_MWd_kgHM[i], 1),
                 round(r.breeding_ratio[i], 4), round(r.fissile_inventory_kg()[i], 0),
                 round(-net / (t[i] - t[0]) * 365.25, 1), round(r.inventory_kg("Pa233")[i], 1),
                 round(100 * r.poison_fraction["Xe135"][i], 3), round(r.wall_time_s, 1)])
S.add_table("MSBR-like depletion (k_eff held at 1.000 by fissile feed/extraction)",
            ["processing", "years", "burnup [MWd/kgHM]", "breeding ratio", "fissile [kg]",
             "net fissile gain [kg/yr]", "Pa-233 [kg]", "Xe absorption [%]", "wall time [s]"], rows)
S.set(msbr_BR_final=float(runs["MSBR reference"].breeding_ratio[-1]))

names = list(runs)
fig, ax = pl.figure(2, 2, (11, 7.4), "Fuel-cycle evolution of the MSBR-like core",
                    "Effect of online processing on breeding and fissile balance")
for i, n in enumerate(names):
    r = runs[n]
    ty = r.time_days / 365.25
    ax[0, 0].plot(ty[1:], r.breeding_ratio[1:], color=pl.SERIES[i], label=n)
    ax[0, 1].plot(ty, r.fissile_feed_kg - r.fissile_feed_kg[0], color=pl.SERIES[i], label=n)
    ax[1, 0].plot(ty, r.inventory_kg("Pa233"), color=pl.SERIES[i], label=n)
    ax[1, 1].plot(ty[1:], 1e5 * (r.k_bos[1:] - 1.0) / r.k_bos[1:], color=pl.SERIES[i], label=n)
ax[0, 0].axhline(1.0, color=pl.AXIS, lw=1)
ax[0, 0].set_ylabel("breeding ratio")
ax[0, 1].set_ylabel("cumulative fissile feed [kg] (<0 = bred surplus)")
ax[1, 0].set_ylabel("Pa-233 in fuel salt [kg]")
ax[1, 1].set_ylabel("reactivity before refuelling [pcm]")
ax[1, 1].axhline(0.0, color=pl.AXIS, lw=1)
ax[1, 1].set_title("after the initial criticality adjustment", loc="left", fontsize=10)
for a in ax.ravel():
    a.set_xlabel("time [years]")
ax[0, 0].legend(loc="lower left")
S.add_figure(pl.save(fig, S.path("msbr_fuel_cycle.png")),
             "Only with rare-earth removal and Pa isolation does the thorium core breed (BR > 1) and produce surplus U-233.")

# poison fractions
r = runs["no processing"]
rm = runs["MSBR reference"]
fig, ax = pl.figure(1, 2, (11, 4.2), "Neutron absorption by fission products",
                    "Share of all absorptions; without processing (left) and with MSBR processing (right)")
labels = {"Xe135": "Xe-135", "Sm149": "Sm-149", "LFP_RE": "rare earths", "LFP_NM": "noble metals",
          "LFP_SS": "salt seekers", "LFP_NG": "noble gases"}
for a, rr in zip(ax, [r, rm]):
    for i, (k, lab) in enumerate(labels.items()):
        a.plot(rr.time_days / 365.25, 100 * rr.poison_fraction[k], color=pl.SERIES[i], label=lab)
    a.set_xlabel("time [years]")
    a.set_ylabel("share of absorptions [%]")
ax[0].legend(loc="upper left", fontsize=8)
S.add_figure(pl.save(fig, S.path("fp_poisoning.png")),
             "Sparging removes Xe-135 and noble gases, chemical processing removes rare earths.")

# actinides and tritium
fig, ax = pl.figure(1, 2, (11, 4.2), "Actinide build-up and tritium (MSBR processing)")
for i, n in enumerate(["U234", "U235", "U236", "Np237", "U232", "Pu239"]):
    ax[0].semilogy(rm.time_days[1:] / 365.25, np.maximum(rm.inventory_kg(n)[1:], 1e-6),
                   color=pl.SERIES[i], label=n)
ax[0].set_xlabel("time [years]")
ax[0].set_ylabel("inventory [kg]")
ax[0].legend(fontsize=8)
lam_T = np.log(2) / (12.32 * 365.25 * 86400)
ax[1].plot(rm.time_days / 365.25, rm.tritium_rate * 86400 * lam_T / 3.7e10, color=pl.SERIES[0])
ax[1].set_xlabel("time [years]")
ax[1].set_ylabel("tritium production [Ci/day]")
ax[1].set_title("Li-6 burn-out reduces tritium production", loc="left")
S.add_figure(pl.save(fig, S.path("actinides_tritium.png")),
             "U-232 builds up through Th-232(n,2n); tritium falls as the initial Li-6 burns out.")
S.set(tritium_BOL_Ci_day=float(rm.tritium_rate[0] * 86400 * lam_T / 3.7e10),
      tritium_EOL_Ci_day=float(rm.tritium_rate[-1] * 86400 * lam_T / 3.7e10))

# LEU once-through with LEU make-up feed
dl = tmsr_leu_design()
rl = DepletionSolver(dl, reprocessing=Reprocessing.sparging_only(),
                     feed=FeedControl(target_k=1.0, fissile_feed={"U235": 0.1975, "U238": 0.8025},
                                      fertile_makeup=(), allow_removal=True)).run(steps)
fig, ax = pl.figure(1, 2, (11, 4.0), "LEU-fuelled TMSR: continuous 19.75 % LEU feed")
ax[0].plot(rl.time_days / 365.25, rl.fissile_feed_kg - rl.fissile_feed_kg[0], color=pl.SERIES[0])
ax[0].set_xlabel("time [years]")
ax[0].set_ylabel("cumulative LEU feed [kg]")
for i, n in enumerate(["U235", "Pu239", "Pu240", "Pu241"]):
    ax[1].plot(rl.time_days / 365.25, rl.inventory_kg(n), color=pl.SERIES[i], label=n)
ax[1].set_xlabel("time [years]")
ax[1].set_ylabel("inventory [kg]")
ax[1].legend()
S.add_figure(pl.save(fig, S.path("leu_feed.png")), "Plutonium builds up from U-238 while LEU feed compensates burnup.")
S.set(leu_feed_kg_per_year=float((rl.fissile_feed_kg[-1] - rl.fissile_feed_kg[0]) /
                                 (rl.time_days[-1] / 365.25)))

# 2D-transport depletion check and ML acceleration
short = [5, 25, 30, 60, 60, 120, 120, 180, 180]
ref2d = DepletionSolver(d, reprocessing=Reprocessing.msbr(), feed=FeedControl(),
                        transport_model=CoreTransport(d, "6g", 12.0)).run(short)
lat = DepletionSolver(d, reprocessing=Reprocessing.msbr(), feed=FeedControl()).run(short)
acc = MLAcceleratedTransport(LatticeTransport(d), d)
mlr = DepletionSolver(d, reprocessing=Reprocessing.msbr(), feed=FeedControl(), transport_model=acc).run(steps)
ref_long = runs["MSBR reference"]
fig, ax = pl.figure(1, 2, (11, 4.0), "Transport fidelity and ML acceleration in depletion")
ax[0].plot(ref2d.time_days / 365.25, ref2d.breeding_ratio, "o-", color=pl.SERIES[0], label="2D core transport")
ax[0].plot(lat.time_days / 365.25, lat.breeding_ratio, "s--", color=pl.SERIES[1], label="2D-calibrated lattice")
ax[0].set_xlabel("time [years]")
ax[0].set_ylabel("breeding ratio")
ax[0].legend()
ax[0].set_title("reduced-order transport vs 2D", loc="left")
ax[1].plot(ref_long.time_days / 365.25, ref_long.breeding_ratio, color=pl.SERIES[0], label="physics every call")
ax[1].plot(mlr.time_days / 365.25, mlr.breeding_ratio, "--", color=pl.SERIES[1],
           label=f"ML-accelerated ({acc.physics_calls}/{acc.calls} physics calls)")
ax[1].set_xlabel("time [years]")
ax[1].set_ylabel("breeding ratio")
ax[1].legend()
ax[1].set_title("trust-gated ML transport", loc="left")
S.add_figure(pl.save(fig, S.path("depletion_fidelity_ml.png")),
             "The fast lattice model tracks the 2D model; the ML-accelerated run reproduces the physics run.")
S.add_table("Depletion transport options", ["model", "transport calls", "wall time [s]",
            "final BR", "max |dBR| vs reference"], [
    ["2D core (6 groups)", ref2d.transport_calls, round(ref2d.wall_time_s, 1), round(ref2d.breeding_ratio[-1], 4), "reference"],
    ["2D-calibrated lattice", lat.transport_calls, round(lat.wall_time_s, 1), round(lat.breeding_ratio[-1], 4),
     round(float(np.max(np.abs(lat.breeding_ratio - ref2d.breeding_ratio))), 4)],
    ["ML-accelerated lattice", f"{acc.physics_calls} physics + {acc.ml_calls} ML", round(mlr.wall_time_s, 1),
     round(mlr.breeding_ratio[-1], 4), round(float(np.max(np.abs(mlr.breeding_ratio - ref_long.breeding_ratio))), 5)]])
S.save()
