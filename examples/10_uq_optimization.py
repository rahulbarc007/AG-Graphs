"""Example 10 - Uncertainty quantification and AI-driven design optimization:
nuclear-data sensitivities and uncertainty (sandwich rule), Monte Carlo
propagation of design tolerances through the surrogate, constrained Bayesian
optimization and a surrogate Pareto front verified with physics."""
import os

import numpy as np
from _common import RESULTS, study

from msrnx import plotting as pl
from msrnx.core import msbr_design, solve_core
from msrnx.ml import (DEFAULT_ND_UNCERTAINTY, bayesian_optimization, make_surrogate, monte_carlo,
                      nuclear_data_sensitivity, surrogate_optimize)
from msrnx.ml.physics import MSR_SPACE, evaluate_hf, run_batch, to_arrays

S = study("10_uq_optimization", "10. Uncertainty quantification and design optimization",
          "Direct-perturbation sensitivities on the full 2D core, Monte Carlo propagation through "
          "Gaussian-process surrogates, and constrained Bayesian optimization.")

# ---- nuclear data sensitivity / uncertainty ------------------------------------
d = msbr_design()
kfn = lambda pert: solve_core(d, "6g", 12.0, adjoint=False, n_modes=1, perturb=pert).k_eff
sens = nuclear_data_sensitivity(kfn, rel_step=0.02)
nz = np.abs(sens.S) > 1e-6
lab_all = [f"{n} {rx}" for n, rx in sens.keys]
lab = [l for l, m in zip(lab_all, nz) if m]
fig, ax = pl.figure(1, 2, (12.5, 5.0), "Nuclear-data sensitivity and uncertainty of k_eff (MSBR-like)",
                    "S = (dk/k)/(dsigma/sigma); contributions use illustrative 1-sigma data uncertainties")
pl.hbar(ax[0], lab, sens.S[nz], fmt="{:+.3f}", xlabel="sensitivity coefficient")
ax[0].set_title("sensitivity coefficients", loc="left")
pl.hbar(ax[1], lab, sens.contribution_pcm[nz], color=pl.SERIES[1], fmt="{:.0f}", xlabel="1-sigma k uncertainty [pcm]")
ax[1].set_title(f"uncertainty contributions (total {sens.total_pcm:.0f} pcm)", loc="left")
S.add_figure(pl.save(fig, S.path("nd_sensitivity.png")),
             "Th-232 capture dominates, then U-233 nu/fission/capture; graphite and carrier-salt (Li-7, F-19, Be-9) data add the rest.")
S.add_table("Nuclear-data sensitivity (2D core, 2% perturbations)", ["nuclide reaction", "S", "rel. unc.",
            "contribution [pcm]"], [[l, round(float(s), 4), unc, round(float(c), 1)] for l, s, unc, c in
                                     sorted(zip(lab, sens.S[nz], sens.rel_unc[nz], sens.contribution_pcm[nz]),
                                            key=lambda t: -t[3])])
S.set(nd_total_uncertainty_pcm=sens.total_pcm)

# ---- Monte Carlo propagation of design tolerances through surrogates --------------
DATA = os.path.join(RESULTS, "08_ml_surrogates", "dataset.npz")
if not os.path.exists(DATA):
    raise SystemExit("Run 08_ml_surrogates.py first (it generates the dataset).")
D = dict(np.load(DATA))
X = D["X"]
gps = {o: make_surrogate("gp", MSR_SPACE).fit(X, D[o]) for o in ["k_eff", "conversion_ratio", "alpha_fuel",
                                                                  "fissile_kg"]}
nom = {p.name: p.nominal for p in MSR_SPACE.params}
nom.update({"salt_fraction": 0.15, "uf4_pct": 0.30})
tol = {"salt_fraction": ("normal", 0.15, 0.003), "uf4_pct": ("normal", 0.30, 0.004),
       "thf4_pct": ("normal", nom["thf4_pct"], 0.10), "channel_radius": ("normal", nom["channel_radius"], 0.02),
       "T_fuel": ("normal", nom["T_fuel"], 10.0), "li6_ppm": ("lognormal", np.log(nom["li6_ppm"]), 0.3)}


def fk(samples):
    n = len(next(iter(samples.values())))
    Z = np.column_stack([samples.get(p.name, np.full(n, nom[p.name])) for p in MSR_SPACE.params])
    return gps["k_eff"].predict(Z)


mc = monte_carlo(fk, tol, n=20000, seed=4)
fig, ax = pl.figure(1, 2, (11, 4.2), "Propagation of manufacturing and operating tolerances",
                    "20,000 Monte Carlo samples through the GP surrogate (physics would need ~1 hour)")
ax[0].hist(mc["y"], bins=60, color=pl.SERIES[0], edgecolor=pl.SURFACE, linewidth=0.5)
ax[0].axvline(mc["p05"], color=pl.AXIS, lw=1)
ax[0].axvline(mc["p95"], color=pl.AXIS, lw=1)
pl.value_label(ax[0], mc["p95"], ax[0].get_ylim()[1] * 0.9, f"5-95%: {mc['p05']:.4f}-{mc['p95']:.4f}", dx=4)
ax[0].set_xlabel("k_eff")
ax[0].set_ylabel("samples")
ax[0].set_title(f"k_eff = {mc['mean']:.4f} +/- {mc['std'] * 1e5:.0f} pcm", loc="left")
pl.hbar(ax[1], list(mc["src"].keys()), list(mc["src"].values()), fmt="{:+.3f}",
        xlabel="standardized regression coefficient")
ax[1].set_title("which tolerance matters", loc="left")
S.add_figure(pl.save(fig, S.path("tolerance_uq.png")), "Fissile loading and salt-fraction tolerances drive the k_eff spread.")
S.set(mc_k_mean=mc["mean"], mc_k_std_pcm=mc["std"] * 1e5)

# ---- constrained Bayesian optimization -------------------------------------------
cons = [("k_eff", ">=", 1.02), ("alpha_fuel", "<=", -3.0)]


def ev(Xn):
    return to_arrays(run_batch(evaluate_hf, MSR_SPACE.as_dicts(Xn)), ["k_eff", "conversion_ratio", "alpha_fuel",
                                                                    "fissile_kg"])


bo = bayesian_optimization(ev, MSR_SPACE, "conversion_ratio", cons, minimize=False, n_init=20, n_iter=30, seed=8)
Xr = MSR_SPACE.sample(50, "random", seed=8)
Yr = ev(Xr)
best_r = []
cur = -np.inf
for i in range(len(Xr)):
    if Yr["k_eff"][i] >= 1.02 and Yr["alpha_fuel"][i] <= -3.0:
        cur = max(cur, Yr["conversion_ratio"][i])
    best_r.append(cur)
fig, ax = pl.figure(1, 1, (7.6, 4.2), "Constrained Bayesian optimization of the breeding ratio",
                    "Maximize CR subject to k_eff >= 1.02 and alpha_fuel <= -3 pcm/K (each point = one 2D core solve)")
n_bo = np.arange(20, 20 + len(bo.best_feasible))
ax.plot(n_bo, bo.best_feasible, "o-", color=pl.SERIES[0], label="Bayesian optimization (EI x P(feasible))", markersize=4)
ax.plot(np.arange(1, len(best_r) + 1), best_r, color=pl.SERIES[1], label="random search")
pl.end_label(ax, n_bo[-1], bo.best_feasible[-1], f"CR = {bo.best_feasible[-1]:.3f}")
ax.set_xlabel("number of physics evaluations")
ax.set_ylabel("best feasible conversion ratio")
ax.legend(loc="lower right")
S.add_figure(pl.save(fig, S.path("bayesian_optimization.png")), "Bayesian optimization finds better feasible designs with fewer expensive runs.")
ib = int(np.argmax([bo.Y["conversion_ratio"][i] if (bo.Y["k_eff"][i] >= 1.02 and bo.Y["alpha_fuel"][i] <= -3) else -1
                    for i in range(len(bo.X))]))
best = dict(zip(MSR_SPACE.names, bo.X[ib]))
S.add_table("Best design found by Bayesian optimization (verified by physics)", ["quantity", "value"],
            [[MSR_SPACE.params[i].label, round(float(bo.X[ib][i]), 4)] for i in range(MSR_SPACE.dim)] +
            [["k_eff", round(float(bo.Y["k_eff"][ib]), 5)], ["conversion ratio", round(float(bo.Y["conversion_ratio"][ib]), 4)],
             ["alpha_fuel [pcm/K]", round(float(bo.Y["alpha_fuel"][ib]), 3)],
             ["U-233 inventory [kg]", round(float(bo.Y["fissile_kg"][ib]), 1)]])
S.set(bo_best_cr=float(bo.Y["conversion_ratio"][ib]), random_best_cr=float(best_r[-1]))

# ---- Pareto front: conversion ratio vs fissile inventory (robust + refinement) -------
outs = ["k_eff", "conversion_ratio", "alpha_fuel", "fissile_kg"]
targets = np.linspace(0.90, 1.20, 6)
Xtrain, Ytrain = X.copy(), {o: D[o].copy() for o in outs}


def optimize_front(models, kappa):
    pred = lambda Z: {o: g.predict(Z) for o, g in models.items()}
    pstd = lambda Z: {o: g.predict(Z, return_std=True) for o, g in models.items()}
    front = []
    for c in targets:
        front.append(surrogate_optimize(pred, MSR_SPACE, lambda p: p["fissile_kg"],
                                        constraints=[("conversion_ratio", ">=", float(c)),
                                                     ("k_eff", ">=", 1.02), ("alpha_fuel", "<=", -3.0)],
                                        seed=1, maxiter=40, popsize=12, predict_std=pstd, kappa=kappa))
    return front


passes = []
models = gps
for it in range(2):
    front = optimize_front(models, kappa=1.5)
    Xf = np.array([f["x"] for f in front])
    ver = ev(Xf)
    passes.append((front, ver))
    # trust refinement: add the verified designs to the training data and refit
    Xtrain = np.vstack([Xtrain, Xf])
    for o in outs:
        Ytrain[o] = np.append(Ytrain[o], ver[o])
    models = {o: make_surrogate("gp", MSR_SPACE).fit(Xtrain, Ytrain[o]) for o in outs}
front, ver = passes[-1]
fig, ax = pl.figure(1, 1, (7.8, 4.4), "Pareto front: breeding vs fissile inventory",
                    "Robust surrogate optima (constraints on mean - 1.5 sigma), refined once and verified with the 2D solver")
ax.scatter(D["fissile_kg"], D["conversion_ratio"], s=10, color=pl.GRID, label="training designs", zorder=1)
ax.plot([f["prediction"]["fissile_kg"] for f in front], [f["prediction"]["conversion_ratio"] for f in front], "o-",
        color=pl.SERIES[0], label="surrogate optimum", zorder=3)
ax.scatter(ver["fissile_kg"], ver["conversion_ratio"], s=60, facecolor="none", edgecolor=pl.SERIES[1], linewidth=2,
           label="physics verification", zorder=4)
ax.set_xlabel("U-233 inventory [kg]")
ax.set_ylabel("conversion ratio")
ax.set_xscale("log")
ax.legend(loc="lower right")
S.add_figure(pl.save(fig, S.path("pareto.png")),
             "Every extra unit of breeding costs fissile inventory; the 2D solver confirms the surrogate optima.")
rows = []
for p_i, (fr, vv) in enumerate(passes):
    for i, (c, f) in enumerate(zip(targets, fr)):
        feas = (vv["k_eff"][i] >= 1.02) and (vv["alpha_fuel"][i] <= -3.0) and (vv["conversion_ratio"][i] >= c - 0.005)
        rows.append([p_i + 1, round(float(c), 3), round(f["prediction"]["conversion_ratio"], 4),
                     round(float(vv["conversion_ratio"][i]), 4), round(f["prediction"]["k_eff"], 4),
                     round(float(vv["k_eff"][i]), 4), round(f["prediction"]["fissile_kg"], 0),
                     round(float(vv["fissile_kg"][i]), 0), "yes" if feas else "no"])
S.add_table("Pareto designs: surrogate vs physics (pass 1 = initial, pass 2 = after refinement)",
            ["pass", "CR target", "CR (ML)", "CR (physics)", "k_eff (ML)", "k_eff (physics)",
             "U-233 kg (ML)", "U-233 kg (physics)", "feasible (physics)"], rows)
S.set(pareto_feasible_fraction=float(np.mean([r[-1] == "yes" for r in rows if r[0] == 2])))
S.save()
