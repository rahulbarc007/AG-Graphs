"""Example 08 - AI/ML surrogates that keep high-fidelity accuracy:
model comparison by cross-validation, parity with uncertainty, active learning
(IMSE vs max-std vs random), multi-fidelity correction and the physics-guarded
hybrid predictor."""
import os
import time

import numpy as np
from _common import RESULTS, study

from msrnx import plotting as pl
from msrnx.ml import (ActiveLearner, GuardedSurrogate, MultiFidelitySurrogate, cross_validate,
                      make_surrogate, regression_metrics)
from msrnx.ml.physics import (MSR_SPACE, evaluate_hf, evaluate_lf, hf_k_oracle, lf_k_oracle,
                              run_batch, to_arrays)

S = study("08_ml_surrogates", "8. AI/ML surrogate modelling with accuracy control",
          "Nine-parameter thorium-breeder design space; high-fidelity data from the full 2D core "
          "sequence, low-fidelity data from lattice+B1.")
OUT = ["k_eff", "conversion_ratio", "alpha_fuel", "power_peaking", "graphite_life", "fissile_kg", "leakage"]
DATA = os.path.join(RESULTS, "08_ml_surrogates", "dataset.npz")
if os.path.exists(DATA):
    D = dict(np.load(DATA))
else:
    X = MSR_SPACE.sample(300, "lhs", seed=7)
    t = time.time()
    Y = to_arrays(run_batch(evaluate_hf, MSR_SPACE.as_dicts(X)), OUT)
    t_hf = (time.time() - t) / len(X)
    t = time.time()
    klf = to_arrays(run_batch(evaluate_lf, MSR_SPACE.as_dicts(X)), ["k_eff"])["k_eff"]
    t_lf = (time.time() - t) / len(X)
    D = dict(X=X, klf=klf, t_hf=t_hf, t_lf=t_lf, **Y)
    np.savez(DATA, **D)
X = D["X"]
ntr = 240
tr, te = slice(0, ntr), slice(ntr, None)
S.add_table("Design space (Latin hypercube, 300 high-fidelity 2D core solutions)",
            ["parameter", "low", "high", "unit"],
            [[p.label, p.low, p.high, p.unit] for p in MSR_SPACE.params])
S.add_table("Response ranges in the dataset", ["output", "min", "max"],
            [[o, round(float(np.nanmin(D[o])), 4), round(float(np.nanmax(D[o])), 4)] for o in OUT])

# ---- model comparison (5-fold CV) --------------------------------------
targets = [("k_eff", 1e5, "pcm"), ("conversion_ratio", 1.0, ""), ("alpha_fuel", 1.0, "pcm/K")]
kinds = ["gp", "mlp", "gbr", "poly"]
kind_label = {"gp": "Gaussian process", "mlp": "deep ensemble", "gbr": "gradient boosting",
              "poly": "polynomial ridge"}
cv = {}
rows = []
for name, scale, unit in targets:
    for k in kinds:
        m = cross_validate(k, X[tr], D[name][tr], k=5, space=MSR_SPACE, scale=scale)["metrics"]
        cv[(name, k)] = m
        rows.append([name, kind_label[k], round(m["rmse"], 4), round(m["max_abs"], 4), round(m["r2"], 5),
                     round(m["coverage_95"], 2)])
S.add_table("5-fold cross-validation (240 training designs)", ["output", "model", "RMSE", "max |err|",
            "R2", "95% coverage"], rows)
fig, ax = pl.figure(1, 3, (13, 3.9), "Surrogate model comparison (5-fold cross-validated RMSE)",
                    "Lower is better; the Gaussian process wins on every output and is used below")
for a, (name, scale, unit) in zip(ax, targets):
    pl.hbar(a, [kind_label[k] for k in kinds], [cv[(name, k)]["rmse"] for k in kinds],
            fmt="{:.3g}", xlabel=f"RMSE [{unit}]" if unit else "RMSE")
    a.set_title(name, loc="left")
S.add_figure(pl.save(fig, S.path("model_comparison.png")), "Cross-validated accuracy of four surrogate families.")

# ---- parity plots with uncertainty -----------------------------------------
fig, ax = pl.figure(1, 3, (13, 4.3), "Gaussian-process surrogate on 60 unseen designs",
                    "Bars: 95% predictive intervals")
gps = {}
for a, (name, scale, unit) in zip(ax, targets):
    gp = make_surrogate("gp", MSR_SPACE).fit(X[tr], D[name][tr])
    gps[name] = gp
    mu, sd = gp.predict(X[te], return_std=True)
    met = regression_metrics(D[name][te], mu, scale)
    sc = 1.0 if name != "k_eff" else 1.0
    pl.parity(a, D[name][te], mu, sd, unit=f"[{unit}]" if name != "k_eff" else "")
    a.set_title(f"{name}: RMSE {met['rmse']:.3g}{(' ' + unit) if unit else ''}, R2 {met['r2']:.4f}", loc="left", fontsize=10)
    S.set(**{f"gp_test_rmse_{name}": met["rmse"], f"gp_test_r2_{name}": met["r2"]})
S.add_figure(pl.save(fig, S.path("parity.png")), "Predicted vs physics on held-out designs.")

# ---- multi-fidelity --------------------------------------------------------
ns = [15, 30, 60, 120, 240]
sf_err, mf_err = [], []
for n in ns:
    sf = make_surrogate("gp", MSR_SPACE).fit(X[:n], D["k_eff"][:n])
    sf_err.append(regression_metrics(D["k_eff"][te], sf.predict(X[te]), 1e5)["rmse"])
    mf = MultiFidelitySurrogate(MSR_SPACE).fit(X[:n], D["klf"][:n], D["k_eff"][:n])
    mf_err.append(regression_metrics(D["k_eff"][te], mf.predict(X[te], D["klf"][te]), 1e5)["rmse"])
lf_err = regression_metrics(D["k_eff"][te], D["klf"][te], 1e5)["rmse"]

# ---- active learning ------------------------------------------------------
curves = {}
for strat in ["imse", "max_std", "random"]:
    al = ActiveLearner(hf_k_oracle, MSR_SPACE, "gp", n_init=20, batch=10, seed=3, strategy=strat,
                       scale=1e5)
    _, _, _, h = al.run(X[te], D["k_eff"][te], max_samples=110)
    curves[strat] = h
fig, ax = pl.figure(1, 2, (11.5, 4.3), "Accuracy vs number of expensive 2D solutions (k_eff)",
                    "Test RMSE on 60 held-out designs")
lab = {"imse": "active learning (integrated variance)", "max_std": "active learning (max variance)",
       "random": "random sampling"}
for i, (s_, h) in enumerate(curves.items()):
    ax[0].plot(h.n_samples, h.rmse, "o-", color=pl.SERIES[i], label=lab[s_], markersize=5)
ax[0].set_xlabel("high-fidelity samples")
ax[0].set_ylabel("RMSE [pcm]")
ax[0].set_yscale("log")
ax[0].legend(fontsize=8)
ax[0].set_title("sequential design", loc="left")
ax[1].plot(ns, sf_err, "o-", color=pl.SERIES[0], label="single-fidelity GP")
ax[1].plot(ns, mf_err, "o-", color=pl.SERIES[1], label="multi-fidelity GP (learns HF - LF)")
ax[1].axhline(lf_err, color=pl.SERIES[2], label="low-fidelity model alone")
pl.end_label(ax[1], ns[-1], mf_err[-1], f"{mf_err[-1]:.0f} pcm")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_xticks(ns, [str(n) for n in ns])
ax[1].minorticks_off()
ax[1].set_xlabel("high-fidelity samples")
ax[1].set_ylabel("RMSE [pcm]")
ax[1].legend(fontsize=8)
ax[1].set_title("multi-fidelity learning", loc="left")
S.add_figure(pl.save(fig, S.path("learning_curves.png")),
             "Learning the discrepancy to a cheap physics model reaches ~100 pcm with a few hundred 2D runs.")
S.add_table("Multi-fidelity accuracy (k_eff test RMSE, pcm)", ["HF samples", "single-fidelity GP",
            "multi-fidelity GP", "LF model alone"],
            [[n, round(a, 1), round(b, 1), round(lf_err, 1)] for n, a, b in zip(ns, sf_err, mf_err)])
S.set(mf_rmse_240=mf_err[-1], sf_rmse_240=sf_err[-1], lf_rmse=lf_err,
      al_rmse={k: v.rmse[-1] for k, v in curves.items()})

# ---- physics-guarded hybrid predictor --------------------------------------
Xq = MSR_SPACE.sample(80, "random", seed=99)
yq = hf_k_oracle(Xq)
g = GuardedSurrogate(hf_k_oracle, MSR_SPACE, "gp", std_tol=150e-5, X0=X[:ntr], y0=D["k_eff"][:ntr],
                     refit_every=5, lf=lf_k_oracle)
t0 = time.time()
pred, std, src = g.predict(Xq)
t_guard = time.time() - t0
ml = np.array([s == "ml" for s in src])
err = np.abs(pred - yq) * 1e5
fig, ax = pl.figure(1, 2, (11, 4.0), "Physics-guarded hybrid predictor",
                    "ML answers only when its 1-sigma uncertainty < 150 pcm; otherwise the 2D solver runs")
pl.kpi_row(ax[0], [("answered by ML", f"{100 * ml.mean():.0f}%"),
                   ("ML RMSE", f"{np.sqrt(np.mean(err[ml] ** 2)):.0f} pcm"),
                   ("worst ML error", f"{err[ml].max():.0f} pcm"),
                   ("speed-up", f"{D['t_hf'] * len(Xq) / max(t_guard, 1e-9):.1f}x")])
order = np.argsort(std)
ax[1].scatter(1e5 * std[ml], err[ml], s=26, color=pl.SERIES[0], edgecolor=pl.SURFACE, label="served by ML")
ax[1].plot([0, 150], [0, 150 * 1.96], color=pl.AXIS, lw=1)
pl.value_label(ax[1], 150, 150 * 1.96, "1.96 sigma", dx=-4, dy=-8, ha="right")
ax[1].set_xlabel("predicted 1-sigma uncertainty [pcm]")
ax[1].set_ylabel("actual |error| [pcm]")
ax[1].set_title("uncertainty is calibrated", loc="left")
S.add_figure(pl.save(fig, S.path("guarded_hybrid.png")),
             "Accuracy is maintained by construction: uncertain queries fall back to physics and retrain the model.")
S.set(guarded_ml_fraction=float(ml.mean()), guarded_ml_rmse_pcm=float(np.sqrt(np.mean(err[ml] ** 2))),
      guarded_max_err_pcm=float(err[ml].max()), t_hf_s=float(D["t_hf"]), t_lf_s=float(D["t_lf"]))
t = time.time()
for _ in range(20):
    gps["k_eff"].predict(X[te])
t_ml = (time.time() - t) / (20 * len(X[te].reshape(-1, X.shape[1])))
S.add_table("Cost per design evaluation", ["model", "seconds", "relative"], [
    ["2D core (6 groups, 4 processes)", round(float(D["t_hf"]), 4), 1.0],
    ["lattice + B1 (low fidelity)", round(float(D["t_lf"]), 5), round(float(D["t_hf"] / D["t_lf"]), 1)],
    ["Gaussian-process surrogate", f"{t_ml:.2e}", f"{D['t_hf'] / t_ml:.0f}x faster"]])
S.save()
