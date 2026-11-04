"""Example 09 - Parameter reduction: global sensitivity (Sobol, Morris), active
subspaces, reduced-input surrogates, POD reduced-order model of the power
distribution and cross-section functionalization (PCA)."""
import os

import numpy as np
from _common import RESULTS, study

from msrnx import plotting as pl
from msrnx.core import msbr_design
from msrnx.ml import (PODModel, STATE_SPACE, XS_NAMES_2G, XSFunctionalization, active_subspace,
                      make_surrogate, morris_screening, reduce_parameters, regression_metrics,
                      sobol_indices, xs_table)
from msrnx.ml.physics import MSR_SPACE, evaluate_hf_field, run_batch

S = study("09_parameter_reduction", "9. Parameter reduction and reduced-order modelling",
          "Variance-based sensitivity on the GP surrogate identifies the few design parameters "
          "that matter; POD and PCA compress fields and cross-section tables.")
DATA = os.path.join(RESULTS, "08_ml_surrogates", "dataset.npz")
if not os.path.exists(DATA):
    raise SystemExit("Run 08_ml_surrogates.py first (it generates the dataset).")
D = dict(np.load(DATA))
X = D["X"]
labels = [p.label for p in MSR_SPACE.params]

# ---- Sobol indices on the surrogate ------------------------------------------
sob = {}
for out in ["k_eff", "conversion_ratio"]:
    gp = make_surrogate("gp", MSR_SPACE).fit(X, D[out])
    sob[out] = sobol_indices(lambda Z: gp.predict(Z), MSR_SPACE, n=4096, seed=1)
fig, ax = pl.figure(1, 2, (12, 4.6), "Global sensitivity: Sobol indices (on the GP surrogate)",
                    "Total-order index S_T = share of output variance involving each parameter (95% bootstrap CI)")
for a, out in zip(ax, sob):
    r = sob[out]
    pl.hbar(a, labels, r.ST, errors=r.ST_conf, fmt="{:.3f}", xlabel="total Sobol index S_T")
    a.set_title(out, loc="left")
S.add_figure(pl.save(fig, S.path("sobol.png")),
             "UF4, ThF4 and salt fraction carry ~95% of the variance; fuel temperature, H/D and reflector thickness have S_T < 0.002.")
rows = []
for i, lab in enumerate(labels):
    rows.append([lab, round(sob["k_eff"].S1[i], 4), round(sob["k_eff"].ST[i], 4),
                 round(sob["conversion_ratio"].S1[i], 4), round(sob["conversion_ratio"].ST[i], 4)])
S.add_table("Sobol indices", ["parameter", "k_eff S1", "k_eff ST", "CR S1", "CR ST"], rows)
keep, drop = reduce_parameters(sob["k_eff"], threshold=0.01)
keep_acc, drop_acc = reduce_parameters(sob["k_eff"], target_rmse=float(np.std(D["k_eff"])) * 0.03)
S.set(k_eff_influential=keep, k_eff_negligible=drop, k_eff_variance_pcm2=sob["k_eff"].variance * 1e10,
      droppable_for_3pct_accuracy=drop_acc)

# ---- Morris screening ---------------------------------------------------------
gpk = make_surrogate("gp", MSR_SPACE).fit(X, D["k_eff"])
mor = morris_screening(lambda Z: gpk.predict(Z) * 1e5, MSR_SPACE, r=60, seed=2)
fig, ax = pl.figure(1, 1, (8.6, 4.6), "Morris elementary-effects screening (k_eff)",
                    "mu*: overall influence; sigma: non-linearity / interactions")
ax.scatter(mor.mu_star, mor.sigma, s=40, color=pl.SERIES[0], edgecolor=pl.SURFACE, zorder=3)
for i, lab in enumerate(labels):
    if mor.mu_star[i] > 0.04 * mor.mu_star.max():
        pl.value_label(ax, mor.mu_star[i], mor.sigma[i], lab, dx=5, dy=3, fontsize=8)
ax.set_xlabel("mu* [pcm per unit-range]")
ax.set_ylabel("sigma [pcm]")
S.add_figure(pl.save(fig, S.path("morris.png")), "Screening with 600 surrogate evaluations reproduces the Sobol ranking.")

# ---- active subspace ---------------------------------------------------------
asr = active_subspace(lambda Z: gpk.predict(Z) * 1e5, MSR_SPACE, n=500, seed=3)
fig, ax = pl.figure(1, 2, (11, 4.2), "Active subspace of k_eff",
                    "Eigen-decomposition of E[grad f grad f^T]: one direction explains most variation")
ax[0].semilogy(np.arange(1, len(asr.eigenvalues) + 1), asr.eigenvalues / asr.eigenvalues[0], "o-",
               color=pl.SERIES[0])
ax[0].set_xlabel("eigenvalue index")
ax[0].set_ylabel("normalized eigenvalue")
ax[0].set_title("spectral gap", loc="left")
y1 = asr.reduced_coordinates(1)[:, 0]
ax[1].scatter(y1, asr.values / 1e5, s=12, color=pl.SERIES[0], edgecolor="none", alpha=0.7)
ax[1].set_xlabel("active variable w1^T x")
ax[1].set_ylabel("k_eff")
ax[1].set_title("sufficient summary plot", loc="left")
S.add_figure(pl.save(fig, S.path("active_subspace.png")),
             "A single linear combination of the nine inputs captures the trend of k_eff (1-D reduced model).")
w1 = asr.eigenvectors[:, 0]
S.add_table("Leading active direction w1 (unit-cube coordinates)", ["parameter", "weight"],
            [[lab, round(float(w), 4)] for lab, w in sorted(zip(labels, w1), key=lambda t: -abs(t[1]))])
S.set(eig_ratio_2_1=float(asr.eigenvalues[1] / asr.eigenvalues[0]))

# ---- reduced-input surrogates ---------------------------------------------------
order = list(np.argsort(-sob["k_eff"].ST))
tr, te = slice(0, 240), slice(240, None)
m_list, err = [], []
for m in range(1, len(order) + 1):
    idx = sorted(order[:m])
    sub = MSR_SPACE.subspace([MSR_SPACE.names[i] for i in idx])
    gp = make_surrogate("gp", sub).fit(X[tr][:, idx], D["k_eff"][tr])
    err.append(regression_metrics(D["k_eff"][te], gp.predict(X[te][:, idx]), 1e5)["rmse"])
    m_list.append(m)
fig, ax = pl.figure(1, 1, (7.6, 4.2), "Reduced-input surrogates (k_eff)",
                    "Keep the m most influential parameters (by Sobol S_T), drop the rest")
ax.semilogy(m_list, err, "o-", color=pl.SERIES[0])
for m, e in zip(m_list, err):
    if m in (1, 3, 5, len(m_list)):
        pl.value_label(ax, m, e, f"{e:.0f} pcm", dy=9, ha="center")
ax.set_xticks(m_list)
ax.set_xlabel("number of retained parameters m")
ax.set_ylabel("test RMSE [pcm]")
S.add_figure(pl.save(fig, S.path("reduced_inputs.png")),
             "The top 3 inputs carry 95% of the variance, but k spans ~+/-12,000 pcm, so even S_T = 0.002 is ~500 pcm: "
             "drop an input only if sqrt(S_T * Var) is below the accuracy target.")
S.add_table("Reduced-input surrogate accuracy", ["m", "parameters kept", "test RMSE [pcm]"],
            [[m, ", ".join(labels[i] for i in sorted(order[:m])) if m <= 4 else f"top {m}", round(e, 1)]
             for m, e in zip(m_list, err)])

# ---- POD of 2D power distributions -------------------------------------------------
Xp = MSR_SPACE.sample(120, "lhs", seed=11)
res = run_batch(evaluate_hf_field, MSR_SPACE.as_dicts(Xp))
Smat = np.array([r["power_map"] for r in res])
pod = PODModel(energy=0.9999, kind="gp", space=MSR_SPACE).fit(Xp[:96], Smat[:96])
P = pod.predict(Xp[96:])
e_pred = np.abs(P - Smat[96:]).max(axis=1)
ranks = np.arange(1, 13)
proj_err = []
for rnk in ranks:
    V = pod.modes if rnk <= pod.rank else None
    U, s, Vt = np.linalg.svd(Smat[:96] - pod.mean, full_matrices=False)
    Vr = Vt[:rnk]
    rec = (Smat[96:] - pod.mean) @ Vr.T @ Vr + pod.mean
    proj_err.append(np.abs(rec - Smat[96:]).max(axis=1).mean())
fig, ax = pl.figure(1, 2, (11, 4.0), "POD reduced-order model of the power distribution",
                    f"600-cell (r,z) power maps compressed to {pod.rank} modes ({pod.compression_ratio():.0f}x)")
ax[0].semilogy(np.arange(1, 21), pod.singular_values[:20] / pod.singular_values[0], "o-", color=pl.SERIES[0])
ax[0].set_xlabel("mode index")
ax[0].set_ylabel("normalized singular value")
ax[0].set_title("energy spectrum", loc="left")
ax[1].semilogy(ranks, proj_err, "o-", color=pl.SERIES[1])
ax[1].set_xlabel("number of POD modes")
ax[1].set_ylabel("mean max |dq/q_avg| (test)")
ax[1].set_title("reconstruction error", loc="left")
S.add_figure(pl.save(fig, S.path("pod_spectrum.png")), "A handful of modes reproduces any power distribution in the design space.")
nr, nz = 20, 30
fig, ax = pl.figure(1, 4, (13, 4.6), "POD modes and an ML-predicted power map",
                    "Normalized core coordinates (r/R, z/H); modes 1-3 and a held-out design")
for j in range(3):
    im = ax[j].imshow(pod.modes[j].reshape(nr, nz).T, origin="lower", extent=[0, 1, 0, 1],
                      cmap=pl.DIV, aspect="auto", vmin=-np.abs(pod.modes[j]).max(), vmax=np.abs(pod.modes[j]).max())
    ax[j].set_title(f"mode {j + 1}", loc="left")
    ax[j].grid(False)
    ax[j].set_xlabel("r/R")
ax[0].set_ylabel("z/H")
k = int(np.argmax(e_pred))
im = ax[3].imshow(P[k].reshape(nr, nz).T, origin="lower", extent=[0, 1, 0, 1], cmap=pl.SEQ, aspect="auto")
ax[3].set_title(f"predicted (worst case err {e_pred[k]:.3f})", loc="left", fontsize=9.5)
ax[3].grid(False)
ax[3].set_xlabel("r/R")
S.add_figure(pl.save(fig, S.path("pod_modes.png")), "Mode 1 is the fundamental shape; higher modes encode radial/axial tilts.")
S.set(pod_rank=pod.rank, pod_compression=pod.compression_ratio(), pod_mean_max_err=float(e_pred.mean()),
      pod_worst_err=float(e_pred.max()))

# ---- cross-section functionalization ---------------------------------------------------
d = msbr_design()
Xs = STATE_SPACE.sample(160, "lhs", seed=5)
Yx = xs_table(d, Xs)
xf = XSFunctionalization(kind="poly").fit(Xs[:120], Yx[:120])
rel = xf.relative_errors(Xs[120:], Yx[120:])
xg = XSFunctionalization(kind="gp").fit(Xs[:120], Yx[:120])
relg = xg.relative_errors(Xs[120:], Yx[120:])
fig, ax = pl.figure(1, 2, (11, 4.0), "Cross-section functionalization (2-group constants)",
                    "PCA of the standardized table + regression of the scores on 4 state variables")
ax[0].semilogy(np.arange(1, len(xf.singular_values) + 1), 1 - xf.cumulative_variance + 1e-16, "o-", color=pl.SERIES[0])
ax[0].set_xlabel("number of principal components")
ax[0].set_ylabel("unexplained variance")
ax[0].set_title(f"{xf.rank} components keep 99.999% of the variance", loc="left", fontsize=10)
pl.group_bars(ax[1], XS_NAMES_2G, {"polynomial": 100 * rel.max(axis=0), "Gaussian process": 100 * relg.max(axis=0)},
              fmt="{:.2f}", ylabel="max relative error [%]", label_values=False)
ax[1].set_title("held-out accuracy per constant", loc="left")
S.add_figure(pl.save(fig, S.path("xs_functionalization.png")),
             f"{len(XS_NAMES_2G)} tabulated constants on a 4-D state grid replaced by {xf.rank} smooth functions.")
S.add_table("XS functionalization max relative error [%] (40 held-out states)", ["constant", "polynomial", "GP"],
            [[n, round(100 * a, 4), round(100 * b, 4)] for n, a, b in zip(XS_NAMES_2G, rel.max(0), relg.max(0))])
S.save()
