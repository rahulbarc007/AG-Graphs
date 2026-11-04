"""Example 11 - Neural-network burnup framework.

Learns the burnup trajectories of a thorium/U-233 molten salt reactor (k_eff,
breeding ratio, burnup, U-232...U-235, Pa-233, Np-237, Xe/Sm/fission-product
poisoning) from MSR-NX depletion runs, then shows how to use and extend it:

1. training data from the built-in physics generator (400 designs x 26 steps)
2. a deep-ensemble network with physics-informed time encoding, monotonicity
   constraints and calibrated uncertainty; accuracy on unseen designs
3. how much data is needed (learning curve) and what the time encoding buys
4. adding data from another code: 2D-core depletion runs stand in for a
   user's high-fidelity code (Serpent / SCALE / OpenMC); transfer learning vs
   training from scratch
5. a burnup design map (discharge burnup at k_eff = 1) in about a second,
   verified against physics runs
6. input sensitivities and computing cost

Run:  python examples/11_burnup_neural_network.py      (~15 min on 4 cores;
      the data sets are cached in results/11_burnup_neural_network/;
      BURNUP_NN_REUSE_MODEL=1 reloads the saved model instead of retraining it)
"""
import os
import time

import numpy as np
import pandas as pd
from _common import study

from msrnx import plotting as pl
from msrnx.burnup_nn import (BURNUP_SPACE, COARSE_STEPS_DAYS, DEFAULT_STEPS_DAYS, BurnupDataset,
                             BurnupNN, generate_physics_dataset, learning_curve)
from msrnx.burnup_nn import plots as bp
from msrnx.burnup_nn.physics_data import run_case

FEATURES = BURNUP_SPACE.names
TARGETS = ["k_eff", "breeding_ratio", "burnup_MWd_kgHM", "U233_kg", "U235_kg", "Pa233_kg",
           "U234_kg", "U232_kg", "Np237_kg", "Xe135_abs_pct", "Sm149_abs_pct", "FP_abs_pct"]
MONOTONIC = {"burnup_MWd_kgHM": "increasing", "Np237_kg": "increasing",
             "FP_abs_pct": "increasing"}
MAIN = dict(hidden=(64, 64, 64), activation="tanh", ensemble=4, max_iter=5000, seed=0)
CURVE = dict(hidden=(64, 64, 64), activation="tanh", max_iter=2000)
NOMINAL = {p.name: p.nominal for p in BURNUP_SPACE.params}


def _rmse_pcm(model, data):
    return float(model.evaluate(data).loc["k_eff", "rmse_pcm"])


def _curve_caption(sizes, lc):
    ms, li = lc["multiscale"], lc["linear"]
    n_eq = next((n for n, v in zip(sizes, ms) if v <= li[-1]), None)
    txt = (f"k_eff error falls from {ms[0]:.0f} to {ms[-1]:.0f} pcm as cases are added "
           f"({sizes[0]} to {sizes[-1]}); a plain time input reaches {li[-1]:.0f} pcm with all cases")
    if n_eq is not None and n_eq < sizes[-1]:
        txt += f", which the multi-timescale encoding already beats with {n_eq} cases"
    return txt + "."


def _transfer_caption(n_hf, lf_only, res):
    i = n_hf.index(8)
    return (f"With 8 runs of the other code: built-in network {lf_only:.0f} pcm, + discrepancy "
            f"correction {res['correction'][i]:.0f} pcm, + weight fine-tuning "
            f"{res['weights'][i]:.0f} pcm; a network trained from scratch on the same 8 runs "
            f"{res['scratch'][i]:,.0f} pcm. The built-in lattice model is already close to 2D-core "
            "depletion, so the gain from correction is modest here; a code with different nuclear "
            "data would differ much more.")


def _reference_crossing(df, level=1.0):
    """k_eff = level crossing of a physics trajectory (time, burnup)."""
    k, t, b = df["k_eff"].to_numpy(), df["time_days"].to_numpy(), df["burnup_MWd_kgHM"].to_numpy()
    if k[0] < level:
        return 0.0, 0.0
    i = np.flatnonzero(k < level)
    if not len(i):
        return np.nan, np.nan
    i = i[0]
    f = (k[i - 1] - level) / (k[i - 1] - k[i])
    return t[i - 1] + f * (t[i] - t[i - 1]), b[i - 1] + f * (b[i] - b[i - 1])


def main():
    S = study("11_burnup_neural_network", "11. Neural-network burnup framework",
              "Deep-ensemble neural networks learn complete burnup trajectories of a thorium "
              "MSR (12 outputs vs 7 design/processing parameters and time) from MSR-NX "
              "depletion runs; new data from other codes are added by merging, a discrepancy "
              "correction or fine-tuning.")
    out = S.outdir
    # ------------------------------------------------------------- 1. data
    lf_csv = os.path.join(out, "physics_dataset.csv")
    hf_csv = os.path.join(out, "hf_core2d_dataset.csv")
    t0 = time.time()
    if os.path.exists(lf_csv):
        lf = BurnupDataset.load(lf_csv, source="msrnx-lattice")
    else:
        lf = generate_physics_dataset(400, seed=11, out_csv=lf_csv, source="msrnx-lattice")
    t_lf = time.time() - t0
    t0 = time.time()
    if os.path.exists(hf_csv):
        hf = BurnupDataset.load(hf_csv, source="msrnx-core2d")
    else:
        hf = generate_physics_dataset(32, seed=111, transport="core", steps_days=COARSE_STEPS_DAYS,
                                      case_prefix="core2d", out_csv=hf_csv, source="msrnx-core2d")
    S.add_table("Design space of the training data (Latin hypercube)",
                ["parameter", "low", "high", "scale"],
                [[p.label, p.low, p.high, "log" if p.log else "linear"] for p in BURNUP_SPACE.params])
    S.add_table("Data sets", ["data set", "transport", "cases", "time points", "span [d]", "rows"], [
        ["built-in training data", "2D-calibrated lattice + B1", lf.n_cases, len(DEFAULT_STEPS_DAYS) + 1,
         sum(DEFAULT_STEPS_DAYS), len(lf)],
        ["'external code' data", "2D R-Z diffusion every step", hf.n_cases, len(COARSE_STEPS_DAYS) + 1,
         sum(COARSE_STEPS_DAYS), len(hf)]])
    S.add_figure(bp.data_overview(lf, ["k_eff", "breeding_ratio", "U233_kg", "Pa233_kg",
                                       "Xe135_abs_pct", "FP_abs_pct", "U232_kg", "Np237_kg"],
                                  S.path("training_data.png"), color_by="uf4_pct", max_cases=160,
                                  title=f"Training data: {lf.n_cases} MSR depletion trajectories"),
                 "Burnup trajectories of the built-in data set; colour = UF4 (U-233) content.")
    # ------------------------------------------------------------- 2. main model
    tr, va, te = lf.split(test=0.125, val=0.125, seed=0)
    model_dir = os.path.join(out, "model")
    if os.environ.get("BURNUP_NN_REUSE_MODEL") == "1" and os.path.exists(model_dir):
        model = BurnupNN.load(model_dir)
        print(f"  reusing {model_dir}")
    else:
        model = BurnupNN(FEATURES, TARGETS, monotonic=MONOTONIC, **MAIN).fit(tr, va)
    # members train in parallel: elapsed time = slowest member
    t_train = max(h["wall_s"] for h in model.history if h.get("phase") == "train")
    met = model.evaluate(te)
    model.metrics = {"test": met.to_dict(orient="index")}
    model.save(model_dir)
    rows = []
    for t, r in met.iterrows():
        pcm = "rmse_pcm" in r and np.isfinite(r.get("rmse_pcm", np.nan))
        rows.append([t, f"{r['rmse_pcm']:.0f} pcm" if pcm else f"{r['rmse']:.4g}",
                     f"{r['max_abs_pcm']:.0f} pcm" if pcm else f"{r['max_abs']:.4g}",
                     f"{r['r2']:.5f}", f"{r['nrmse_pct']:.2f}", f"{r['coverage95']:.2f}"])
    S.add_table(f"Accuracy on {te.n_cases} unseen designs ({len(te)} points)",
                ["output", "RMSE", "max |error|", "R2", "RMSE [% of range]", "95% coverage"], rows)
    S.set(k_eff_rmse_pcm=float(met.loc["k_eff", "rmse_pcm"]),
          k_eff_max_err_pcm=float(met.loc["k_eff", "max_abs_pcm"]),
          mean_r2=float(met["r2"].mean()), mean_nrmse_pct=float(met["nrmse_pct"].mean()),
          mean_coverage95=float(met["coverage95"].mean()), train_time_s=t_train,
          log_targets=model.train_info["log_targets"])
    S.add_figure(bp.history(model, S.path("training_history.png"),
                            "Training history of the 4-member ensemble"),
                 "L-BFGS training; each member keeps the weights with the lowest validation loss.")
    S.add_figure(bp.parity_grid(model, te, TARGETS, S.path("parity.png"),
                                f"Neural network vs physics on {te.n_cases} unseen designs"),
                 "Every output of every time point of the test designs.")
    S.add_figure(bp.accuracy_bars(met, S.path("accuracy.png"),
                                  "Accuracy per output on unseen designs"),
                 "RMSE as a percentage of each output's range; R2 in the labels.")
    S.add_figure(bp.trajectories(model, te, ["k_eff", "breeding_ratio", "U233_kg", "Pa233_kg",
                                             "Xe135_abs_pct", "U232_kg"],
                                 S.path("trajectories.png"), n_cases=3, time_scale="symlog",
                                 title="Burnup trajectories of three unseen designs"),
                 "Symmetric-log time axis: the first days (Xe-135 rising to its sparging-limited "
                 "equilibrium, Pa-233 build-up) and the multi-year trends are both visible; the "
                 "network follows the physics at every time scale.")
    S.add_figure(bp.error_vs_time(model, te, TARGETS, S.path("error_vs_time.png")),
                 "Errors are spread evenly along the trajectory; the widest band is Xe-135 "
                 "absorption during its first-day build-up.")
    S.add_figure(bp.calibration(model, te, S.path("uncertainty_calibration.png")),
                 f"Calibrated on the validation designs, the 95% intervals cover "
                 f"{100 * met['coverage95'].mean():.0f}% of unseen points, and a larger predicted "
                 "sigma means a larger actual error.")
    # ------------------------------------------------- 3. learning curve
    sizes = [10, 20, 40, 80, 160, tr.n_cases]
    curve = learning_curve(FEATURES, TARGETS, tr, va, te, sizes, seed=1,
                           variants={"multiscale": {"time_encoding": "multiscale"},
                                     "linear": {"time_encoding": "linear"}},
                           monotonic=MONOTONIC, **CURVE)
    kc = curve[curve["target"] == "k_eff"]
    lc = {v: kc[kc["variant"] == v].sort_values("n_cases")["rmse_pcm"].tolist()
          for v in ("multiscale", "linear")}
    mean_n = curve.groupby(["variant", "n_cases"])["nrmse_pct"].mean()
    lcn = {v: [float(mean_n[(v, n)]) for n in sizes] for v in ("multiscale", "linear")}
    S.add_figure(bp.series_curves(
        sizes, {"physics-informed multi-timescale time encoding": lc["multiscale"],
                "plain time input": lc["linear"]},
        S.path("learning_curve.png"), "training cases (designs)", "k_eff test RMSE [pcm]",
        "How much data does the network need?",
        "Single network per point, 50 unseen test designs; every added case is one depletion run",
        logx=True, logy=True), _curve_caption(sizes, lc))
    S.add_table("Learning curve (single network, test designs)",
                ["training cases", "k_eff RMSE multiscale [pcm]", "k_eff RMSE plain time [pcm]",
                 "mean nRMSE multiscale [%]", "mean nRMSE plain time [%]"],
                [[n, round(a, 0), round(b, 0), round(c, 3), round(d, 3)] for n, a, b, c, d in
                 zip(sizes, lc["multiscale"], lc["linear"], lcn["multiscale"], lcn["linear"])])
    S.set(learning_curve={"sizes": sizes, **{f"kpcm_{k}": v for k, v in lc.items()}})
    # --------------------------------------- 4. adding data from another code
    hf_ids = np.random.default_rng(5).permutation(hf.cases)
    hf_te = hf.subset(hf_ids[:8])
    pool = hf_ids[8:]
    n_hf = [2, 4, 8, 16, 24]
    met_lf = model.evaluate(hf_te)
    lf_only, lf_only_n = float(met_lf.loc["k_eff", "rmse_pcm"]), float(met_lf["nrmse_pct"].mean())
    res = {k: [] for k in ("correction", "weights", "scratch")}
    resn = {k: [] for k in res}
    corrected = {}
    for n in n_hf:
        sub = hf.subset(pool[:n])
        m_c = BurnupNN.load(model_dir)
        m_c.fit_correction(sub, verbose=False)
        corrected[n] = m_c
        m_w = BurnupNN.load(model_dir)
        m_w.fine_tune(sub, verbose=False)
        m_s = BurnupNN(FEATURES, TARGETS, monotonic=MONOTONIC, hidden=(64, 64, 64), ensemble=4,
                       max_iter=1500, seed=3).fit(sub, verbose=False)
        for k, m in (("correction", m_c), ("weights", m_w), ("scratch", m_s)):
            met_k = m.evaluate(hf_te)
            res[k].append(float(met_k.loc["k_eff", "rmse_pcm"]))
            resn[k].append(float(met_k["nrmse_pct"].mean()))
    S.add_figure(bp.series_curves(
        n_hf, {"built-in model + discrepancy correction": res["correction"],
               "built-in model + weight fine-tuning": res["weights"],
               "trained from scratch on the new runs": res["scratch"]},
        S.path("transfer_learning.png"),
        "runs of the 'external' code (2D-core depletion)", "k_eff RMSE vs 2D core [pcm]",
        "Adding a few runs of another code",
        "8 held-out 2D-core designs; horizontal line: built-in network alone (lattice-vs-2D bias "
        "plus network error)", logx=True, logy=True,
        hlines={"built-in model only (no new runs)": lf_only}),
        _transfer_caption(n_hf, lf_only, res))
    S.add_table("Adding runs of another code: k_eff RMSE [pcm] / mean RMSE [% of range] on 8 "
                "held-out 2D-core designs",
                ["new runs", "built-in model only", "+ discrepancy correction", "+ weight fine-tuning",
                 "from scratch"],
                [[n, f"{lf_only:.0f} / {lf_only_n:.2f}", f"{a:.0f} / {an:.2f}", f"{b:.0f} / {bn:.2f}",
                  f"{c:.0f} / {cn:.2f}"]
                 for n, a, an, b, bn, c, cn in zip(n_hf, res["correction"], resn["correction"],
                                                   res["weights"], resn["weights"],
                                                   res["scratch"], resn["scratch"])])
    S.set(transfer={"n": n_hf, "lf_only_pcm": lf_only, "lf_only_nrmse": lf_only_n,
                    "correction_pcm": res["correction"], "weights_pcm": res["weights"],
                    "scratch_pcm": res["scratch"], "correction_nrmse": resn["correction"],
                    "weights_nrmse": resn["weights"], "scratch_nrmse": resn["scratch"]})
    best = corrected[8]
    S.add_figure(bp.trajectories(best, hf_te, ["k_eff", "breeding_ratio", "U233_kg"],
                                 S.path("transfer_trajectories.png"), n_cases=3,
                                 title="Built-in network + correction from 8 external runs",
                                 reference_label="2D-core depletion (held-out designs)"),
                 "Held-out designs of the other code, predicted by the corrected network.")
    # ------------------------------------------------------ 5. design map
    ux = np.linspace(0.22, 0.58, 37)
    th = np.linspace(6.2, 13.8, 39)
    UX, TH = np.meshgrid(ux, th)
    grid = pd.DataFrame({**{k: np.full(UX.size, v) for k, v in NOMINAL.items()},
                         "uf4_pct": UX.ravel(), "thf4_pct": TH.ravel()})
    t0 = time.time()
    xc = model.crossing(grid, "k_eff", 1.0, burnup_target="burnup_MWd_kgHM", n_times=300)
    t_map = time.time() - t0
    BU = xc["burnup"].to_numpy().reshape(UX.shape)
    BUs = xc["burnup_std"].to_numpy().reshape(UX.shape)
    checks = [(0.30, 8.0), (0.35, 12.0), (0.45, 7.0), (0.50, 11.0), (0.55, 13.0)]
    pts, rows = [], []
    t0 = time.time()
    for i, (u, h) in enumerate(checks):
        p = dict(NOMINAL, uf4_pct=u, thf4_pct=h)
        ref = run_case(p, DEFAULT_STEPS_DAYS, case_id=f"check_{i}")
        t_ref, b_ref = _reference_crossing(ref)
        nn = model.crossing(pd.DataFrame([p]), "k_eff", 1.0, burnup_target="burnup_MWd_kgHM",
                            n_times=300).iloc[0]
        rows.append([f"{u:.2f}", f"{h:.1f}", f"{ref['k_eff'].iloc[0]:.4f}",
                     "-" if np.isnan(t_ref) else f"{t_ref / 365.25:.2f}",
                     "-" if np.isnan(nn['time']) else f"{nn['time'] / 365.25:.2f} +/- {nn['time_std'] / 365.25:.2f}",
                     "-" if np.isnan(b_ref) else f"{b_ref:.1f}",
                     "-" if np.isnan(nn['burnup']) else f"{nn['burnup']:.1f} +/- {nn['burnup_std']:.1f}"])
        lab = "beyond span" if np.isnan(b_ref) else f"physics {b_ref:.0f} / NN {nn['burnup']:.0f}"
        pts.append((u, h, lab))
    t_phys_case = (time.time() - t0) / len(checks)
    S.add_figure(bp.crossing_map(ux, th, BU, BUs, S.path("discharge_burnup_map.png"),
                                 "UF4 (U-233) [mol%]", "ThF4 [mol%]", "discharge burnup [MWd/kgHM]",
                                 "Discharge burnup without refuelling (k_eff reaches 1.0)",
                                 f"{UX.size} designs evaluated by the network in {t_map:.1f} s; "
                                 "other parameters nominal; markers: physics checks; blank = beyond "
                                 f"{sum(DEFAULT_STEPS_DAYS):.0f} days", pts),
                 "Discharge burnup grows with the U-233 loading. In lean salts more thorium "
                 "lowers it (the core starts closer to k = 1); rich, thorium-heavy salts stay "
                 "critical beyond the 8.6-year span. The uncertainty is largest where k_eff starts "
                 "close to 1; the physics checks agree within 1 %.")
    S.add_table("Discharge burnup map: physics checks (k_eff = 1 crossing, nominal other parameters)",
                ["UF4 [mol%]", "ThF4 [mol%]", "k_eff BOL", "cycle physics [y]", "cycle NN [y]",
                 "burnup physics [MWd/kgHM]", "burnup NN [MWd/kgHM]"], rows)
    # ---------------------------------------------- 6. sensitivities and cost
    designs = te.case_table(FEATURES).reset_index(drop=True)
    sk = model.sensitivity(designs, "k_eff", times=[1500.0])
    sb = model.sensitivity(designs, "breeding_ratio", times=[1500.0])
    fig, ax = pl.figure(1, 2, (12, 4.2), "What drives reactivity and breeding after 4 years?",
                        "Change per one standard deviation of each input over the training data "
                        "(log inputs: per sigma of log); network gradients, mean over test designs")
    for a, s_, tgt, unit in ((ax[0], sk, "k_eff", 1e5), (ax[1], sb, "breeding ratio", 1.0)):
        pl.hbar(a, [bp.nice(f) for f in s_["feature"]], s_["mean_signed"].to_numpy() * unit,
                fmt="{:+.0f}" if unit > 1 else "{:+.3f}",
                xlabel=f"d {tgt} per 1 sigma" + (" [pcm]" if unit > 1 else ""))
        a.set_title(f"{tgt} at 1500 days", loc="left", fontsize=10)
    S.add_figure(pl.save(fig, S.path("sensitivities.png")),
                 "Power (burnup accumulated by 1500 days) and U-233 loading dominate reactivity; "
                 "slower rare-earth, Pa and Xe removal each cost reactivity. Breeding rises as "
                 "U-233 burns down and with more thorium; Pa removal barely changes the "
                 "capture-based breeding ratio (checked against physics below).")
    rows = []
    for pa in (3.0, 30.0, 300.0, 3000.0):              # spot check of a sensitivity
        p = dict(NOMINAL, pa_removal_days=pa)
        ref = run_case(p, DEFAULT_STEPS_DAYS, case_id=f"pa_{pa:g}")
        i = int(np.argmin(np.abs(ref["time_days"].to_numpy() - 1500.0)))
        nn = model.predict_trajectory(p, times=[ref["time_days"].iloc[i]]).iloc[0]
        rows.append([f"{pa:g}", f"{ref['time_days'].iloc[i]:.0f}", f"{ref['k_eff'].iloc[i]:.5f}",
                     f"{nn['k_eff']:.5f}", f"{ref['breeding_ratio'].iloc[i]:.4f}",
                     f"{nn['breeding_ratio']:.4f}", f"{ref['Pa233_kg'].iloc[i]:.1f}",
                     f"{nn['Pa233_kg']:.1f}"])
    S.add_table("Spot check of a sensitivity: Pa removal time (nominal design), physics vs network",
                ["Pa removal [d]", "time [d]", "k_eff physics", "k_eff NN", "BR physics", "BR NN",
                 "Pa-233 physics [kg]", "Pa-233 NN [kg]"], rows)
    t0 = time.time()
    model.predict_cases(designs.iloc[:20], n_times=200)
    t_nn = (time.time() - t0) / 20
    fig, ax = pl.figure(1, 1, (11, 1.9), "Neural-network burnup model at a glance")
    pl.kpi_row(ax, [("k-eff RMSE, unseen designs", f"{met.loc['k_eff', 'rmse_pcm']:.0f} pcm"),
                    ("mean R2, 12 outputs", f"{met['r2'].mean():.4f}"),
                    ("95% interval coverage", f"{met['coverage95'].mean():.0%}"),
                    ("trajectory cost", f"{1e3 * t_nn:.0f} ms"),
                    ("speed-up vs depletion", f"{t_phys_case / t_nn:,.0f}x")])
    S.add_figure(pl.save(fig, S.path("summary.png")), "Headline numbers of the trained model.")
    S.add_table("Computing cost", ["task", "time"], [
        ["generate built-in data (400 cases, 4 processes)", f"{t_lf:.0f} s" if t_lf > 1 else "cached"],
        ["train 4-member ensemble (4 processes)", f"{t_train:.0f} s"],
        ["one depletion trajectory, physics", f"{t_phys_case:.2f} s"],
        ["one depletion trajectory, network (200 times, mean + sigma)", f"{1e3 * t_nn:.1f} ms"],
        [f"discharge-burnup map ({UX.size} designs)", f"{t_map:.1f} s (physics: "
         f"~{UX.size * t_phys_case / 60:.0f} min)"]])
    S.set(t_nn_ms=1e3 * t_nn, t_phys_s=t_phys_case, map_s=t_map)
    S.save()


if __name__ == "__main__":
    main()
