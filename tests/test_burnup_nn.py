"""Verification of the neural-network burnup framework (msrnx.burnup_nn)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from msrnx.burnup_nn import BurnupDataset, BurnupNN, MLP, gradient_check, write_template
from msrnx.burnup_nn.dataset import DataError
from msrnx.burnup_nn.model import _make_objective

FAST = dict(hidden=(24, 24), ensemble=2, max_iter=500, n_jobs=1, patience=30)


def bateman_dataset(n_cases=40, seed=0, times=None, source="analytic"):
    """Two-nuclide chain N1 -(a)-> N2 -(b)-> : exact solution as a burnup data set."""
    rng = np.random.default_rng(seed)
    times = np.array([0, 1, 3, 10, 30, 60, 100, 150, 200, 300]) if times is None else times
    rows = []
    for i in range(n_cases):
        a, b, n0 = rng.uniform(0.005, 0.05), rng.uniform(0.01, 0.1), rng.uniform(0.5, 2.0)
        n1 = n0 * np.exp(-a * times)
        n2 = n0 * a / (b - a) * (np.exp(-a * times) - np.exp(-b * times))
        for t, x1, x2 in zip(times, n1, n2):
            rows.append({"case_id": f"c{i:03d}", "time_days": t, "a": a, "b": b, "n0": n0,
                         "N1": x1, "N2": x2, "cum": n0 - x1})
    return BurnupDataset(pd.DataFrame(rows), source=source)


# --------------------------------------------------------------------- engine
@pytest.mark.parametrize("act", ["tanh", "silu", "gelu", "softplus", "relu"])
@pytest.mark.parametrize("residual", [False, True])
def test_mlp_gradients(act, residual):
    rng = np.random.default_rng(1)
    net = MLP(4, 3, (7, 7), act, residual, seed=2)
    assert gradient_check(net, rng.normal(size=(25, 4)), rng.normal(size=(25, 3))) < 1e-4


def test_full_objective_gradient_with_mask_weights_constraint_and_anchor():
    """Finite-difference check of the complete training objective."""
    rng = np.random.default_rng(3)
    n, d, T = 30, 5, 3
    X, Y = rng.normal(size=(n, d)), rng.normal(size=(n, T))
    M = (rng.random((n, T)) > 0.3).astype(float)
    W = rng.uniform(0.5, 2.0, n)
    tw = np.array([1.0, 2.0, 0.5])
    X0 = rng.normal(size=(12, d))
    X1 = X0 + 0.05 * rng.normal(size=(12, d))
    colloc = (X0, X1, np.full(12, 0.01), np.array([0, 2]), np.array([1.0, -1.0]))
    net = MLP(d, T, (6, 6), "tanh", seed=4)
    anchor = [p + 0.1 * rng.normal(size=p.shape) for p in net.params]
    obj = _make_objective(X, np.where(M > 0, Y, 0.0), M, W, tw, 1e-3, colloc, 0.5, anchor, 1e-2, 0)
    loss, grads = obj(net, None)
    g = np.concatenate([q.ravel() for q in grads])
    x0 = net.get_flat()
    worst = 0.0
    for i in rng.choice(len(x0), 40, replace=False):
        xp, xm = x0.copy(), x0.copy()
        xp[i] += 1e-6
        xm[i] -= 1e-6
        net.set_flat(xp)
        lp = obj(net, None)[0]
        net.set_flat(xm)
        lm = obj(net, None)[0]
        num = (lp - lm) / 2e-6
        worst = max(worst, abs(num - g[i]) / max(abs(num) + abs(g[i]), 1e-8))
    net.set_flat(x0)
    assert loss > 0 and worst < 1e-4


# ---------------------------------------------------------------------- data
def test_dataset_io_rename_scale_constants_and_split(tmp_path):
    ds = bateman_dataset(12)
    df = ds.df.rename(columns={"case_id": "run", "time_days": "days", "N1": "N1_g"})
    df["N1_g"] *= 1000.0
    csv, xlsx = tmp_path / "user.csv", tmp_path / "user.xlsx"
    df.drop(columns="source").to_csv(csv, index=False)
    df.drop(columns="source").to_excel(xlsx, index=False)
    for path in (csv, xlsx):
        d = BurnupDataset.load(str(path), rename={"run": "case_id", "days": "time_days",
                                                  "N1_g": "N1"}, scale={"N1": 1e-3},
                               constants={"T_fuel": 900.0})
        assert d.n_cases == 12 and np.allclose(d.df["N1"], ds.df["N1"])
        assert (d.df["T_fuel"] == 900.0).all() and d.sources == ["user"]
    tr, va, te = ds.split(test=0.25, val=0.25, seed=1)
    sets = [set(x.cases) for x in (tr, va, te)]
    assert sum(map(len, sets)) == 12 and not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_merge_heterogeneous_sources_and_checks():
    a = bateman_dataset(6, seed=0, source="code_A")
    b = bateman_dataset(4, seed=1, source="code_B")
    b.df = b.df.drop(columns=["N2"])            # code B does not report N2
    m = a + b
    assert m.n_cases == 10                      # colliding ids were prefixed
    assert any(c.startswith("code_B:") for c in m.cases)
    msgs = m.check(["a", "b", "n0"], ["N1", "N2"])
    assert any("blank" in t for _, t in msgs)
    with pytest.raises(DataError):
        m.check(["a", "missing_feature"], ["N1"])
    assert m.detect_monotonic(["N1", "cum"]) == {"N1": "decreasing", "cum": "increasing"}


def test_template_has_expected_columns(tmp_path):
    p = write_template(str(tmp_path / "t.csv"), ["x1", "x2"], ["k_eff", "U233_kg"],
                       example_values={"x1": 1.5})
    df = pd.read_csv(p)
    assert list(df.columns) == ["case_id", "time_days", "x1", "x2", "k_eff", "U233_kg"]
    assert (df["x1"] == 1.5).all() and df["k_eff"].isna().all()


# --------------------------------------------------------------------- model
@pytest.fixture(scope="module")
def bateman_model():
    ds = bateman_dataset(48, seed=0)
    tr, va, te = ds.split(test=0.2, val=0.15, seed=0)
    model = BurnupNN(["a", "b", "n0"], ["N1", "N2", "cum"],
                     monotonic={"cum": "increasing"}, **FAST).fit(tr, va, verbose=False)
    return model, te


def test_learns_bateman_chain(bateman_model):
    model, te = bateman_model
    met = model.evaluate(te)               # 38 training cases in a 3-D parameter space:
    assert (met["r2"] > 0.98).all(), met   # the error is data-limited (~1-2 % of range)
    assert (met["nrmse_pct"] < 3.5).all(), met


def test_trajectory_is_nonnegative_and_monotone(bateman_model):
    model, te = bateman_model
    p = model.predict_cases(te.case_table(model.features).reset_index(), n_times=150)
    assert (p[["N1", "N2", "cum"]].to_numpy() >= 0).all()
    worst = p.groupby("case_id")["cum"].apply(lambda v: np.diff(v.to_numpy()).min()).min()
    assert worst > -1e-3 * p["cum"].max()


def test_save_load_roundtrip(bateman_model, tmp_path):
    model, te = bateman_model
    path = model.save(str(tmp_path / "m"))
    meta = json.load(open(os.path.join(path, "model.json")))
    assert meta["framework"] == "msrnx.burnup_nn" and meta["n_members"] == 2
    m2 = BurnupNN.load(path)
    pd.testing.assert_frame_equal(model.predict(te), m2.predict(te))


def test_masked_targets_train_and_evaluate():
    a = bateman_dataset(24, seed=2, source="A")
    b = bateman_dataset(16, seed=3, source="B")
    b.df["N2"] = np.nan                          # B lacks N2 entirely
    tr, va, te = (a + b).split(test=0.2, val=0.15, seed=0)
    m = BurnupNN(["a", "b", "n0"], ["N1", "N2"], **FAST).fit(tr, va, verbose=False)
    met = m.evaluate(te)
    assert met.loc["N2", "n"] < met.loc["N1", "n"]       # blanks are skipped
    assert (met["r2"] > 0.98).all(), met


def test_fine_tune_adapts_to_shifted_physics():
    base = bateman_dataset(40, seed=4)
    tr, va, _ = base.split(test=0.0, val=0.2, seed=0)
    model = BurnupNN(["a", "b", "n0"], ["N1", "N2"], **FAST).fit(tr, va, verbose=False)
    shifted = bateman_dataset(16, seed=5)
    shifted.df["N2"] *= 1.15                    # "high-fidelity" code: +15 % on N2
    ft_tr, _, ft_te = shifted.split(test=0.5, val=0.0, seed=0)
    before = model.evaluate(ft_te).loc["N2", "rmse"]
    model.fine_tune(ft_tr, max_iter=300, anchor=1e-4, verbose=False)
    after = model.evaluate(ft_te).loc["N2", "rmse"]
    assert after < 0.65 * before, (before, after)   # floor: the base model's own error


def test_discrepancy_correction_helps_and_never_hurts(tmp_path):
    base = bateman_dataset(40, seed=4)
    tr, va, _ = base.split(test=0.0, val=0.2, seed=0)
    model = BurnupNN(["a", "b", "n0"], ["N1", "N2"], **FAST).fit(tr, va, verbose=False)
    # (1) a "code" whose N2 is 15 % higher: a few runs remove most of the bias
    shifted = bateman_dataset(14, seed=5)
    shifted.df["N2"] *= 1.15
    c_tr, _, c_te = shifted.split(test=0.5, val=0.0, seed=0)
    before = model.evaluate(c_te).loc["N2", "rmse"]
    summary = model.fit_correction(c_tr, verbose=False)
    assert summary.loc["N2", "corrected"]
    after = model.evaluate(c_te).loc["N2", "rmse"]
    assert after < 0.5 * before, (before, after)
    # saved and restored with the model
    m2 = BurnupNN.load(model.save(str(tmp_path / "m")))
    pd.testing.assert_frame_equal(model.predict(c_te), m2.predict(c_te))
    # (2) runs of the same physics: the CV-chosen correction does not degrade the model
    model.clear_correction()
    same = bateman_dataset(14, seed=6)
    s_tr, _, s_te = same.split(test=0.5, val=0.0, seed=0)
    ref = model.evaluate(s_te)["rmse"]
    model.fit_correction(s_tr, verbose=False)
    assert (model.evaluate(s_te)["rmse"] < 1.25 * ref + 1e-3).all()


def test_crossing_matches_analytic():
    rng = np.random.default_rng(6)
    rows = []
    for i in range(40):                          # k(t) = k0 - s t  ->  k = 1 at (k0-1)/s
        k0, s = rng.uniform(1.05, 1.3), rng.uniform(1e-4, 4e-4)
        for t in np.linspace(0, 3000, 16):
            rows.append({"case_id": f"c{i}", "time_days": t, "k0": k0, "s": s,
                         "k_eff": k0 - s * t, "burnup": 0.02 * t})
    ds = BurnupDataset(pd.DataFrame(rows))
    tr, va, _ = ds.split(test=0.0, val=0.2, seed=0)
    m = BurnupNN(["k0", "s"], ["k_eff", "burnup"], **FAST).fit(tr, va, verbose=False)
    designs = pd.DataFrame({"k0": [1.1, 1.2, 1.25], "s": [2e-4, 2e-4, 1.5e-4]})
    xc = m.crossing(designs, "k_eff", 1.0, burnup_target="burnup")
    exact = (designs["k0"] - 1.0) / designs["s"]
    assert np.allclose(xc["time"], exact, rtol=0.05), (xc["time"].tolist(), exact.tolist())
    assert np.allclose(xc["burnup"], 0.02 * exact, rtol=0.06)
    assert (xc["censored"] == 0).all()
    sens = m.sensitivity(designs, "k_eff")
    assert set(sens["feature"]) == {"k0", "s"}


def test_time_encoding_beats_raw_time():
    """Physics-informed multi-timescale encoding vs a plain time input on a
    stiff chain (fast and slow components)."""
    ds = bateman_dataset(40, seed=7, times=np.array([0, .5, 1, 2, 5, 10, 30, 100, 300, 1000.]))
    tr, va, te = ds.split(test=0.25, val=0.15, seed=0)
    err = {}
    for enc in ("multiscale", "linear"):
        m = BurnupNN(["a", "b", "n0"], ["N2"], time_encoding=enc, **FAST).fit(tr, va, verbose=False)
        err[enc] = m.evaluate(te).loc["N2", "rmse"]
    assert err["multiscale"] < err["linear"], err


def test_learning_curve_more_data_helps():
    from msrnx.burnup_nn import learning_curve
    ds = bateman_dataset(50, seed=9)
    tr, va, te = ds.split(test=0.2, val=0.1, seed=0)
    lc = learning_curve(["a", "b", "n0"], ["N1", "N2"], tr, va, te, [4, 30], n_jobs=1,
                        hidden=(16, 16), max_iter=300)
    assert set(lc["n_cases"]) == {4, 30} and set(lc["target"]) == {"N1", "N2"}
    err = lc.groupby("n_cases")["nrmse_pct"].mean()
    assert err[30] < err[4]


# ----------------------------------------------------------------------- CLI
def test_cli_train_predict(tmp_path):
    from msrnx.burnup_nn.__main__ import main
    ds = bateman_dataset(30, seed=8)
    ds.to_csv(str(tmp_path / "data.csv"))
    cfg = {"title": "CLI test", "data": [{"path": "data.csv"}], "features": ["a", "b", "n0"],
           "targets": ["N1", "N2"], "model": {"hidden": [12], "ensemble": 2, "max_iter": 80,
                                              "n_jobs": 1},
           "split": {"test": 0.2, "val": 0.2, "seed": 0}, "output_dir": "run"}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    main(["check", str(tmp_path / "cfg.json")])
    main(["train", str(tmp_path / "cfg.json")])
    run = tmp_path / "run"
    for f in ("model/model.json", "model/weights.npz", "REPORT.md", "report.html",
              "test_metrics.csv", "figures/parity.png", "figures/trajectories.png"):
        assert (run / f).exists(), f
    pd.DataFrame({"a": [0.02], "b": [0.05], "n0": [1.0]}).to_csv(tmp_path / "new.csv", index=False)
    main(["predict", "--model", str(run / "model"), "--input", str(tmp_path / "new.csv"),
          "--out", str(tmp_path / "pred")])
    p = pd.read_csv(tmp_path / "pred" / "predictions.csv")
    assert {"N1", "N2", "N1_std", "extrapolation"} <= set(p.columns) and len(p) > 50


def test_physics_generator_two_cases():
    from msrnx.burnup_nn import BURNUP_SPACE, generate_physics_dataset
    ds = generate_physics_dataset(2, steps_days=(5, 25, 60), seed=1, n_jobs=1, verbose=False)
    assert ds.n_cases == 2 and len(ds) == 8
    for c in BURNUP_SPACE.names + ["k_eff", "burnup_MWd_kgHM", "U233_kg", "Pa233_kg",
                                   "Xe135_abs_pct", "breeding_ratio"]:
        assert c in ds.df.columns
    for _, d in ds.df.groupby("case_id"):
        assert np.all(np.diff(d["burnup_MWd_kgHM"]) > 0)
        assert d["Pa233_kg"].iloc[0] == 0.0 and d["Pa233_kg"].iloc[-1] > 0.0
