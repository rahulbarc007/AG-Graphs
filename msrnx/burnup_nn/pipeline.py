"""Config-driven workflow: your data -> trained burnup network -> charts + report.

A run is described by a JSON file (see ``examples/burnup_nn/config.json``)::

    {
      "title": "My MSR burnup surrogate",
      "data": [
        {"path": "physics_dataset.csv", "source": "msrnx"},
        {"path": "serpent_runs.xlsx", "source": "serpent", "weight": 3.0,
         "rename": {"BU": "burnup_MWd_kgHM", "keff": "k_eff"},
         "scale": {"U233_g": 0.001}, "constants": {"T_fuel": 900.0}}
      ],
      "case_column": "case_id", "time_column": "time_days",
      "features": ["uf4_pct", "thf4_pct", "salt_fraction", "power_MW"],
      "targets":  ["k_eff", "U233_kg", "Pa233_kg", "burnup_MWd_kgHM"],
      "monotonic": "auto",
      "model": {"hidden": [64, 64, 64], "ensemble": 5, "max_iter": 3000},
      "split": {"test": 0.15, "val": 0.15, "seed": 0},
      "crossing": {"target": "k_eff", "level": 1.0, "burnup_target": "burnup_MWd_kgHM"},
      "output_dir": "runs/my_surrogate"
    }

Relative paths are resolved against the folder of the config file.
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Optional, Union

import numpy as np
import pandas as pd

from .dataset import WEIGHT, BurnupDataset, DataError
from .model import BurnupNN, NNConfig, is_pcm_target

DEFAULTS = {
    "title": "Neural-network burnup model",
    "data": [],
    "case_column": "case_id",
    "time_column": "time_days",
    "features": [],
    "targets": [],
    "log_targets": "auto",
    "log_features": "auto",
    "monotonic": "auto",
    "target_weights": {},
    "model": {},
    "split": {"test": 0.15, "val": 0.15, "seed": 0},
    "plots": {"color_by": None, "trajectory_targets": None, "n_trajectory_cases": 3,
              "max_panels": 12},
    "crossing": None,
    "output_dir": "burnup_nn_run",
}


# ---------------------------------------------------------------------------
def load_config(cfg: Union[str, dict]) -> dict:
    """Read a JSON config (or take a dict) and fill in defaults."""
    base = os.getcwd()
    if isinstance(cfg, str):
        base = os.path.dirname(os.path.abspath(cfg))
        with open(cfg) as f:
            cfg = json.load(f)
    out = copy.deepcopy(DEFAULTS)
    for k, v in cfg.items():
        if k.startswith("_"):              # comments
            continue
        if k not in DEFAULTS:
            raise ValueError(f"unknown config key '{k}'; valid keys: {sorted(DEFAULTS)}")
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    out["_base"] = cfg.get("_base", base)
    NNConfig.from_dict(out["model"])       # validate early
    return out


def _path(cfg: dict, p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(cfg["_base"], p))


def load_datasets(cfg: dict, verbose: bool = True) -> BurnupDataset:
    """Load and merge every entry of ``cfg["data"]``."""
    if not cfg["data"]:
        raise DataError("config has no 'data' entries")
    sets = []
    for entry in cfg["data"]:
        e = {"path": entry} if isinstance(entry, str) else dict(entry)
        path = _path(cfg, e.pop("path"))
        if not os.path.exists(path):
            hint = ""
            if "physics_dataset" in os.path.basename(path):
                hint = ("\n  generate it with:  python -m msrnx.burnup_nn generate --cases 400 "
                        f"--out {os.path.relpath(path)}")
            raise DataError(f"data file not found: {path}{hint}")
        weight = float(e.pop("weight", 1.0))
        ds = BurnupDataset.load(path, cfg["case_column"], cfg["time_column"], **e)
        if weight != 1.0:
            ds.df[WEIGHT] = ds.df.get(WEIGHT, 1.0) * weight
        if verbose:
            print(f"  loaded {path}: {ds.n_cases} cases, {len(ds)} rows (source '{ds.sources[0]}'"
                  + (f", weight {weight:g}" if weight != 1.0 else "") + ")")
        sets.append(ds)
    return BurnupDataset.concat(sets) if len(sets) > 1 else sets[0]


def check_data(cfg: dict, verbose: bool = True):
    """Load the data of a config and report problems + monotonic candidates."""
    cfg = load_config(cfg) if not isinstance(cfg, dict) or "_base" not in cfg else cfg
    ds = load_datasets(cfg, verbose)
    msgs = ds.check(cfg["features"], cfg["targets"], raise_on_error=False)
    mono = ds.detect_monotonic(cfg["targets"]) if cfg["time_column"] else {}
    if verbose:
        print(f"\n{ds}")
        for lvl, m in msgs:
            print(f"  [{lvl}] {m}")
        if not msgs:
            print("  [ok] no problems found")
        if mono:
            print(f"  monotone in every case (candidates for 'monotonic'): {mono}")
        cols = [c for c in cfg["features"] + cfg["targets"] if c in ds.df.columns]
        print("\n" + ds.summary(cols).to_string(float_format=lambda v: f"{v:.5g}"))
    return ds, msgs, mono


def _monotonic(cfg: dict, data: BurnupDataset) -> dict:
    spec = cfg["monotonic"]
    if not spec or not cfg["time_column"]:
        return {}
    if spec == "auto":
        return data.detect_monotonic(cfg["targets"])
    return dict(spec)


def build_model(cfg: dict, data: Optional[BurnupDataset] = None) -> BurnupNN:
    return BurnupNN(cfg["features"], cfg["targets"], cfg["time_column"], cfg["case_column"],
                    cfg["log_targets"], cfg["log_features"],
                    _monotonic(cfg, data) if data is not None else {},
                    cfg["target_weights"], NNConfig.from_dict(cfg["model"]))


def _metrics_rows(met: pd.DataFrame):
    rows = []
    for t, r in met.iterrows():
        pcm = is_pcm_target(t)
        rows.append([t, f"{r['rmse_pcm']:.0f} pcm" if pcm else f"{r['rmse']:.4g}",
                     f"{r['max_abs_pcm']:.0f} pcm" if pcm else f"{r['max_abs']:.4g}",
                     f"{r['r2']:.5f}", f"{r['nrmse_pct']:.2f}", f"{r['rel_rmse_pct']:.2f}",
                     f"{r['coverage95']:.2f}" if np.isfinite(r["coverage95"]) else "-"])
    return rows


METRIC_HEADERS = ["target", "RMSE", "max |error|", "R2", "RMSE [% range]", "RMSE [% mean]",
                  "95% coverage"]


def _charts(study, model: BurnupNN, cfg: dict, train, test, title_suffix: str = ""):
    """Standard chart set for a trained model."""
    from . import plots
    pc = cfg["plots"]

    def present(ds, names):
        return [t for t in names if t in ds.df.columns and ds.df[t].notna().any()]

    targets = present(train, model.targets)[:pc.get("max_panels", 12)]
    color_by = pc.get("color_by") or (model.features[0] if model.features else None)
    if model.time_column and targets:
        study.add_figure(plots.data_overview(train, targets, study.path("data_overview.png"),
                                             color_by=color_by),
                         "Training trajectories of every output, coloured by "
                         f"{plots.nice(color_by) if color_by else 'case'}.")
    study.add_figure(plots.history(model, study.path("training_history.png"),
                                   f"Training history{title_suffix}"),
                     "Loss of every ensemble member; training stops at the best validation loss.")
    if test is None:
        return
    targets = present(test, model.targets)[:pc.get("max_panels", 12)]
    tt = [t for t in (pc.get("trajectory_targets") or targets[:6]) if t in targets]
    if not targets:
        return
    study.add_figure(plots.parity_grid(model, test, targets, study.path("parity.png")),
                     "Neural network vs reference on test cases never used in training.")
    met = model.evaluate(test)
    study.add_figure(plots.accuracy_bars(met.loc[[t for t in targets if t in met.index]],
                                         study.path("accuracy.png")),
                     "Normalized RMSE per output on the test cases.")
    if model.time_column:
        study.add_figure(plots.trajectories(model, test, tt, study.path("trajectories.png"),
                                            n_cases=pc.get("n_trajectory_cases", 3),
                                            reference_label="reference"),
                         "Test trajectories with the ensemble 95% interval.")
        study.add_figure(plots.error_vs_time(model, test, targets, study.path("error_vs_time.png")),
                         "Error distribution along the burnup trajectory.")
    if len(model.nets) > 1:
        study.add_figure(plots.calibration(model, test, study.path("calibration.png")),
                         "Calibration of the ensemble uncertainty on the test cases.")
    sens_target = (cfg.get("crossing") or {}).get("target") or targets[0]
    sens_target = sens_target if sens_target in model.targets else targets[0]
    if model.features:
        sens = model.sensitivity(test, sens_target)
        study.add_figure(plots.sensitivity_chart(sens, sens_target,
                                                 study.path("sensitivity.png")),
                         f"Input sensitivities of {sens_target} (analytic network gradients).")


def _report(study, out_dir: str, title: str):
    from ..report import build_html_report, build_markdown_report
    study.save()
    md = build_markdown_report(out_dir, os.path.join(out_dir, "REPORT.md"), title,
                               study_dirs=[study.outdir], generator="python -m msrnx.burnup_nn")
    html = build_html_report(out_dir, os.path.join(out_dir, "report.html"), title,
                             study_dirs=[study.outdir])
    return md, html


def train(cfg: Union[str, dict], out_dir: Optional[str] = None, verbose: bool = True):
    """Full workflow: load + check data, split by case, train, evaluate,
    save model, charts and report.  Returns (model, test metrics DataFrame)."""
    from ..report import Study
    cfg = load_config(cfg)
    out_dir = _path(cfg, out_dir or cfg["output_dir"])
    t0 = time.time()
    print(f"== {cfg['title']}\n-- loading data")
    data = load_datasets(cfg, verbose)
    for lvl, m in data.check(cfg["features"], cfg["targets"]):
        print(f"  [{lvl}] {m}")
    sp = cfg["split"]
    tr, va, te = data.split(sp.get("test", 0.15), sp.get("val", 0.15), sp.get("seed", 0))
    print(f"-- split by case: {tr.n_cases} train / {va.n_cases if va else 0} validation / "
          f"{te.n_cases if te else 0} test cases")
    model = build_model(cfg, tr)
    if model.monotonic:
        print(f"-- monotonic constraints: {model.monotonic}")
    model.fit(tr, va, verbose=verbose)
    study = Study("model_report", cfg["title"], os.path.join(out_dir, "figures"),
                  f"Neural-network burnup model trained on {tr.n_cases} cases from "
                  f"{', '.join(data.sources)}; tested on {te.n_cases if te else 0} unseen cases.")
    study.add_table("Data", ["item", "value"], [
        ["sources", ", ".join(data.sources)], ["cases (train/val/test)",
         f"{tr.n_cases}/{va.n_cases if va else 0}/{te.n_cases if te else 0}"],
        ["rows", len(data)], ["features", ", ".join(cfg["features"])],
        ["targets", ", ".join(cfg["targets"])], ["time column", str(cfg["time_column"])],
        ["log-space targets", ", ".join(model.train_info["log_targets"]) or "-"],
        ["monotonic constraints", ", ".join(f"{k} ({v})" for k, v in model.monotonic.items()) or "-"]])
    c = model.cfg
    study.add_table("Network", ["item", "value"], [
        ["architecture", f"{model.nets[0].sizes} {c.activation}"],
        ["ensemble members", len(model.nets)], ["parameters per member", model.nets[0].n_params],
        ["optimizer", model.train_info["optimizer"]], ["time encoding", c.time_encoding],
        ["training time (CPU s, all members)", f"{model.train_info['wall_s']:.0f}"]])
    met = None
    if te is not None:
        met = model.evaluate(te)
        model.metrics = {"test": met.to_dict(orient="index")}
        study.add_table("Accuracy on unseen test cases", METRIC_HEADERS, _metrics_rows(met))
        study.set(test_metrics=model.metrics["test"])
        _, pred = model.evaluate(te, return_predictions=True)
        os.makedirs(out_dir, exist_ok=True)
        pred.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False, float_format="%.7g")
        met.to_csv(os.path.join(out_dir, "test_metrics.csv"), float_format="%.6g")
    model_dir = model.save(os.path.join(out_dir, "model"))
    _charts(study, model, cfg, tr, te)
    if cfg.get("crossing") and model.time_column and te is not None:
        _crossing_table(study, model, cfg["crossing"], te)
    md, html = _report(study, out_dir, cfg["title"])
    print(f"\n-- finished in {time.time() - t0:.0f} s\n   model : {model_dir}\n   report: {md}\n"
          f"           {html}")
    return model, met


def _crossing_table(study, model: BurnupNN, xc: dict, data: BurnupDataset, max_rows: int = 12):
    """Compare predicted and reference crossing times (e.g. k_eff = 1) per case."""
    tgt, lvl = xc.get("target", "k_eff"), float(xc.get("level", 1.0))
    bt = xc.get("burnup_target")
    designs = data.case_table(model.features)
    pred = model.crossing(designs.reset_index(drop=True), tgt, lvl,
                          burnup_target=bt if bt in model.targets else None)
    rows = []
    for (cid, _), (_, p) in zip(designs.iterrows(), pred.iterrows()):
        d = data.case(cid)
        y = d[tgt].to_numpy(float)
        t = d[model.time_column].to_numpy(float)
        ref = np.nan
        if y[0] < lvl:
            ref = 0.0
        else:
            k = np.flatnonzero(y < lvl)
            if len(k):
                i = k[0]
                ref = t[i - 1] + (y[i - 1] - lvl) / (y[i - 1] - y[i]) * (t[i] - t[i - 1])
        if np.isnan(p["time"]):
            nn = "-"
        elif np.isfinite(p["time_std"]):
            nn = f"{p['time']:.0f} +/- {p['time_std']:.0f}"
        else:
            nn = f"{p['time']:.0f}"
        rows.append([cid, "-" if np.isnan(ref) else f"{ref:.0f}", nn])
    study.add_table(f"Time to reach {tgt} = {lvl:g} on test cases ({model.time_column})",
                    ["case", "reference", "neural network"], rows[:max_rows])


def fine_tune(cfg: Union[str, dict], model_dir: str, out_dir: Optional[str] = None,
              method: str = "auto", anchor: float = 1e-3, freeze: int = 0, max_iter: int = 300,
              verbose: bool = True):
    """Adapt a trained model to the data of ``cfg`` (e.g. runs of your own code).

    method  ``"correction"`` additive discrepancy correction (robust with a few
            runs), ``"weights"`` weight fine-tuning (needs more runs), ``"auto"``
            = correction below 12 training cases, otherwise weights.
    The new data are split by case; the test cases measure the error before
    and after."""
    from ..report import Study
    cfg = load_config(cfg)
    out_dir = _path(cfg, out_dir or cfg["output_dir"])
    model = BurnupNN.load(model_dir)
    data = load_datasets(cfg, verbose)
    sp = cfg["split"]
    tr, va, te = data.split(sp.get("test", 0.2), sp.get("val", 0.0), sp.get("seed", 0))
    if method == "auto":
        method = "correction" if tr.n_cases < 12 else "weights"
    if method not in ("correction", "weights"):
        raise ValueError("method must be 'auto', 'correction' or 'weights'")
    print(f"-- adapting to {tr.n_cases} cases with method '{method}'"
          f"{f', testing on {te.n_cases} held-out cases' if te is not None else ''}")
    before = model.evaluate(te) if te is not None else None
    if method == "correction":
        summary = model.fit_correction(tr, verbose=verbose)
        if va is not None:
            model.calibrate(va)
    else:
        summary = None
        model.fine_tune(tr, va, max_iter=max_iter, anchor=anchor, freeze=freeze, verbose=verbose)
    label = "discrepancy correction" if method == "correction" else "fine-tuning"
    study = Study("fine_tune_report", cfg["title"], os.path.join(out_dir, "figures"),
                  f"{label.capitalize()} of {model_dir} on {tr.n_cases} cases from "
                  f"{', '.join(data.sources)}.")
    if summary is not None:
        study.add_table("Discrepancy correction per target (leave-one-case-out CV, standardized units)",
                        ["target", "points", "corrected", "CV RMSE before", "CV RMSE after"],
                        [[t, int(r["points"]), "yes" if r["corrected"] else "no",
                          f"{r['cv_rmse_before_std']:.4f}", f"{r['cv_rmse_after_std']:.4f}"]
                         for t, r in summary.iterrows()])
    if te is not None:
        after = model.evaluate(te)
        rows = [[t, f"{before.loc[t, 'nrmse_pct']:.2f}", f"{after.loc[t, 'nrmse_pct']:.2f}",
                 f"{before.loc[t, 'r2']:.4f}", f"{after.loc[t, 'r2']:.4f}"]
                for t in after.index if t in before.index]
        study.add_table(f"Test accuracy before / after {label}",
                        ["target", "RMSE % range (before)", "RMSE % range (after)", "R2 (before)",
                         "R2 (after)"], rows)
        model.metrics["fine_tune_test"] = after.to_dict(orient="index")
    model.save(os.path.join(out_dir, "model"))
    _charts(study, model, cfg, tr, te, f" (pre-training + {label})")
    md, html = _report(study, out_dir, cfg["title"])
    print(f"\n-- adapted model: {os.path.join(out_dir, 'model')}\n   report: {md}")
    return model


def predict(model_dir: str, designs: Union[str, pd.DataFrame], out_dir: str,
            times=None, n_times: int = 200, crossing: Optional[dict] = None,
            n_plot: int = 4) -> pd.DataFrame:
    """Predict burnup trajectories for new designs (one row per design, or a
    long table that already contains times) -> CSV + charts."""
    from . import plots
    from .. import plotting as pl
    model = BurnupNN.load(model_dir)
    df = designs
    if isinstance(designs, str):
        df = BurnupDataset.load(designs, model.case_column, None).df.drop(columns="source")
    os.makedirs(out_dir, exist_ok=True)
    if model.time_column and model.time_column in df.columns:
        pred = pd.concat([df.reset_index(drop=True), model.predict(df.reset_index(drop=True))], axis=1)
    else:
        pred = model.predict_cases(df, times, n_times)
    ood = ~model.in_domain(pred, margin=0.02)
    pred["extrapolation"] = ood
    pred.to_csv(os.path.join(out_dir, "predictions.csv"), index=False, float_format="%.7g")
    if ood.any():
        print(f"  warning: {int(ood.sum())} rows lie outside the training range "
              "(column 'extrapolation' = True) - treat them with caution")
    if model.time_column:
        cid = model.case_column
        ids = list(pd.unique(pred[cid]))[:n_plot]
        targets = model.targets[:6]
        nrows, ncols = (int(np.ceil(len(targets) / 3)), min(3, len(targets)))
        fig, ax = pl.figure(nrows, ncols, (3.9 * ncols, 2.9 * nrows + 0.9),
                            "Predicted burnup trajectories",
                            "line = ensemble mean, band = 95% interval")
        ax = np.atleast_1d(ax).ravel()
        for i, c in enumerate(ids):
            d = pred[pred[cid] == c]
            for a, y in zip(ax, targets):
                m = d[y].to_numpy()
                lo, hi = plots.band(model, y, m, d[f"{y}_std"].to_numpy())
                a.fill_between(d[model.time_column], lo, hi, color=pl.SERIES[i], alpha=0.12, lw=0)
                a.plot(d[model.time_column], m, color=pl.SERIES[i], label=str(c) if a is ax[0] else None)
        for a, y in zip(ax, targets):
            a.set_title(plots.nice(y), loc="left", fontsize=10)
            a.set_xlabel(plots.nice(model.time_column))
        for a in ax[len(targets):]:
            a.set_visible(False)
        ax[0].legend(fontsize=8)
        pl.save(fig, os.path.join(out_dir, "predicted_trajectories.png"))
    if crossing and model.time_column:
        base = df.drop_duplicates(subset=[model.case_column]) if model.case_column in df.columns else df
        xc = model.crossing(base.reset_index(drop=True), crossing.get("target", "k_eff"),
                            float(crossing.get("level", 1.0)),
                            burnup_target=crossing.get("burnup_target")
                            if crossing.get("burnup_target") in model.targets else None)
        if model.case_column in base.columns:
            xc.insert(0, model.case_column, base[model.case_column].to_numpy())
        xc.to_csv(os.path.join(out_dir, "crossing.csv"), index=False, float_format="%.6g")
        print(xc.to_string(float_format=lambda v: f"{v:.4g}"))
    print(f"  predictions written to {out_dir}")
    return pred
