"""Charts for the neural-network burnup framework (MSR-NX house style).

Every function saves one PNG and returns its path.  Conventions follow
:mod:`msrnx.plotting`: fixed categorical order, sequential single-hue ramp for
magnitudes, hairline grids, one y-axis per panel, legends for two or more
series, text in ink colours.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .. import plotting as pl
from .model import BurnupNN, is_pcm_target

plt = pl.plt
LINE_CMAP = pl.LinearSegmentedColormap.from_list("burnup_lines", pl.SEQ_RAMP[3:])


def nice(name: str) -> str:
    """Readable axis label from a column name."""
    rep = {"k_eff": "k-eff", "burnup_MWd_kgHM": "burnup [MWd/kgHM]",
           "breeding_ratio": "breeding ratio", "time_days": "time [days]",
           "fissile_kg": "fissile inventory [kg]", "Xe135_abs_pct": "Xe-135 absorption [%]",
           "Sm149_abs_pct": "Sm-149 absorption [%]", "Pa233_abs_pct": "Pa-233 absorption [%]",
           "FP_abs_pct": "fission-product absorption [%]", "tritium_Ci_day": "tritium [Ci/day]",
           "flux_n_cm2_s": "flux [n/cm2/s]", "uf4_pct": "UF4 [mol%]", "thf4_pct": "ThF4 [mol%]",
           "salt_fraction": "salt volume fraction", "power_MW": "power [MW]",
           "re_removal_days": "rare-earth removal [d]", "pa_removal_days": "Pa removal [d]",
           "xe_removal_s": "Xe sparging time [s]"}
    if name in rep:
        return rep[name]
    if name.endswith("_kg"):
        n = name[:-3]
        iso = "".join(c for c in n if c.isalpha()) + "-" + "".join(c for c in n if c.isdigit())
        return f"{iso} [kg]" if any(c.isdigit() for c in n) else f"{n} [kg]"
    return name.replace("_", " ")


def _grid(n: int, ncols: int = 4):
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def _axes(fig_ax, n):
    ax = np.atleast_1d(fig_ax).ravel()
    for a in ax[n:]:
        a.set_visible(False)
    return ax


def _scale_unit(target: str):
    return (1e5, "pcm") if is_pcm_target(target) else (1.0, "")


# ---------------------------------------------------------------------------
def data_overview(ds, targets: Sequence[str], path: str, color_by: Optional[str] = None,
                  max_cases: int = 150, title: Optional[str] = None, seed: int = 0) -> str:
    """Small multiples: every target vs time, one thin line per case, coloured
    by a feature (sequential ramp + colour bar)."""
    targets = list(targets)
    t = ds.time_column
    cases = ds.cases
    if len(cases) > max_cases:
        cases = np.random.default_rng(seed).choice(cases, max_cases, replace=False)
    sub = ds.subset(cases).df
    nrows, ncols = _grid(len(targets), 4)
    fig, ax = pl.figure(nrows, ncols, (3.3 * ncols, 2.7 * nrows + 0.7),
                        title or f"Training data: {ds.n_cases} burnup trajectories",
                        f"{len(sub)} rows shown from {len(cases)} cases"
                        + (f"; colour = {nice(color_by)}" if color_by else ""))
    ax = _axes(ax, len(targets))
    norm = None
    if color_by:
        v = sub.groupby(ds.case_column, sort=False)[color_by].first()
        lo, hi = v.min(), v.max()
        norm = pl.matplotlib.colors.Normalize(lo, hi if hi > lo else lo + 1)
    for a, y in zip(ax, targets):
        for cid, d in sub.groupby(ds.case_column, sort=False):
            col = LINE_CMAP(norm(d[color_by].iloc[0])) if norm else pl.SERIES[0]
            a.plot(d[t], d[y], color=col, lw=0.8, alpha=0.75)
        a.set_title(nice(y), loc="left", fontsize=10)
        a.set_xlabel(nice(t))
    if norm is not None:
        sm = pl.matplotlib.cm.ScalarMappable(norm=norm, cmap=LINE_CMAP)
        cb = fig.colorbar(sm, ax=ax[:len(targets)].tolist(), shrink=0.6, pad=0.01)
        cb.set_label(nice(color_by), color=pl.INK2)
        cb.outline.set_visible(False)
    return pl.save(fig, path)


def history(model: BurnupNN, path: str, title: str = "Training history") -> str:
    """Training objective and validation loss of every ensemble member."""
    fig, ax = pl.figure(1, 1, (7.6, 4.2), title,
                        f"{len(model.nets)} ensemble members; loss = weighted MSE of "
                        "standardized targets")
    phases = sorted({h.get("phase", "train") for h in model.history},
                    key=lambda p: (p != "train", p))
    offset = 0
    for ph in phases:
        hs = [h for h in model.history if h.get("phase", "train") == ph]
        top = 0
        for i, h in enumerate(hs):
            it = np.asarray(h["iter"], float) + offset
            ax.plot(it, h["train"], color=pl.SERIES[0], lw=1.2, alpha=0.8,
                    label="training objective" if (i == 0 and ph == phases[0]) else None)
            if np.isfinite(h["val"]).any() and h.get("val") != h.get("train"):
                ax.plot(it, h["val"], color=pl.SERIES[1], lw=1.2, alpha=0.8,
                        label="validation loss" if (i == 0 and ph == phases[0]) else None)
            top = max(top, it.max() if len(it) else 0)
        if len(phases) > 1:
            ax.axvline(offset, color=pl.AXIS, lw=0.8)
            ax.text(offset, 0.98, f" {ph}", transform=ax.get_xaxis_transform(), va="top",
                    ha="left", fontsize=8, color=pl.INK2)
        offset = top
    ax.set_yscale("log")
    ax.set_xlabel("iteration" if model.history and model.history[0].get("optimizer") == "lbfgs"
                  else "epoch")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")
    return pl.save(fig, path)


def parity_grid(model: BurnupNN, data, targets: Optional[Sequence[str]], path: str,
                title: str = "Neural network vs physics on unseen cases") -> str:
    """Predicted vs reference for every target (R^2 and RMSE in the panel title)."""
    targets = list(targets or model.targets)
    df = data.df if hasattr(data, "df") else data
    pred = model.predict(df)
    nrows, ncols = _grid(len(targets), 4)
    fig, ax = pl.figure(nrows, ncols, (3.2 * ncols, 3.1 * nrows + 0.5), title,
                        f"{len(df)} points from {df[model.case_column].nunique()} cases; "
                        "grey line = perfect agreement")
    ax = _axes(ax, len(targets))
    for a, y in zip(ax, targets):
        o = df[y].to_numpy(float)
        p = pred[y].to_numpy(float)
        ok = np.isfinite(o)
        sc, unit = _scale_unit(y)              # values in natural units, RMSE in pcm for k
        o, p = o[ok], p[ok]
        lo, hi = min(o.min(), p.min()), max(o.max(), p.max())
        pad = 0.04 * (hi - lo or 1.0)
        a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=pl.AXIS, lw=1, zorder=1)
        a.scatter(o, p, s=9, color=pl.SERIES[0], alpha=0.55, linewidths=0, zorder=2)
        e = p - o
        rmse = np.sqrt(np.mean(e ** 2)) * sc
        r2 = 1 - (e ** 2).sum() / max(((o - o.mean()) ** 2).sum(), 1e-300)
        a.set_title(nice(y), loc="left", fontsize=9.5)
        txt = f"R2 {r2:.4f}\nRMSE {rmse:.3g}{' ' + unit if unit else ''}"
        a.text(0.04, 0.96, txt, transform=a.transAxes, va="top", ha="left", fontsize=8,
               color=pl.INK2)
        a.set_xlim(lo - pad, hi + pad)
        a.set_ylim(lo - pad, hi + pad)
        a.set_xlabel("reference")
        a.set_ylabel("neural network")
    return pl.save(fig, path)


def band(model: BurnupNN, target: str, mean, std, z: float = 1.96):
    """Lower/upper interval bounds; clipped at zero for quantities that were
    never negative in the training data."""
    lo, hi = mean - z * std, mean + z * std
    if model.nonnegative(target):
        lo = np.maximum(lo, 0.0)
    return lo, hi


def _time_axis(a, scale: str, tmax: float):
    if scale == "symlog":
        lin = max(tmax * 1e-3, 1e-9)
        a.set_xscale("symlog", linthresh=lin, linscale=0.35)
        a.set_xlim(left=0.0)
        first = int(np.ceil(np.log10(3.0 * lin)))
        ticks = [0.0] + [10.0 ** k for k in range(first, int(np.floor(np.log10(tmax))) + 1)]
        a.set_xticks(ticks, [f"{v:,.0f}" if v >= 1 else f"{v:g}" for v in ticks])
        a.xaxis.set_minor_locator(mticker.NullLocator())


def trajectories(model: BurnupNN, data, targets: Sequence[str], path: str,
                 cases: Optional[Sequence[str]] = None, n_cases: int = 3,
                 title: str = "Burnup trajectories: physics vs neural network",
                 reference_label: str = "physics", n_times: int = 250,
                 time_scale: str = "linear", subtitle: Optional[str] = None) -> str:
    """Selected cases: reference points, ensemble mean and 95 % band.

    The y-range follows the reference points and the predicted means (a very
    uncertain model shows bands running off the panel instead of flattening
    the curves); ``time_scale="symlog"`` stretches the first days so that the
    Xe-135 / Pa-233 transients are visible."""
    targets = list(targets)
    t = model.time_column
    if cases is None:
        ids = data.cases
        pick = np.linspace(0, len(ids) - 1, min(n_cases, len(ids))).round().astype(int)
        cases = [ids[i] for i in pick]
    cases = list(cases)[:4]
    design = data.case_table(model.features).loc[cases].reset_index()
    tmax = max(float(data.df[t].max()), 1e-9)
    pred = model.predict_cases(design, times=model.time_grid(n_times, tmax),
                               case_column=data.case_column)
    nrows, ncols = _grid(len(targets), 3)
    fig, ax = pl.figure(nrows, ncols, (3.9 * ncols, 2.9 * nrows + 0.9), title,
                        subtitle or f"markers = {reference_label}, line = ensemble mean, "
                                    "band = 95% interval")
    ax = _axes(ax, len(targets))
    span = {y: [np.inf, -np.inf] for y in targets}
    for i, cid in enumerate(cases):
        col = pl.SERIES[i]
        ref = data.case(cid)
        pr = pred[pred[data.case_column] == str(cid)]
        for a, y in zip(ax, targets):
            m = pr[y].to_numpy()
            lo, hi = band(model, y, m, pr[f"{y}_std"].to_numpy())
            a.fill_between(pr[t], lo, hi, color=col, alpha=0.12, lw=0)
            a.plot(pr[t], m, color=col, lw=1.8, label=str(cid) if a is ax[0] else None)
            vals = [m]
            if y in ref.columns:
                a.plot(ref[t], ref[y], "o", color=col, ms=4.5, mec=pl.SURFACE, mew=1.0)
                vals.append(ref[y].to_numpy(float))
            v = np.concatenate(vals)
            v = v[np.isfinite(v)]
            span[y] = [min(span[y][0], v.min()), max(span[y][1], v.max())]
    for a, y in zip(ax, targets):
        lo, hi = span[y]
        pad = 0.12 * ((hi - lo) or abs(hi) or 1.0)
        a.set_ylim(max(lo - pad, 0.0) if model.nonnegative(y) and lo >= 0 else lo - pad, hi + pad)
        a.set_title(nice(y), loc="left", fontsize=10)
        a.set_xlabel(nice(t) + (" (symlog)" if time_scale == "symlog" else ""))
        _time_axis(a, time_scale, tmax)
    ax[0].legend(loc="best", fontsize=8)
    return pl.save(fig, path)


def error_vs_time(model: BurnupNN, data, targets: Sequence[str], path: str, n_bins: int = 12,
                  title: str = "Where along the trajectory are the errors?") -> str:
    """Median and 90th percentile of |error| (normalized by the target range)
    in time bins: one panel per target."""
    targets = list(targets)
    df = data.df if hasattr(data, "df") else data
    pred = model.predict(df)
    t = df[model.time_column].to_numpy(float)
    tpos = np.where(t > 0, t, np.nan)
    edges = np.unique(np.concatenate([[0.0], np.geomspace(np.nanmin(tpos), t.max(), n_bins)]))
    nrows, ncols = _grid(len(targets), 4)
    fig, ax = pl.figure(nrows, ncols, (3.2 * ncols, 2.6 * nrows + 0.7), title,
                        "|error| as % of the target range in the reference data; "
                        "line = median, band = 10-90th percentile")
    ax = _axes(ax, len(targets))
    for a, y in zip(ax, targets):
        o = df[y].to_numpy(float)
        ok = np.isfinite(o)
        rng = (np.nanmax(o) - np.nanmin(o)) or 1.0
        e = 100 * np.abs(pred[y].to_numpy(float) - o) / rng
        xs, med, lo, hi = [], [], [], []
        for a0, a1 in zip(edges[:-1], edges[1:]):
            m = ok & (t >= a0) & (t <= a1 if a1 == edges[-1] else t < a1)
            if m.sum() >= 3:
                xs.append(np.sqrt(max(a0, edges[1] * 0.5) * a1))
                med.append(np.median(e[m]))
                lo.append(np.quantile(e[m], 0.1))
                hi.append(np.quantile(e[m], 0.9))
        a.fill_between(xs, lo, hi, color=pl.SERIES[0], alpha=0.12, lw=0)
        a.plot(xs, med, "o-", color=pl.SERIES[0], ms=4)
        a.set_xscale("log")
        a.set_title(nice(y), loc="left", fontsize=10)
        a.set_xlabel(nice(model.time_column))
        a.set_ylabel("|error| [% of range]")
        a.set_ylim(bottom=0)
    return pl.save(fig, path)


def accuracy_bars(metrics: pd.DataFrame, path: str, title: str = "Accuracy per output",
                  subtitle: Optional[str] = None) -> str:
    """Ranked bars of the normalized RMSE (% of range) per target + R^2 labels."""
    m = metrics.copy()
    labels = [f"{nice(i)}   R2 {m.loc[i, 'r2']:.4f}" for i in m.index]
    fig, ax = pl.figure(1, 1, (8.2, 0.42 * len(m) + 1.4), title,
                        subtitle or "RMSE as % of the reference range on unseen cases (lower is better)")
    pl.hbar(ax, labels, m["nrmse_pct"].to_numpy(float), fmt="{:.2f}%", xlabel="normalized RMSE [%]")
    return pl.save(fig, path)


def calibration(model: BurnupNN, data, path: str,
                title: str = "Is the predicted uncertainty trustworthy?") -> str:
    """Reliability of the ensemble intervals (pooled over targets) and error vs sigma."""
    df = data.df if hasattr(data, "df") else data
    pred = model.predict(df)
    levels = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    from scipy.stats import norm
    zq = norm.ppf(0.5 + levels / 2)
    zs, per_target = [], {}
    for y in model.targets:
        if y not in df.columns:
            continue
        o = df[y].to_numpy(float)
        s = pred[f"{y}_std"].to_numpy(float)
        ok = np.isfinite(o) & (s > 0)
        if ok.sum() < 5:
            continue
        z = np.abs(pred[y].to_numpy(float)[ok] - o[ok]) / s[ok]
        zs.append(z)
        per_target[y] = np.array([np.mean(z <= q) for q in zq])
    fig, ax = pl.figure(1, 2, (11, 4.3), title,
                        "Left: share of unseen points inside the central interval; "
                        "right: actual error vs predicted sigma")
    if not zs:
        ax[0].text(0.5, 0.5, "single network: no ensemble spread", ha="center",
                   transform=ax[0].transAxes, color=pl.INK2)
        return pl.save(fig, path)
    z = np.concatenate(zs)
    cov = np.array([np.mean(z <= q) for q in zq])
    band = np.array(list(per_target.values()))
    ax[0].plot([0.45, 1.0], [0.45, 1.0], color=pl.AXIS, lw=1)
    ax[0].fill_between(levels, band.min(axis=0), band.max(axis=0), color=pl.SERIES[0], alpha=0.12,
                       lw=0, label="range over outputs")
    ax[0].plot(levels, cov, "o-", color=pl.SERIES[0], label="all outputs pooled")
    pl.value_label(ax[0], 0.95, cov[levels == 0.95][0], f"95% -> {cov[levels == 0.95][0]:.0%}",
                   ha="right", dx=-6, dy=8)
    ax[0].set_xlabel("nominal coverage")
    ax[0].set_ylabel("observed coverage")
    ax[0].legend(loc="upper left")
    # error vs sigma (pooled over targets, standardized by each target's range)
    E, S = [], []
    for y in per_target:
        o = df[y].to_numpy(float)
        ok = np.isfinite(o)
        rng = (np.nanmax(o) - np.nanmin(o)) or 1.0
        E.append(np.abs(pred[y].to_numpy(float)[ok] - o[ok]) / rng * 100)
        S.append(pred[f"{y}_std"].to_numpy(float)[ok] / rng * 100)
    E, S = np.concatenate(E), np.concatenate(S)
    q = np.quantile(S, np.linspace(0, 1, 11))
    xs, ys = [], []
    for a0, a1 in zip(q[:-1], q[1:]):
        m = (S >= a0) & (S <= a1)
        if m.sum() > 5:
            xs.append(S[m].mean())
            ys.append(E[m].mean())
    hi = max(max(xs), max(ys)) * 1.1
    ax[1].plot([0, hi], [0, 0.798 * hi], color=pl.AXIS, lw=1)
    pl.value_label(ax[1], hi, 0.798 * hi, "ideal (0.8 sigma)", ha="right", dx=-4, dy=-8)
    ax[1].plot(xs, ys, "o-", color=pl.SERIES[0])
    ax[1].set_xlabel("predicted 1-sigma [% of range]")
    ax[1].set_ylabel("mean |error| [% of range]")
    ax[1].set_xlim(0, hi)
    ax[1].set_ylim(0, hi)
    return pl.save(fig, path)


def sensitivity_chart(sens: pd.DataFrame, target: str, path: str, unit: str = "",
                      title: Optional[str] = None) -> str:
    """Signed sensitivity bars (change of the target per 1 sigma of each input)."""
    fig, ax = pl.figure(1, 1, (7.4, 0.45 * len(sens) + 1.5),
                        title or f"What drives {nice(target)}?",
                        "Change per one standard deviation of each input over the training "
                        "data (log inputs: per sigma of log); ensemble-mean gradients")
    labels = [nice(f) for f in sens["feature"]]
    pl.hbar(ax, labels, sens["mean_signed"].to_numpy(float), fmt="{:+.3g}",
            xlabel=f"d {nice(target)} per 1 sigma of input" + (f" [{unit}]" if unit else ""))
    return pl.save(fig, path)


def crossing_map(x, y, Z, Zs, path: str, xlabel: str, ylabel: str, zlabel: str,
                 title: str, subtitle: str = "", points=None, levels=None) -> str:
    """Map of a burnup quantity (left) and its 1-sigma uncertainty (right).

    ``points``: optional list of (x, y, label) markers (e.g. physics checks).
    """
    fig, ax = pl.figure(1, 2, (12, 4.9), title, subtitle)
    Zm = np.ma.masked_invalid(Z)
    im = ax[0].pcolormesh(x, y, Zm, cmap=pl.SEQ, shading="auto")
    cs = ax[0].contour(x, y, Zm, levels=levels or 6, colors=pl.SURFACE, linewidths=0.8)
    ax[0].clabel(cs, fmt="%.0f", fontsize=7.5, colors=pl.SURFACE)
    cb = fig.colorbar(im, ax=ax[0], shrink=0.9, pad=0.02)
    cb.set_label(zlabel, color=pl.INK2)
    cb.outline.set_visible(False)
    im2 = ax[1].pcolormesh(x, y, np.ma.masked_invalid(Zs), cmap=pl.SEQ, shading="auto")
    cb2 = fig.colorbar(im2, ax=ax[1], shrink=0.9, pad=0.02)
    cb2.set_label(f"1-sigma of {zlabel}", color=pl.INK2)
    cb2.outline.set_visible(False)
    for a in ax:
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)
        a.grid(False)
    if points:
        for px, py, lab in points:
            ax[0].plot(px, py, "o", color=pl.SERIES[1], ms=7, mec=pl.SURFACE, mew=1.5)
            ax[0].annotate(lab, (px, py), xytext=(5, 5), textcoords="offset points", fontsize=7.5,
                           color=pl.INK, bbox=dict(boxstyle="round,pad=0.2", fc=pl.SURFACE,
                                                   ec="none", alpha=0.85))
    ax[0].set_title("ensemble mean", loc="left", fontsize=10)
    ax[1].set_title("ensemble uncertainty", loc="left", fontsize=10)
    return pl.save(fig, path)


def series_curves(x, series: dict, path: str, xlabel: str, ylabel: str, title: str,
                  subtitle: str = "", logx: bool = False, logy: bool = False,
                  hlines: Optional[dict] = None, fmt: str = "{:.0f}") -> str:
    """Generic multi-series curve chart (learning curves, transfer learning...)
    with a legend and direct end labels; ``hlines``: {label: value}."""
    fig, ax = pl.figure(1, 1, (7.6, 4.4), title, subtitle)
    i = 0
    for i, (lab, ys) in enumerate(series.items()):
        ys = np.asarray(ys, float)
        ax.plot(x, ys, "o-", color=pl.SERIES[i], label=lab)
        pl.end_label(ax, x[-1], ys[-1], fmt.format(ys[-1]))
    for j, (lab, v) in enumerate((hlines or {}).items()):
        col = pl.SERIES[(i + 1 + j) % 8]
        ax.axhline(v, color=col, lw=1.5, label=lab)
    if logx:
        ax.set_xscale("log")
        ax.set_xticks(x, [str(v) for v in x])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8.5)
    return pl.save(fig, path)
