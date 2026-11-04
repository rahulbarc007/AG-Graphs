"""Consistent, accessible chart styling for every MSR-NX figure.

Palette (validated for colour-vision deficiency, fixed slot order), recessive
hairline grids, 2 px lines, ringed markers, one y-axis per panel (no dual
axes), sequential single-hue maps for magnitudes and a blue-grey-red
diverging map for signed differences.  Text uses ink colours, never series
colours.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
SEQ_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("msrnx_seq", SEQ_RAMP)
DIV = LinearSegmentedColormap.from_list("msrnx_div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec",
                                                      "#f2a9a9", "#e34948", "#a32626"])


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.labelsize": 10,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 7, "lines.markeredgewidth": 1.5, "lines.markeredgecolor": SURFACE,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK2,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "image.cmap": "msrnx_seq",
    })
    try:
        matplotlib.colormaps.register(SEQ)
    except ValueError:
        pass


apply_style()


def figure(nrows: int = 1, ncols: int = 1, size=(7.2, 4.4), title: Optional[str] = None,
           subtitle: Optional[str] = None, **kw):
    """Figure with a left-aligned title/subtitle band reserved above the panels."""
    fig, ax = plt.subplots(nrows, ncols, figsize=size, constrained_layout=True, **kw)
    band_in = (0.38 if title else 0.0) + (0.26 if subtitle else 0.0)
    if band_in > 0:
        top = 1.0 - band_in / size[1]
        fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, top))
        y = 1.0 - 0.10 / size[1]
        if title:
            fig.text(0.01, y, title, ha="left", va="top", fontsize=13, fontweight="bold",
                     color=INK)
            y -= 0.36 / size[1]
        if subtitle:
            fig.text(0.01, y, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    return fig, ax


def save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def end_label(ax, x, y, text, color=None, dx: float = 4, dy: float = 0, fontsize: float = 8.5):
    """Direct label at a line end (ink colour text with a small swatch)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center",
                fontsize=fontsize, color=INK2)
    if color is not None:
        ax.plot([x], [y], "o", color=color, markersize=6, markeredgecolor=SURFACE, zorder=5)


def value_label(ax, x, y, text, ha="left", va="center", dx=3, dy=0, fontsize=8.5, color=INK2):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                fontsize=fontsize, color=color)


def hbar(ax, labels: Sequence[str], values: Sequence[float], color: str = SERIES[0],
         fmt: str = "{:.3g}", errors=None, sort: bool = True, xlabel: str = ""):
    """Ranked horizontal bars with value labels (one hue for one series)."""
    labels = list(labels)
    values = np.asarray(values, float)
    order = np.argsort(np.abs(values)) if sort else np.arange(len(values))[::-1]
    labels = [labels[i] for i in order]
    values = values[order]
    err = None if errors is None else np.asarray(errors)[order]
    y = np.arange(len(values))
    ax.barh(y, values, height=0.62, color=color, edgecolor=SURFACE, linewidth=2, xerr=err,
            error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=AXIS, lw=0.8)
    span = np.abs(values).max() if len(values) else 1.0
    for yi, v in zip(y, values):
        ax.annotate(fmt.format(v), (v, yi), xytext=(3 if v >= 0 else -3, 0),
                    textcoords="offset points", ha="left" if v >= 0 else "right", va="center",
                    fontsize=8, color=INK2)
    ax.set_xlabel(xlabel)
    lo, hi = min(values.min(), 0), max(values.max(), 0)
    ax.set_xlim(lo - 0.18 * span * (lo < 0), hi + 0.22 * span * (hi > 0) + 1e-30)
    return ax


def group_bars(ax, categories, series: dict, fmt: str = "{:.3g}", ylabel: str = "",
               label_values: bool = True):
    """Grouped vertical bars, one categorical colour per series (fixed order)."""
    n = len(series)
    x = np.arange(len(categories))
    w = min(0.8 / n, 0.32)
    for i, (name, vals) in enumerate(series.items()):
        xi = x + (i - (n - 1) / 2) * w
        ax.bar(xi, vals, width=w, color=SERIES[i % 8], edgecolor=SURFACE, linewidth=2, label=name)
        if label_values:
            for xx, v in zip(xi, vals):
                ax.annotate(fmt.format(v), (xx, v), xytext=(0, 2 if v >= 0 else -2),
                            textcoords="offset points", ha="center",
                            va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK2)
    ax.set_xticks(x, categories)
    ax.grid(axis="x", visible=False)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel(ylabel)
    if n > 1:
        ax.legend(loc="best")
    return ax


def rz_map(ax, r_edges, z_edges, field2d, label: str = "", cmap=None, diverging: bool = False,
           outline=None):
    """Pseudocolour map of an (nr, nz) field on the R-Z mesh with a scale bar."""
    if diverging:
        vmax = np.nanmax(np.abs(field2d))
        im = ax.pcolormesh(r_edges, z_edges, field2d.T, cmap=DIV, vmin=-vmax, vmax=vmax,
                           shading="flat")
    else:
        im = ax.pcolormesh(r_edges, z_edges, field2d.T, cmap=cmap or SEQ, shading="flat")
    ax.set_xlabel("r [cm]")
    ax.set_ylabel("z [cm]")
    ax.set_aspect("equal")
    ax.grid(False)
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(label, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK2)
    if outline is not None:
        for (r0, r1, z0, z1) in outline:
            ax.plot([r0, r1, r1, r0, r0], [z0, z0, z1, z1, z0], color=SURFACE, lw=0.8)
    return im


def parity(ax, y_true, y_pred, y_std=None, scale: float = 1.0, unit: str = "",
           color: str = SERIES[0], label: Optional[str] = None):
    yt = np.asarray(y_true) * scale
    yp = np.asarray(y_pred) * scale
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    pad = 0.04 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=AXIS, lw=1, zorder=1)
    if y_std is not None:
        ax.errorbar(yt, yp, yerr=1.96 * np.asarray(y_std) * scale, fmt="none", ecolor=GRID,
                    elinewidth=1, zorder=2)
    ax.scatter(yt, yp, s=26, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3, label=label)
    ax.set_xlabel(f"physics solver {unit}")
    ax.set_ylabel(f"ML prediction {unit}")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    return ax


def step_groups(ax, bounds, values, color=SERIES[0], label=None, per_lethargy: bool = False):
    b = np.asarray(bounds)
    v = np.asarray(values, float)
    if per_lethargy:
        v = v / np.log(b[:-1] / b[1:])
    ax.stairs(v[::-1], b[::-1], color=color, lw=2, label=label, baseline=None)
    ax.set_xscale("log")
    return ax


def kpi_row(ax, items, title: Optional[str] = None):
    """Draw a row of stat tiles (label + value) on an empty axes."""
    ax.axis("off")
    n = len(items)
    for i, (lab, val) in enumerate(items):
        x = (i + 0.5) / n
        ax.text(x, 0.62, val, ha="center", va="center", fontsize=17, fontweight="bold", color=INK,
                transform=ax.transAxes)
        ax.text(x, 0.22, lab, ha="center", va="center", fontsize=8.5, color=INK2,
                transform=ax.transAxes)
    if title:
        ax.set_title(title, loc="left")
    return ax
