"""Neural-network burnup model:  (design features, time) -> burnup quantities.

``BurnupNN`` learns every target of a :class:`BurnupDataset` at once with a deep
ensemble of fully connected networks and adds the physics knowledge that makes
small depletion data sets go a long way:

* **multi-timescale time encoding** - besides t and sqrt(t) the network sees
  saturation functions 1 - exp(-t/tau) on log-spaced tau (default 3e-4 ... 0.3
  of the time span): depletion solutions are sums of exponentials (Xe-135 in
  hours, Pm/Sm-149 in days, Pa-233 in weeks, actinides over years), so the
  network only has to learn their amplitudes;
* **positivity** - inventories spanning decades are learned in log space and can
  never become negative;
* **monotonicity constraints** - e.g. burnup, U-232/U-236 build-up or Th-232
  consumption must not reverse: penalised on collocation points (finite
  differences in time, so it also holds between data points);
* **masked loss** - blank target cells (codes reporting different nuclides) are
  ignored, per-row weights allow trusting some sources more;
* **uncertainty** - ensemble spread, calibrated on validation cases;
* **transfer learning** - ``fine_tune`` adapts a model pre-trained on many cheap
  runs to a few high-fidelity runs (weights anchored to the pre-trained values).
"""
from __future__ import annotations

import contextlib
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .dataset import BurnupDataset, DataError
from .nn import MLP, fit_adam, fit_lbfgs

FORMAT_VERSION = 1
PCM_TARGET_HINTS = ("k_eff", "keff", "k_inf", "kinf", "k_bos")


@dataclass
class NNConfig:
    """Network and training hyper-parameters (all overridable from a config file)."""
    hidden: tuple = (64, 64, 64)       # hidden layer widths
    activation: str = "tanh"           # tanh | silu | gelu | softplus | relu
    residual: bool = False             # residual connections between equal-width layers
    ensemble: int = 5                  # members of the deep ensemble (uncertainty)
    optimizer: str = "auto"            # lbfgs (full batch) | adam (mini-batch) | auto
    max_iter: int = 5000               # L-BFGS iterations
    epochs: int = 1500                 # Adam epochs
    batch_size: int = 256              # Adam mini-batch
    lr: float = 2e-3                   # Adam learning rate
    l2: float = 1e-6                   # weight penalty (L-BFGS) / decoupled decay (Adam)
    patience: int = 40                 # early stopping: checks (L-BFGS) or epochs (Adam)
    check_every: int = 10              # L-BFGS iterations between validation checks
    lbfgs_memory: int = 100            # L-BFGS curvature pairs
    time_encoding: str = "multiscale"  # multiscale | linear
    n_time_scales: int = 4             # saturation features of the multiscale encoding
    monotonic_weight: float = 1.0      # weight of the monotonicity penalty
    n_collocation: int = 2048          # collocation points for the constraints (max)
    bootstrap: bool = False            # resample cases for each member (bagging)
    n_jobs: int = 0                    # parallel processes for members (0 = auto)
    seed: int = 0

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "NNConfig":
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown model options {sorted(unknown)}; valid: {sorted(names)}")
        if "hidden" in d:
            d["hidden"] = tuple(int(h) for h in d["hidden"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Input / output transforms
# ---------------------------------------------------------------------------
class TimeEncoding:
    """t -> [t, sqrt(t), 1-exp(-t/tau_1..k)] mapped to [-1, 1]."""

    def __init__(self, kind: str = "multiscale", t_max: Optional[float] = None,
                 taus: Optional[Sequence[float]] = None, n_scales: int = 4):
        if kind not in ("multiscale", "linear"):
            raise ValueError("time_encoding must be 'multiscale' or 'linear'")
        self.kind, self.t_max, self.n_scales = kind, t_max, n_scales
        self.taus = None if taus is None else [float(x) for x in taus]

    def fit(self, t) -> "TimeEncoding":
        t = np.asarray(t, float)
        if self.t_max is None:
            self.t_max = float(np.nanmax(t)) if np.isfinite(t).any() else 1.0
        self.t_max = self.t_max if self.t_max > 0 else 1.0
        if self.kind == "multiscale" and self.taus is None:
            self.taus = list(self.t_max * np.logspace(np.log10(3e-4), np.log10(0.3),
                                                      self.n_scales))
        return self

    @property
    def names(self) -> list:
        if self.kind == "linear":
            return ["t"]
        return ["t", "sqrt_t"] + [f"sat_{tau:.3g}" for tau in self.taus]

    def transform(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        s = t / self.t_max
        cols = [2.0 * s - 1.0]
        if self.kind == "multiscale":
            cols.append(2.0 * np.sqrt(np.maximum(s, 0.0)) - 1.0)
            cols += [1.0 - 2.0 * np.exp(-np.maximum(t, 0.0) / tau) for tau in self.taus]
        return np.column_stack(cols)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "t_max": self.t_max, "taus": self.taus,
                "n_scales": self.n_scales}

    @classmethod
    def from_dict(cls, d: dict) -> "TimeEncoding":
        return cls(d["kind"], d["t_max"], d["taus"], d.get("n_scales", 4))


class ColumnTransform:
    """Optional log (with offset, for non-negative data) then standardization."""

    def __init__(self, log: bool = False, offset: float = 0.0, mean: float = 0.0,
                 std: float = 1.0):
        self.log, self.offset, self.mean, self.std = bool(log), float(offset), float(mean), float(std)

    @classmethod
    def fit(cls, x, log: bool = False) -> "ColumnTransform":
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        offset = 0.0
        if log:
            if len(x) == 0 or x.min() < 0:
                log = False
            else:
                pos = x[x > 0]
                offset = 1e-3 * float(np.median(pos)) if len(pos) else 1.0
        z = np.log(np.maximum(x, 0.0) + offset) if log else x
        mean = float(z.mean()) if len(z) else 0.0
        std = float(z.std()) if len(z) else 1.0
        if not np.isfinite(std) or std < 1e-12 * max(1.0, abs(mean)):
            std = 1.0
        return cls(log, offset, mean, std)

    def forward(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        z = np.log(np.maximum(x, 0.0) + self.offset) if self.log else x
        return (z - self.mean) / self.std

    def inverse(self, u) -> np.ndarray:
        z = np.asarray(u, float) * self.std + self.mean
        return np.maximum(np.exp(z) - self.offset, 0.0) if self.log else z

    def dinverse(self, u) -> np.ndarray:
        """d(physical) / d(standardized)."""
        z = np.asarray(u, float) * self.std + self.mean
        return self.std * np.exp(z) if self.log else np.full_like(z, self.std)

    def to_dict(self) -> dict:
        return {"log": self.log, "offset": self.offset, "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnTransform":
        return cls(**d)


def _auto_log_target(df: pd.DataFrame, col: str, case_col: str) -> bool:
    x = df[col].to_numpy(float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or x.min() < 0:
        return False
    peaks = df.groupby(case_col, sort=False)[col].max().to_numpy(float)
    peaks = peaks[np.isfinite(peaks) & (peaks > 0)]
    if len(peaks) < 3:
        return False
    lo, hi = np.quantile(peaks, [0.1, 0.9])
    return bool(lo > 0 and hi / lo > 30.0)


def _auto_log_feature(x) -> bool:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return bool(len(x) and x.min() > 0 and x.max() / x.min() > 30.0)


def _resolve(spec, names, auto_fn) -> dict:
    """'auto' | 'none' | 'all' | list of names  ->  {name: bool}."""
    if spec in (None, "none", False):
        return {n: False for n in names}
    if spec in ("all", True):
        return {n: True for n in names}
    if spec == "auto":
        return {n: auto_fn(n) for n in names}
    chosen = set(spec)
    unknown = chosen - set(names)
    if unknown:
        raise ValueError(f"log transform requested for unknown columns {sorted(unknown)}")
    return {n: n in chosen for n in names}


def is_pcm_target(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in PCM_TARGET_HINTS)


# ---------------------------------------------------------------------------
# Training (module level so that ensemble members can run in worker processes)
# ---------------------------------------------------------------------------
def _make_objective(X, Y, M, W, tw, l2, colloc, mono_w, anchor, anchor_w, seed):
    sw_full = W[:, None] * M
    rng = np.random.default_rng(seed)

    def objective(net, idx):
        if idx is None:
            Xb, Yb, Sb = X, Y, sw_full
        else:
            Xb, Yb, Sb = X[idx], Y[idx], sw_full[idx]
        P, cache = net.forward(Xb, keep=True)
        R = P - Yb
        denom = Sb.sum(axis=0)
        present = denom > 0
        twb = np.where(present, tw, 0.0)
        twb = twb / max(twb.sum(), 1e-300)
        coef = np.where(present, twb / np.where(present, denom, 1.0), 0.0)
        SR = Sb * R
        loss = float((coef * (SR * R).sum(axis=0)).sum())
        grads = net.backward(cache, 2.0 * SR * coef)
        if l2 > 0:
            for k in range(0, len(grads), 2):
                Wk = net.params[k]
                loss += 0.5 * l2 * float((Wk * Wk).sum())
                grads[k] = grads[k] + l2 * Wk
        if anchor is not None and anchor_w > 0:
            for k, a in enumerate(anchor):
                d = net.params[k] - a
                loss += 0.5 * anchor_w * float((d * d).sum())
                grads[k] = grads[k] + anchor_w * d
        if colloc is not None and mono_w > 0:
            X0, X1, dtn, mi, ms = colloc
            if idx is not None:
                pick = rng.integers(0, len(X0), min(len(X0), max(64, len(idx))))
                X0, X1, dtn = X0[pick], X1[pick], dtn[pick]
            P0, c0 = net.forward(X0, keep=True)
            P1, c1 = net.forward(X1, keep=True)
            S = (P1[:, mi] - P0[:, mi]) * ms / dtn[:, None]
            V = np.maximum(-S, 0.0)
            lam = mono_w / V.size
            loss += lam * float((V * V).sum())
            dS = -2.0 * lam * V
            dP1 = np.zeros_like(P1)
            dP1[:, mi] = dS * ms / dtn[:, None]
            g1 = net.backward(c1, dP1)
            g0 = net.backward(c0, -dP1)
            grads = [a + b + c for a, b, c in zip(grads, g1, g0)]
        return loss, grads

    return objective


def _data_loss(net, X, Y, M, W, tw) -> float:
    if X is None or len(X) == 0:
        return float("nan")
    S = W[:, None] * M
    R = net.forward(X) - Y
    denom = S.sum(axis=0)
    present = denom > 0
    twb = np.where(present, tw, 0.0)
    twb = twb / max(twb.sum(), 1e-300)
    return float((np.where(present, twb / np.where(present, denom, 1.0), 0.0)
                  * (S * R * R).sum(axis=0)).sum())


def _blas_single_thread():
    """One BLAS thread per member: the elementwise work dominates and the L-BFGS
    internals pay heavily for thread wake-ups on short vectors."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(1)
    except Exception:  # pragma: no cover - threadpoolctl ships with scikit-learn
        return contextlib.nullcontext()


def _train_member(payload: dict):
    with _blas_single_thread():
        cfg = payload["cfg"]
        if payload.get("init") is not None:
            net = MLP.from_state(payload["state"], payload["init"])
        else:
            s = payload["state"]["sizes"]
            net = MLP(s[0], s[-1], s[1:-1], cfg["activation"], cfg["residual"], payload["seed"])
        X, Y, M, W = payload["X"], payload["Y"], payload["M"], payload["W"]
        tw = payload["tw"]
        objective = _make_objective(X, Y, M, W, tw, cfg["l2"] if cfg["optimizer"] == "lbfgs" else 0.0,
                                    payload.get("colloc"), cfg["monotonic_weight"],
                                    payload.get("anchor"), payload.get("anchor_w", 0.0),
                                    payload["seed"] + 17)
        Xv = payload.get("Xv")
        val_fn = None
        if Xv is not None and len(Xv):
            Yv, Mv, Wv = payload["Yv"], payload["Mv"], payload["Wv"]
            val_fn = lambda n: _data_loss(n, Xv, Yv, Mv, Wv, tw)  # noqa: E731
        frozen = tuple(payload.get("frozen", ()))
        if cfg["optimizer"] == "lbfgs":
            hist = fit_lbfgs(net, objective, val_fn, payload.get("max_iter", cfg["max_iter"]),
                             cfg["patience"], cfg["check_every"], frozen,
                             memory=cfg.get("lbfgs_memory", 100))
        else:
            hist = fit_adam(net, objective, len(X), val_fn, payload.get("epochs", cfg["epochs"]),
                            cfg["batch_size"], payload.get("lr", cfg["lr"]), cfg["l2"],
                            patience=cfg["patience"], seed=payload["seed"], frozen=frozen)
        hist["train_data_loss"] = _data_loss(net, X, Y, M, W, tw)
        return [p.copy() for p in net.params], hist


def _run_members(payloads: list, n_jobs: int, verbose: bool) -> list:
    t0 = time.time()
    if n_jobs > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(payloads))) as ex:
            out = list(ex.map(_train_member, payloads))
    else:
        out = [_train_member(p) for p in payloads]
    if verbose:
        its = [h.get("n_iter", len(h["iter"])) for _, h in out]
        print(f"  trained {len(out)} member(s) in {time.time() - t0:.1f} s "
              f"(iterations {min(its)}-{max(its)})")
    return out


# ---------------------------------------------------------------------------
class BurnupNN:
    """Deep-ensemble neural network for burnup trajectories.

    Parameters
    ----------
    features      input columns (design / operating parameters)
    targets       output columns (k_eff, inventories, ...)
    time_column   time-like column (days or MWd/kgHM); ``None`` = static model
    log_targets   ``"auto"`` (log for non-negative quantities whose case maxima
                  span > 30x), ``"none"``, ``"all"`` or a list of names
    log_features  same for inputs (``"auto"``: positive with max/min > 30)
    monotonic     ``{target: "increasing" | "decreasing"}`` physics constraints
    target_weights {target: weight} relative importance in the loss
    config / **options   :class:`NNConfig` hyper-parameters
    """

    def __init__(self, features: Sequence[str], targets: Sequence[str],
                 time_column: Optional[str] = "time_days", case_column: str = "case_id",
                 log_targets: Union[str, Sequence[str]] = "auto",
                 log_features: Union[str, Sequence[str]] = "auto",
                 monotonic: Optional[dict] = None, target_weights: Optional[dict] = None,
                 config: Optional[NNConfig] = None, **options):
        self.features = list(features)
        self.targets = list(targets)
        if not self.targets:
            raise ValueError("at least one target is required")
        if time_column in self.features:
            raise ValueError(f"time column '{time_column}' must not be a feature")
        self.time_column, self.case_column = time_column, case_column
        self.log_targets_spec, self.log_features_spec = log_targets, log_features
        if config is not None and options:
            config = NNConfig.from_dict({**asdict(config), **options})
        self.cfg = config or NNConfig.from_dict(options)
        self.monotonic = dict(monotonic or {})
        for t, d in self.monotonic.items():
            if t not in self.targets:
                raise ValueError(f"monotonic constraint on unknown target '{t}'")
            if d not in ("increasing", "decreasing"):
                raise ValueError(f"monotonic['{t}'] must be 'increasing' or 'decreasing'")
        if self.monotonic and time_column is None:
            raise ValueError("monotonic constraints need a time column")
        self.target_weights = dict(target_weights or {})
        self.ftrans, self.ttrans, self.tenc = {}, {}, None
        self.nets, self.history = [], []
        self.sigma_scale = np.ones(len(self.targets))
        self.sigma_floor = np.zeros(len(self.targets))
        self.domain = {}
        self.target_range = {}
        self.train_info = {}
        self.metrics = {}
        self.correction = None          # additive discrepancy correction (fit_correction)

    # ------------------------------------------------------------- transforms
    @property
    def fitted(self) -> bool:
        return bool(self.nets)

    @property
    def n_inputs(self) -> int:
        return len(self.features) + (len(self.tenc.names) if self.tenc is not None else 0)

    def _fit_transforms(self, data: BurnupDataset):
        df = data.df
        lf = _resolve(self.log_features_spec, self.features,
                      lambda n: _auto_log_feature(df[n].to_numpy(float)))
        self.ftrans = {f: ColumnTransform.fit(df[f].to_numpy(float), lf[f]) for f in self.features}
        lt = _resolve(self.log_targets_spec, self.targets,
                      lambda n: _auto_log_target(df, n, data.case_column))
        self.ttrans = {y: ColumnTransform.fit(df[y].to_numpy(float), lt[y]) for y in self.targets}
        if self.time_column is not None:
            self.tenc = TimeEncoding(self.cfg.time_encoding, n_scales=self.cfg.n_time_scales).fit(
                df[self.time_column].to_numpy(float))
        self.domain = {f: [float(df[f].min()), float(df[f].max())] for f in self.features}
        self.target_range = {y: [float(df[y].min()), float(df[y].max())] for y in self.targets}
        if self.time_column is not None:
            self.domain[self.time_column] = [float(df[self.time_column].min()),
                                             float(df[self.time_column].max())]

    def _inputs(self, df: pd.DataFrame, t: Optional[np.ndarray] = None) -> np.ndarray:
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise DataError(f"input columns missing: {missing}")
        cols = [self.ftrans[f].forward(df[f].to_numpy(float)) for f in self.features]
        if self.tenc is not None:
            if t is None:
                if self.time_column not in df.columns:
                    raise DataError(f"time column '{self.time_column}' missing")
                t = df[self.time_column].to_numpy(float)
            cols.append(self.tenc.transform(t))
        X = np.column_stack(cols) if cols else np.zeros((len(df), 0))
        if not np.all(np.isfinite(X)):
            raise DataError("non-finite inputs (blank feature cells?)")
        return X

    def _outputs(self, df: pd.DataFrame):
        Y = np.column_stack([self.ttrans[y].forward(df[y].to_numpy(float))
                             if y in df.columns else np.full(len(df), np.nan)
                             for y in self.targets])
        M = np.isfinite(Y)
        return np.where(M, Y, 0.0), M.astype(float)

    def _target_weight_vector(self) -> np.ndarray:
        return np.array([float(self.target_weights.get(y, 1.0)) for y in self.targets])

    def _collocation(self, data: BurnupDataset):
        if not self.monotonic or self.cfg.monotonic_weight <= 0:
            return None
        rng = np.random.default_rng(self.cfg.seed + 7)
        df = data.df
        n = int(min(self.cfg.n_collocation, max(256, len(df))))  # never more than the data
        rows = rng.integers(0, len(df), n)
        sub = df.iloc[rows]
        tmax = self.tenc.t_max
        top = tmax / 1.02
        # 10 % at t = 0 (the first step after the initial state), the rest
        # uniform or log-uniform in time so every time scale is covered
        u = rng.random(n)
        t = np.where(u < 0.1, 0.0, np.where(u < 0.55, rng.uniform(0.0, top, n),
                                            top * 10.0 ** rng.uniform(-5.0, 0.0, n)))
        dt = np.maximum(0.02 * t, 1e-5 * tmax)            # relative (scale-free) steps
        X0 = self._inputs(sub, t)
        X1 = self._inputs(sub, t + dt)
        mi = np.array([self.targets.index(k) for k in self.monotonic])
        ms = np.array([1.0 if v == "increasing" else -1.0 for v in self.monotonic.values()])
        return X0, X1, np.maximum(dt / tmax, 1e-3), mi, ms

    def _optimizer(self, n_rows: int) -> str:
        opt = self.cfg.optimizer
        if opt == "auto":
            return "lbfgs" if n_rows <= 60000 else "adam"
        if opt not in ("lbfgs", "adam"):
            raise ValueError("optimizer must be 'lbfgs', 'adam' or 'auto'")
        return opt

    def _n_jobs(self, n_members: int) -> int:
        nj = self.cfg.n_jobs or (os.cpu_count() or 1)
        return max(1, min(nj, n_members))

    # ---------------------------------------------------------------- training
    def fit(self, train: BurnupDataset, val: Optional[BurnupDataset] = None,
            verbose: bool = True) -> "BurnupNN":
        """Train the ensemble (validation cases drive early stopping and the
        uncertainty calibration)."""
        train.check(self.features, self.targets)
        if self.time_column is not None and train.time_column != self.time_column:
            raise DataError(f"data set time column '{train.time_column}' != model "
                            f"'{self.time_column}'")
        self._fit_transforms(train)
        cfg = self.cfg
        X = self._inputs(train.df)
        Y, M = self._outputs(train.df)
        W = train.sample_weight()
        opt = self._optimizer(len(X))
        Xv = Yv = Mv = Wv = None
        if val is not None and len(val):
            Xv = self._inputs(val.df)
            Yv, Mv = self._outputs(val.df)
            Wv = val.sample_weight()
        colloc = self._collocation(train)
        tw = self._target_weight_vector()
        sizes = [X.shape[1], *cfg.hidden, len(self.targets)]
        state = {"sizes": sizes, "activation": cfg.activation, "residual": cfg.residual}
        cfgd = {**asdict(cfg), "optimizer": opt}
        case_ids = train.df[train.case_column].to_numpy()
        payloads = []
        for m in range(cfg.ensemble):
            seed = cfg.seed + 1000 * m
            Wm = W
            if cfg.bootstrap and m > 0:
                rng = np.random.default_rng(seed)
                cases = train.cases
                counts = pd.Series(rng.choice(cases, len(cases), replace=True)).value_counts()
                Wm = W * pd.Series(case_ids).map(counts).fillna(0.0).to_numpy(float)
            payloads.append({"cfg": cfgd, "state": state, "seed": seed, "X": X, "Y": Y, "M": M,
                             "W": Wm, "tw": tw, "Xv": Xv, "Yv": Yv, "Mv": Mv, "Wv": Wv,
                             "colloc": colloc})
        if verbose:
            print(f"[BurnupNN] {len(self.targets)} targets <- {len(self.features)} features"
                  f"{' + time' if self.tenc is not None else ''}; {train.n_cases} cases / "
                  f"{len(X)} rows; net {sizes}, {cfg.activation}, {opt}, ensemble {cfg.ensemble}"
                  + (f", monotonic {list(self.monotonic)}" if colloc is not None else ""))
        out = _run_members(payloads, self._n_jobs(cfg.ensemble), verbose)
        self.nets = [MLP.from_state(state, params) for params, _ in out]
        self.history = [dict(h, phase="train", member=i) for i, (_, h) in enumerate(out)]
        self.sigma_scale = np.ones(len(self.targets))
        self.sigma_floor = np.zeros(len(self.targets))
        self.train_info = {"n_cases": int(train.n_cases), "n_rows": int(len(X)),
                           "sources": train.sources, "optimizer": opt,
                           "n_params": int(self.nets[0].n_params),
                           "wall_s": float(sum(h["wall_s"] for _, h in out)),
                           "log_targets": [y for y in self.targets if self.ttrans[y].log],
                           "log_features": [f for f in self.features if self.ftrans[f].log]}
        if val is not None and len(val):
            self.calibrate(val)
        return self

    def fine_tune(self, data: BurnupDataset, val: Optional[BurnupDataset] = None,
                  max_iter: int = 300, anchor: float = 1e-3, freeze: int = 0,
                  epochs: Optional[int] = None, lr: Optional[float] = None,
                  verbose: bool = True) -> "BurnupNN":
        """Transfer learning: continue training every member on new data.

        The input/output transforms of the pre-trained model are kept; the
        weights start from the pre-trained values and are pulled back towards
        them with strength ``anchor`` (L2-SP).  ``freeze`` keeps the first
        ``freeze`` layers fixed.  Targets absent from the new data (blank)
        keep their pre-trained behaviour through the anchor.

        Weight fine-tuning needs a fair number of runs (roughly ten or more);
        with only a few runs prefer :meth:`fit_correction`, which cannot make
        the cross-validated error worse.  An existing correction is removed.
        """
        if not self.fitted:
            raise RuntimeError("fit (or load) a model before fine-tuning")
        self.correction = None
        missing = [f for f in self.features if f not in data.df.columns]
        if missing:
            raise DataError(f"fine-tuning data lacks feature columns {missing}")
        present = [y for y in self.targets if y in data.df.columns]
        if not present:
            raise DataError("fine-tuning data contains none of the model targets")
        data.check(self.features, present)
        X = self._inputs(data.df)
        Y, M = self._outputs(data.df)
        W = data.sample_weight()
        Xv = Yv = Mv = Wv = None
        if val is not None and len(val):
            Xv = self._inputs(val.df)
            Yv, Mv = self._outputs(val.df)
            Wv = val.sample_weight()
        opt = self._optimizer(len(X))
        cfgd = {**asdict(self.cfg), "optimizer": opt}
        state = self.nets[0].state()
        frozen = tuple(range(2 * int(freeze)))
        colloc = None
        if self.monotonic and self.cfg.monotonic_weight > 0:
            colloc = self._collocation(data)
        payloads = []
        for m, net in enumerate(self.nets):
            p = {"cfg": cfgd, "state": state, "seed": self.cfg.seed + 1000 * m + 1,
                 "init": [q.copy() for q in net.params], "anchor": [q.copy() for q in net.params],
                 "anchor_w": float(anchor), "frozen": frozen, "X": X, "Y": Y, "M": M, "W": W,
                 "tw": self._target_weight_vector(), "Xv": Xv, "Yv": Yv, "Mv": Mv, "Wv": Wv,
                 "colloc": colloc, "max_iter": int(max_iter)}
            if epochs is not None:
                p["epochs"] = int(epochs)
            if lr is not None:
                p["lr"] = float(lr)
            payloads.append(p)
        if verbose:
            print(f"[BurnupNN] fine-tuning {len(self.nets)} member(s) on {data.n_cases} cases / "
                  f"{len(X)} rows (anchor {anchor:g}, frozen layers {freeze}, {opt})")
        out = _run_members(payloads, self._n_jobs(len(self.nets)), verbose)
        self.nets = [MLP.from_state(state, params) for params, _ in out]
        self.history += [dict(h, phase="fine-tune", member=i) for i, (_, h) in enumerate(out)]
        self.train_info.setdefault("fine_tune", []).append(
            {"n_cases": int(data.n_cases), "n_rows": int(len(X)), "sources": data.sources,
             "anchor": float(anchor), "freeze": int(freeze), "max_iter": int(max_iter)})
        if val is not None and len(val):
            self.calibrate(val)
        return self

    # ------------------------------------------------- discrepancy correction
    def _basis(self, X: np.ndarray, interactions: bool) -> np.ndarray:
        """[1, inputs, feature x time-encoding products] for the correction."""
        nf = len(self.features)
        cols = [np.ones((len(X), 1)), X]
        if interactions and X.shape[1] > nf and nf > 0:
            F, T = X[:, :nf], X[:, nf:]
            cols.append((F[:, :, None] * T[:, None, :]).reshape(len(X), -1))
        return np.hstack(cols)

    def fit_correction(self, data: BurnupDataset, interactions: bool = True,
                       alphas: Optional[Sequence[float]] = None,
                       verbose: bool = True) -> pd.DataFrame:
        """Additive discrepancy correction from (a few) runs of another code.

        Learns delta(x, t) = new data - network for every target as a ridge
        regression on the network inputs (design features, time encoding and,
        with ``interactions``, their products) plus the network's own output
        (so a multiplicative bias between codes is captured), in standardized
        target space - the classical multi-fidelity "learn only the difference"
        approach.  The
        ridge penalty is chosen per target by **leave-one-case-out**
        cross-validation; a target is left uncorrected unless the correction
        lowers its cross-validated error, so the correction cannot make the
        cross-validated error worse.  It is applied on top of every ensemble
        member and saved with the model.  Returns a per-target summary.
        """
        self._require_fitted()
        missing = [f for f in self.features if f not in data.df.columns]
        if missing:
            raise DataError(f"correction data lacks feature columns {missing}")
        alphas = np.logspace(-4, 4, 17) if alphas is None else np.asarray(alphas, float)
        df = data.df
        X = self._inputs(df)
        Z = np.mean([net.forward(X) for net in self.nets], axis=0)
        Y, M = self._outputs(df)
        Phi = self._basis(X, interactions)
        cases = df[data.case_column].to_numpy()
        W = np.zeros((Phi.shape[1] + 1, len(self.targets)))
        rows, alpha_used = [], []
        eye = np.eye(Phi.shape[1] + 1)
        for j, y in enumerate(self.targets):
            ok = M[:, j] > 0
            r, cs = (Y[:, j] - Z[:, j])[ok], cases[ok]
            P = np.column_stack([Phi[ok], Z[ok, j]])
            uc = np.unique(cs)
            before = float(np.sqrt(np.mean(r ** 2))) if len(r) else float("nan")
            best_err, best_a = (np.mean(r ** 2) if len(r) else np.inf), None
            if len(uc) >= 2:
                for a in alphas:
                    err = 0.0
                    for c in uc:
                        tr = cs != c
                        w = np.linalg.solve(P[tr].T @ P[tr] + a * eye, P[tr].T @ r[tr])
                        err += float(np.sum((P[~tr] @ w - r[~tr]) ** 2))
                    err /= len(r)
                    if err < best_err:
                        best_err, best_a = err, float(a)
            if best_a is not None:
                W[:, j] = np.linalg.solve(P.T @ P + best_a * eye, P.T @ r)
            alpha_used.append(best_a)
            rows.append({"target": y, "points": int(ok.sum()), "corrected": best_a is not None,
                         "ridge_alpha": best_a, "cv_rmse_before_std": before,
                         "cv_rmse_after_std": float(np.sqrt(best_err)) if len(r) else float("nan")})
        self.correction = {"W": W, "interactions": bool(interactions), "alpha": alpha_used,
                           "n_cases": int(data.n_cases), "sources": data.sources}
        summary = pd.DataFrame(rows).set_index("target")
        if verbose:
            n_corr = int(summary["corrected"].sum())
            print(f"[BurnupNN] discrepancy correction from {data.n_cases} cases / {len(df)} rows: "
                  f"{n_corr}/{len(self.targets)} targets corrected (leave-one-case-out CV)")
        return summary

    def clear_correction(self) -> "BurnupNN":
        self.correction = None
        return self

    # -------------------------------------------------------------- prediction
    def _member_predictions(self, X: np.ndarray) -> np.ndarray:
        """(members, rows, targets) predictions in physical units.

        Quantities that were never negative in the training data (inventories,
        burnup, poison fractions, k) are clipped at zero - a physics guard that
        matters near t = 0 where many of them start from exactly zero."""
        Z = np.stack([net.forward(X) for net in self.nets])
        if self.correction is not None:
            W = self.correction["W"]                  # last row scales the network output
            Z = Z + (self._basis(X, self.correction["interactions"]) @ W[:-1])[None] + Z * W[-1]
        out = np.empty_like(Z)
        for j, y in enumerate(self.targets):
            out[:, :, j] = self.ttrans[y].inverse(Z[:, :, j])
            if self.nonnegative(y):
                np.maximum(out[:, :, j], 0.0, out=out[:, :, j])
        return out

    def _stats(self, P: np.ndarray, raw: bool = False):
        mean = P.mean(axis=0)
        if P.shape[0] > 1:
            std = P.std(axis=0, ddof=1)
            if not raw:
                std = std * self.sigma_scale
        else:
            std = np.zeros_like(mean)
        if not raw:
            std = np.sqrt(std ** 2 + self.sigma_floor ** 2)
        return mean, std

    def _frame(self, df):
        if isinstance(df, BurnupDataset):
            return df.df
        if isinstance(df, dict):
            return pd.DataFrame(df)
        return df

    def predict(self, data, return_std: bool = True, raw_std: bool = False) -> pd.DataFrame:
        """Predict every target for the rows of a data set / DataFrame.

        Returns a DataFrame with columns ``<target>`` (ensemble mean) and
        ``<target>_std`` (calibrated 1-sigma), index aligned with the input.
        """
        self._require_fitted()
        df = self._frame(data)
        P = self._member_predictions(self._inputs(df))
        mean, std = self._stats(P, raw_std)
        out = pd.DataFrame(mean, columns=self.targets, index=df.index)
        if return_std:
            for j, y in enumerate(self.targets):
                out[f"{y}_std"] = std[:, j]
        return out

    def predict_members(self, data) -> np.ndarray:
        """Raw member predictions, shape (members, rows, targets)."""
        self._require_fitted()
        return self._member_predictions(self._inputs(self._frame(data)))

    def time_grid(self, n_times: int = 200, t_max: Optional[float] = None) -> np.ndarray:
        """Dense time grid (log-refined near t=0 to resolve the fast transients)."""
        t_max = t_max or (self.tenc.t_max if self.tenc is not None else 1.0)
        n_log = n_times // 3
        g = np.unique(np.concatenate([[0.0], t_max * np.logspace(-4, -1, n_log),
                                      np.linspace(0.0, t_max, n_times - n_log)]))
        return g

    def predict_cases(self, designs: pd.DataFrame, times=None, n_times: int = 200,
                      case_column: Optional[str] = None) -> pd.DataFrame:
        """Trajectories for many designs: ``designs`` has one row per design
        (feature columns); returns a long table case x time with mean/std."""
        self._require_fitted()
        if self.tenc is None:
            return pd.concat([designs.reset_index(drop=True),
                              self.predict(designs.reset_index(drop=True))], axis=1)
        times = self.time_grid(n_times) if times is None else np.asarray(times, float)
        designs = designs.reset_index(drop=True)
        cc = case_column or self.case_column
        ids = designs[cc].astype(str).to_numpy() if cc in designs.columns else \
            np.array([f"design_{i}" for i in range(len(designs))])
        rep = designs.loc[np.repeat(np.arange(len(designs)), len(times)), self.features]
        rep = rep.reset_index(drop=True)
        rep.insert(0, self.time_column, np.tile(times, len(designs)))
        rep.insert(0, cc, np.repeat(ids, len(times)))
        pred = self.predict(rep)
        return pd.concat([rep, pred], axis=1)

    def predict_trajectory(self, features: dict, times=None, n_times: int = 200) -> pd.DataFrame:
        """Trajectory of one design given as ``{feature: value}``."""
        row = pd.DataFrame([{f: features[f] for f in self.features}])
        return self.predict_cases(row, times, n_times).drop(columns=[self.case_column])

    def nonnegative(self, target: str) -> bool:
        """True if the target never was negative in the training data."""
        r = self.target_range.get(target)
        return bool(r is not None and r[0] >= 0.0)

    def in_domain(self, data, margin: float = 0.0) -> np.ndarray:
        """True for rows whose inputs lie inside the training range (+ margin
        as a fraction of the range) - outside, predictions are extrapolations."""
        df = self._frame(data)
        ok = np.ones(len(df), bool)
        for c, (lo, hi) in self.domain.items():
            if c not in df.columns:
                continue
            span = (hi - lo) or abs(hi) or 1.0
            x = df[c].to_numpy(float)
            ok &= (x >= lo - margin * span) & (x <= hi + margin * span)
        return ok

    # ------------------------------------------------------- accuracy and UQ
    def calibrate(self, val: BurnupDataset):
        """Scale the ensemble spread so that 95 % intervals cover 95 % of the
        validation errors (per target); single networks get a constant sigma."""
        df = val.df
        P = self._member_predictions(self._inputs(df))
        mean, std = self._stats(P, raw=True)
        self.sigma_scale = np.ones(len(self.targets))
        self.sigma_floor = np.zeros(len(self.targets))
        for j, y in enumerate(self.targets):
            if y not in df.columns:
                continue
            obs = df[y].to_numpy(float)
            ok = np.isfinite(obs)
            if ok.sum() < 5:
                continue
            err = np.abs(obs[ok] - mean[ok, j])
            if P.shape[0] > 1 and np.all(std[ok, j] > 0):
                r = np.quantile(err / std[ok, j], 0.95) / 1.96
                self.sigma_scale[j] = float(np.clip(r, 0.25, 100.0))
            else:
                self.sigma_floor[j] = float(np.sqrt(np.mean(err ** 2)))
        return self

    def evaluate(self, data: BurnupDataset, return_predictions: bool = False):
        """Accuracy per target on a data set (physical units).

        Columns: rmse, mae, max_abs, bias, r2, nrmse_pct (RMSE / observed range),
        rel_rmse_pct (RMSE / mean |y|), coverage95 (fraction inside the 95 %
        interval), n; k-like targets also get the RMSE in pcm.
        """
        self._require_fitted()
        df = data.df
        pred = self.predict(df)
        rows = {}
        for y in self.targets:
            if y not in df.columns:
                continue
            obs = df[y].to_numpy(float)
            ok = np.isfinite(obs)
            if not ok.any():
                continue
            o, p, s = obs[ok], pred[y].to_numpy()[ok], pred[f"{y}_std"].to_numpy()[ok]
            e = p - o
            ss_tot = float(((o - o.mean()) ** 2).sum()) or 1e-300
            rng_ = float(o.max() - o.min()) or 1e-300
            r = {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
                 "max_abs": float(np.max(np.abs(e))), "bias": float(np.mean(e)),
                 "r2": float(1.0 - (e ** 2).sum() / ss_tot),
                 "nrmse_pct": float(100.0 * np.sqrt(np.mean(e ** 2)) / rng_),
                 "rel_rmse_pct": float(100.0 * np.sqrt(np.mean(e ** 2)) /
                                       max(np.mean(np.abs(o)), 1e-300)),
                 "coverage95": float(np.mean(np.abs(e) <= 1.96 * s)) if np.any(s > 0) else float("nan"),
                 "n": int(ok.sum())}
            if is_pcm_target(y):
                r["rmse_pcm"] = 1e5 * r["rmse"]
                r["max_abs_pcm"] = 1e5 * r["max_abs"]
            rows[y] = r
        met = pd.DataFrame(rows).T
        if return_predictions:
            return met, pd.concat([df.reset_index(drop=True), pred.add_prefix("pred_").reset_index(drop=True)], axis=1)
        return met

    def sensitivity(self, data, target: str, times=None) -> pd.DataFrame:
        """Change of ``target`` per one standard deviation of each (log-)feature
        of the training data, from analytic input gradients (ensemble mean),
        averaged over the rows of ``data``.

        If ``data`` has no time column (one row per design) - or ``times`` is
        given - every design is evaluated on ``times`` (default: 40 points
        over the training span), i.e. the sensitivity averaged over the
        trajectory; pass e.g. ``times=[1500]`` for one point in time."""
        self._require_fitted()
        df = self._frame(data).reset_index(drop=True)
        if self.tenc is not None and (times is not None or self.time_column not in df.columns):
            times = (np.linspace(0.0, self.tenc.t_max, 40) if times is None
                     else np.atleast_1d(np.asarray(times, float)))
            df = df.loc[np.repeat(np.arange(len(df)), len(times)), self.features].reset_index(drop=True)
            df[self.time_column] = np.tile(times, len(df) // len(times))
        X = self._inputs(df)
        j = self.targets.index(target)
        tt = self.ttrans[target]
        nf = len(self.features)
        G = []
        for net in self.nets:
            Z, cache = net.forward(X, keep=True)
            dY = np.zeros_like(Z)
            dY[:, j] = 1.0
            _, dX = net.backward(cache, dY, input_grad=True)
            G.append(dX[:, :nf] * tt.dinverse(Z[:, j])[:, None])
        G = np.mean(G, axis=0)
        return pd.DataFrame({"feature": self.features, "mean_abs": np.abs(G).mean(axis=0),
                             "mean_signed": G.mean(axis=0),
                             "log_scaled": [self.ftrans[f].log for f in self.features]}
                            ).sort_values("mean_abs", ascending=False).reset_index(drop=True)

    # ----------------------------------------------------- burnup utilities
    def crossing(self, designs: Union[dict, pd.DataFrame], target: str = "k_eff",
                 level: float = 1.0, direction: str = "down", times=None, n_times: int = 400,
                 burnup_target: Optional[str] = None) -> pd.DataFrame:
        """First time at which ``target`` crosses ``level`` (e.g. k_eff = 1:
        end of an un-refuelled cycle).  Evaluated for every ensemble member, so
        each design gets a mean and a 1-sigma spread; ``censored`` marks
        members that never cross within the time grid.  With ``burnup_target``
        the predicted burnup at the crossing is returned as well (discharge
        burnup).  Designs already past the level at t=0 get time 0.
        """
        self._require_fitted()
        if self.tenc is None:
            raise RuntimeError("crossing() needs a time-dependent model")
        if isinstance(designs, dict):
            designs = pd.DataFrame([designs])
        designs = designs.reset_index(drop=True)
        times = self.time_grid(n_times) if times is None else np.unique(np.asarray(times, float))
        nd, nt = len(designs), len(times)
        rep = designs.loc[np.repeat(np.arange(nd), nt), self.features].reset_index(drop=True)
        X = self._inputs(rep, np.tile(times, nd))
        P = self._member_predictions(X)                       # (m, nd*nt, T)
        y = P[:, :, self.targets.index(target)].reshape(len(self.nets), nd, nt)
        sgn = 1.0 if direction == "down" else -1.0
        g = sgn * (y - level)                                 # >0 before the crossing
        tc = np.full(y.shape[:2], np.nan)
        for m in range(y.shape[0]):
            for d in range(nd):
                gd = g[m, d]
                if gd[0] <= 0:
                    tc[m, d] = 0.0
                    continue
                k = np.flatnonzero(gd <= 0)
                if len(k):
                    i = k[0]
                    f = gd[i - 1] / (gd[i - 1] - gd[i])
                    tc[m, d] = times[i - 1] + f * (times[i] - times[i - 1])
        out = pd.DataFrame({"time": _nan_mean(tc), "time_std": _nan_std(tc),
                            "censored": np.isnan(tc).mean(axis=0)})
        out.loc[out["censored"] > 0.5, ["time", "time_std"]] = np.nan
        if burnup_target is not None:
            b = P[:, :, self.targets.index(burnup_target)].reshape(len(self.nets), nd, nt)
            bc = np.full_like(tc, np.nan)
            for m in range(tc.shape[0]):
                for d in range(nd):
                    if np.isfinite(tc[m, d]):
                        bc[m, d] = np.interp(tc[m, d], times, b[m, d])
            out["burnup"] = _nan_mean(bc)
            out["burnup_std"] = _nan_std(bc)
            out.loc[out["censored"] > 0.5, ["burnup", "burnup_std"]] = np.nan
        return pd.concat([designs[self.features], out], axis=1)

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> str:
        """Write ``model.json`` (everything but weights) + ``weights.npz``."""
        self._require_fitted()
        os.makedirs(path, exist_ok=True)
        meta = {"format": FORMAT_VERSION, "framework": "msrnx.burnup_nn",
                "features": self.features, "targets": self.targets,
                "time_column": self.time_column, "case_column": self.case_column,
                "log_targets": self.log_targets_spec, "log_features": self.log_features_spec,
                "config": {**asdict(self.cfg), "hidden": list(self.cfg.hidden)},
                "monotonic": self.monotonic, "target_weights": self.target_weights,
                "feature_transforms": {f: t.to_dict() for f, t in self.ftrans.items()},
                "target_transforms": {y: t.to_dict() for y, t in self.ttrans.items()},
                "time_encoding": self.tenc.to_dict() if self.tenc is not None else None,
                "sigma_scale": self.sigma_scale.tolist(), "sigma_floor": self.sigma_floor.tolist(),
                "domain": self.domain, "target_range": self.target_range,
                "network": self.nets[0].state(),
                "n_members": len(self.nets), "train_info": self.train_info,
                "metrics": self.metrics, "history": self.history,
                "correction": None if self.correction is None else
                {**self.correction, "W": self.correction["W"].tolist()}}
        with open(os.path.join(path, "model.json"), "w") as f:
            json.dump(meta, f, indent=1, default=_json_default)
        arrays = {f"m{m}_p{k}": p for m, net in enumerate(self.nets) for k, p in enumerate(net.params)}
        np.savez_compressed(os.path.join(path, "weights.npz"), **arrays)
        return path

    @classmethod
    def load(cls, path: str) -> "BurnupNN":
        with open(os.path.join(path, "model.json")) as f:
            meta = json.load(f)
        if meta.get("framework") != "msrnx.burnup_nn":
            raise ValueError(f"{path} is not a msrnx.burnup_nn model")
        cfg = NNConfig.from_dict(meta["config"])
        model = cls(meta["features"], meta["targets"], meta["time_column"], meta["case_column"],
                    meta.get("log_targets", "auto"), meta.get("log_features", "auto"),
                    meta.get("monotonic"), meta.get("target_weights"), config=cfg)
        model.ftrans = {f: ColumnTransform.from_dict(d) for f, d in meta["feature_transforms"].items()}
        model.ttrans = {y: ColumnTransform.from_dict(d) for y, d in meta["target_transforms"].items()}
        model.tenc = TimeEncoding.from_dict(meta["time_encoding"]) if meta["time_encoding"] else None
        model.sigma_scale = np.array(meta["sigma_scale"], float)
        model.sigma_floor = np.array(meta.get("sigma_floor", [0.0] * len(model.targets)), float)
        model.domain = meta.get("domain", {})
        model.target_range = meta.get("target_range", {})
        model.train_info = meta.get("train_info", {})
        model.metrics = meta.get("metrics", {})
        model.history = meta.get("history", [])
        if meta.get("correction"):
            model.correction = {**meta["correction"],
                                "W": np.array(meta["correction"]["W"], float)}
        with np.load(os.path.join(path, "weights.npz")) as w:
            n_par = 2 * (len(meta["network"]["sizes"]) - 1)
            model.nets = [MLP.from_state(meta["network"], [w[f"m{m}_p{k}"] for k in range(n_par)])
                          for m in range(meta["n_members"])]
        return model

    # ---------------------------------------------------------------- misc
    def _require_fitted(self):
        if not self.fitted:
            raise RuntimeError("the model is not trained yet - call fit() or BurnupNN.load()")

    def describe(self) -> str:
        lines = [f"BurnupNN: {len(self.targets)} targets, {len(self.features)} features"
                 f"{', time: ' + self.time_column if self.time_column else ''}"]
        if self.fitted:
            lines.append(f"  network {self.nets[0].sizes} ({self.cfg.activation}), "
                         f"{len(self.nets)} members, {self.nets[0].n_params} parameters each")
            lines.append(f"  trained on {self.train_info.get('n_cases')} cases / "
                         f"{self.train_info.get('n_rows')} rows from {self.train_info.get('sources')}")
            if self.train_info.get("log_targets"):
                lines.append(f"  log-space targets: {self.train_info['log_targets']}")
            if self.monotonic:
                lines.append(f"  monotonic: {self.monotonic}")
            if self.correction is not None:
                lines.append(f"  discrepancy correction from {self.correction['n_cases']} cases "
                             f"({self.correction['sources']})")
        return "\n".join(lines)


def _curve_worker(args):
    features, targets, opts, train, val, test, ids, n = args
    m = BurnupNN(features, targets, **opts)
    m.fit(train.subset(ids[:n]), val, verbose=False)
    return m.evaluate(test)


def learning_curve(features: Sequence[str], targets: Sequence[str], train: BurnupDataset,
                   val: Optional[BurnupDataset], test: BurnupDataset, sizes: Sequence[int],
                   variants: Optional[dict] = None, seed: int = 0,
                   n_jobs: Optional[int] = None, **options) -> pd.DataFrame:
    """Test accuracy vs number of training cases - "how many more runs do I need?".

    One network per point (``ensemble=1`` unless overridden), trained on the
    first ``n`` cases of a fixed random order, so larger sets contain the
    smaller ones.  ``variants`` = ``{label: {option overrides}}`` compares
    settings (e.g. time encodings).  Returns a long table: variant, n_cases,
    target and the metrics of :meth:`BurnupNN.evaluate`.
    """
    variants = variants or {"default": {}}
    ids = np.random.default_rng(seed).permutation(train.cases)
    jobs, keys = [], []
    for label, over in variants.items():
        for n in sizes:
            n = int(min(n, len(ids)))
            opts = {"ensemble": 1, "n_jobs": 1, **options, **over}
            jobs.append((list(features), list(targets), opts, train, val, test, ids, n))
            keys.append((label, n))
    n_jobs = max(1, min(len(jobs), n_jobs or os.cpu_count() or 1))
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            mets = list(ex.map(_curve_worker, jobs))
    else:
        mets = [_curve_worker(j) for j in jobs]
    rows = []
    for (label, n), met in zip(keys, mets):
        for t, r in met.iterrows():
            rows.append({"variant": label, "n_cases": n, "target": t, **r.to_dict()})
    return pd.DataFrame(rows)


def _nan_mean(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=0)


def _nan_std(a: np.ndarray) -> np.ndarray:
    if a.shape[0] < 2:
        return np.zeros(a.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=0, ddof=1)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def load_model(path: str) -> BurnupNN:
    return BurnupNN.load(path)
