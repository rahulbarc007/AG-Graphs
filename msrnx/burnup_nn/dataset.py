"""Burnup data container - the single entry point for *your* depletion data.

Format ("long" table, one row per case and time point)::

    case_id, time_days, <feature columns ...>, <target columns ...>
    run_01,  0.0,       0.35, 10.0, 0.18, ...,   1.3248, 1141.1, 0.0, ...
    run_01,  30.0,      0.35, 10.0, 0.18, ...,   1.3037, 1112.3, 9.45, ...

* ``case_id``   identifies one depletion trajectory (one design / one run).
* time column   time [days] - or burnup [MWd/kgHM], or any monotone burnup
                variable (set ``time_column``).  ``time_column=None`` turns the
                framework into a plain static design surrogate.
* features      inputs of the network: design / operating parameters (constant
                within a case, or time dependent such as a power history).
* targets       outputs to learn: k_eff, inventories, breeding ratio, ...
                Blank cells (NaN) are allowed - e.g. when your code does not
                report a nuclide; the loss simply skips missing entries.
* optional      ``source`` (data origin, set automatically) and ``weight``
                (per-row training weight).

Files from different codes (MSR-NX, Serpent, SCALE, OpenMC, in-house tools) are
loaded with :meth:`BurnupDataset.load` (CSV, TSV, Excel, JSON, Parquet, NPZ),
harmonized with ``rename`` / ``scale`` / ``constants`` and merged with ``+``.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

SOURCE = "source"
WEIGHT = "weight"


class DataError(ValueError):
    """Raised when a data set cannot be used for training."""


class BurnupDataset:
    """Long-format burnup table (see module docstring)."""

    def __init__(self, df: pd.DataFrame, case_column: str = "case_id",
                 time_column: Optional[str] = "time_days", source: Optional[str] = None):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df = df.copy()
        if case_column not in df.columns:
            if time_column is None:            # static data: one case per row
                df[case_column] = [f"row_{i}" for i in range(len(df))]
            else:
                raise DataError(f"case column '{case_column}' not found; columns are "
                                f"{list(df.columns)[:12]}...")
        if time_column is not None and time_column not in df.columns:
            raise DataError(f"time column '{time_column}' not found; columns are "
                            f"{list(df.columns)[:12]}...")
        df[case_column] = df[case_column].astype(str)
        if source is not None:
            df[SOURCE] = str(source)
        elif SOURCE not in df.columns:
            df[SOURCE] = "data"
        df[SOURCE] = df[SOURCE].astype(str)
        sort = [case_column] + ([time_column] if time_column else [])
        order = pd.Index(pd.unique(df[case_column]))
        df["_order"] = order.get_indexer(df[case_column])
        df = df.sort_values(["_order"] + sort[1:], kind="stable").drop(columns="_order")
        self.df = df.reset_index(drop=True)
        self.case_column = case_column
        self.time_column = time_column

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str, case_column: str = "case_id",
             time_column: Optional[str] = "time_days", rename: Optional[dict] = None,
             scale: Optional[dict] = None, constants: Optional[dict] = None,
             sheet=None, query: Optional[str] = None, source: Optional[str] = None,
             **read_kw) -> "BurnupDataset":
        """Read a CSV / TSV (optionally .gz/.bz2/.xz/.zip) / Excel / JSON /
        Parquet / NPZ file.

        rename     {column in file: framework column}, applied first
        scale      {column: factor} unit conversion after renaming
                   (e.g. ``{"U233_kg": 1e-3}`` if the file is in grams)
        constants  {column: value} columns to add with a constant value, e.g. a
                   feature your runs did not vary (``{"T_fuel": 900.0}``)
        query      pandas query string to select rows, e.g. ``"time_days <= 1500"``
        source     label of the data origin (default: file name)
        """
        if not os.path.exists(path):
            raise DataError(f"data file not found: {path}")
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext in (".gz", ".bz2", ".xz", ".zip"):          # compressed text tables
            ext = os.path.splitext(root)[1].lower() or ".csv"
            if ext not in (".csv", ".tsv", ".txt", ".dat"):
                raise DataError(f"compressed files must be CSV/TSV ({path})")
        if ext in (".csv", ".txt", ".dat"):
            df = pd.read_csv(path, comment=read_kw.pop("comment", "#"), **read_kw)
        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t", comment=read_kw.pop("comment", "#"), **read_kw)
        elif ext in (".xlsx", ".xlsm", ".xls"):
            df = pd.read_excel(path, sheet_name=0 if sheet is None else sheet, **read_kw)
        elif ext == ".json":
            df = pd.read_json(path, **read_kw)
        elif ext == ".parquet":
            df = pd.read_parquet(path, **read_kw)
        elif ext == ".npz":
            with np.load(path, allow_pickle=False) as z:
                df = pd.DataFrame({k: z[k] for k in z.files if z[k].ndim == 1})
        else:
            raise DataError(f"unsupported file type '{ext}' ({path}); use CSV, TSV, Excel, "
                            "JSON, Parquet or NPZ - or build a DataFrame and call "
                            "BurnupDataset(df)")
        df.columns = [str(c).strip() for c in df.columns]
        if rename:
            missing = [k for k in rename if k not in df.columns]
            if missing:
                raise DataError(f"{path}: columns to rename not found: {missing}")
            df = df.rename(columns=rename)
        for col, f in (scale or {}).items():
            if col not in df.columns:
                raise DataError(f"{path}: column to scale not found: '{col}'")
            df[col] = pd.to_numeric(df[col], errors="coerce") * float(f)
        for col, v in (constants or {}).items():
            df[col] = v
        if query:
            df = df.query(query)
        if source is None:
            base = os.path.basename(path)
            for _ in range(2 if base.lower().endswith((".gz", ".bz2", ".xz", ".zip")) else 1):
                base = os.path.splitext(base)[0]
            source = base
        return cls(df, case_column, time_column, source)

    @classmethod
    def concat(cls, datasets: Sequence["BurnupDataset"]) -> "BurnupDataset":
        """Merge data sets (union of columns; absent columns become NaN).

        Case identifiers that collide between data sets are prefixed with the
        source label, so trajectories from different codes never mix.
        """
        datasets = [d for d in datasets if d is not None]
        if not datasets:
            raise DataError("nothing to concatenate")
        c, t = datasets[0].case_column, datasets[0].time_column
        frames, seen = [], set()
        for d in datasets:
            if d.case_column != c or d.time_column != t:
                raise DataError("all data sets must use the same case/time column names "
                                f"('{c}', '{t}'); use rename= when loading")
            df = d.df.copy()
            ids = set(df[c])
            if ids & seen:
                df[c] = df[SOURCE] + ":" + df[c]
                ids = set(df[c])
                if ids & seen:
                    raise DataError("duplicate case identifiers across data sets even after "
                                    "prefixing with the source label")
            seen |= ids
            frames.append(df)
        return cls(pd.concat(frames, ignore_index=True, sort=False), c, t)

    def add(self, other: "BurnupDataset") -> "BurnupDataset":
        return BurnupDataset.concat([self, other])

    __add__ = add

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return (f"BurnupDataset({self.n_cases} cases, {len(self)} rows, "
                f"sources={self.sources}, columns={len(self.df.columns)})")

    @property
    def cases(self) -> np.ndarray:
        return pd.unique(self.df[self.case_column])

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    @property
    def sources(self) -> list:
        return list(pd.unique(self.df[SOURCE]))

    @property
    def columns(self) -> list:
        return list(self.df.columns)

    def numeric_columns(self) -> list:
        skip = {self.case_column, SOURCE}
        return [c for c in self.df.columns if c not in skip
                and pd.api.types.is_numeric_dtype(self.df[c])]

    def case(self, case_id) -> pd.DataFrame:
        return self.df[self.df[self.case_column] == str(case_id)]

    def subset(self, cases: Iterable) -> "BurnupDataset":
        keep = set(map(str, cases))
        return BurnupDataset(self.df[self.df[self.case_column].isin(keep)], self.case_column,
                             self.time_column)

    def select_sources(self, sources: Iterable[str]) -> "BurnupDataset":
        keep = set(sources)
        return BurnupDataset(self.df[self.df[SOURCE].isin(keep)], self.case_column,
                             self.time_column)

    def head_cases(self, n: int) -> "BurnupDataset":
        return self.subset(self.cases[:n])

    def time(self) -> np.ndarray:
        return self.df[self.time_column].to_numpy(float)

    def case_table(self, columns: Sequence[str]) -> pd.DataFrame:
        """First-row value of ``columns`` for every case (static design table)."""
        return self.df.groupby(self.case_column, sort=False)[list(columns)].first()

    def sample_weight(self) -> np.ndarray:
        if WEIGHT in self.df.columns:
            return self.df[WEIGHT].fillna(1.0).to_numpy(float)
        return np.ones(len(self.df))

    # ------------------------------------------------------------- splitting
    def split(self, test: float = 0.15, val: float = 0.15, seed: int = 0):
        """Random split **by case** -> (train, val, test).

        Whole trajectories are kept together so that no time point of a test
        design is ever seen in training (no leakage between neighbouring steps).
        """
        ids = np.random.default_rng(seed).permutation(self.cases)
        n = len(ids)
        n_test = int(round(test * n)) if test > 0 else 0
        n_val = int(round(val * n)) if val > 0 else 0
        if test > 0:
            n_test = max(n_test, 1)
        if val > 0:
            n_val = max(n_val, 1)
        if n - n_test - n_val < 1:
            raise DataError(f"only {n} cases: not enough for a {test:.0%}/{val:.0%} "
                            "test/validation split")
        te, va, tr = ids[:n_test], ids[n_test:n_test + n_val], ids[n_test + n_val:]
        return self.subset(tr), (self.subset(va) if n_val else None), (self.subset(te) if n_test else None)

    # ------------------------------------------------------------- checking
    def check(self, features: Sequence[str], targets: Sequence[str],
              raise_on_error: bool = True) -> list:
        """Validate the table for the given features/targets.

        Returns a list of ``(level, message)`` with level ``error``, ``warning``
        or ``info``; raises :class:`DataError` on errors unless told not to.
        """
        msgs = []
        df = self.df
        cols = set(df.columns)
        if self.time_column in features:
            msgs.append(("error", f"the time column '{self.time_column}' must not be listed "
                                  "as a feature (it is encoded automatically)"))
        for kind, names in (("feature", features), ("target", targets)):
            missing = [c for c in names if c not in cols]
            if missing:
                msgs.append(("error", f"{kind} columns not found: {missing}"))
            for c in names:
                if c in cols and not pd.api.types.is_numeric_dtype(df[c]):
                    bad = pd.to_numeric(df[c], errors="coerce").isna() & df[c].notna()
                    msgs.append(("error", f"{kind} '{c}' is not numeric ({int(bad.sum())} "
                                          f"non-numeric cells, e.g. {df[c][bad].iloc[:3].tolist()})"))
        overlap = set(features) & set(targets)
        if overlap:
            msgs.append(("error", f"columns used as both feature and target: {sorted(overlap)}"))
        if not any(m[0] == "error" for m in msgs):
            for c in features:
                nan = df[c].isna()
                if nan.any():
                    bad_cases = pd.unique(df.loc[nan, self.case_column])[:5]
                    msgs.append(("error", f"feature '{c}' has {int(nan.sum())} blank cells "
                                          f"(cases {list(bad_cases)}...); fill them or use "
                                          "constants= when loading"))
                elif df[c].nunique() == 1:
                    msgs.append(("warning", f"feature '{c}' is constant ({df[c].iloc[0]}) - "
                                            "it carries no information"))
                else:
                    varying = df.groupby(self.case_column, sort=False)[c].nunique() > 1
                    if varying.any():
                        msgs.append(("info", f"feature '{c}' varies within {int(varying.sum())} "
                                             "case(s): treated as a time-dependent input"))
            for c in targets:
                frac = df[c].isna().mean()
                if frac == 1.0:
                    msgs.append(("error", f"target '{c}' has no values"))
                elif frac > 0:
                    msgs.append(("info", f"target '{c}': {100 * frac:.1f}% blank cells "
                                         "(ignored by the loss)"))
            if self.time_column:
                t = df[self.time_column]
                if t.isna().any():
                    msgs.append(("error", f"time column '{self.time_column}' has blank cells"))
                if (t < 0).any():
                    msgs.append(("error", f"negative values in time column '{self.time_column}'"))
                dup = df.duplicated([self.case_column, self.time_column])
                if dup.any():
                    msgs.append(("warning", f"{int(dup.sum())} duplicated (case, time) rows"))
                single = (df.groupby(self.case_column, sort=False).size() == 1).sum()
                if single:
                    msgs.append(("info", f"{int(single)} case(s) have a single time point"))
        if self.n_cases < 5:
            msgs.append(("warning", f"only {self.n_cases} cases - expect poor generalization"))
        errors = [m for lvl, m in msgs if lvl == "error"]
        if errors and raise_on_error:
            raise DataError("data check failed:\n  - " + "\n  - ".join(errors))
        return msgs

    def summary(self, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        cols = list(columns) if columns is not None else self.numeric_columns()
        d = self.df[cols]
        return pd.DataFrame({"count": d.count(), "blank": d.isna().sum(), "min": d.min(),
                             "max": d.max(), "mean": d.mean()})

    def detect_monotonic(self, targets: Sequence[str], rtol: float = 1e-6) -> dict:
        """Targets that are monotone in time within **every** case.

        Returns ``{target: "increasing" | "decreasing"}`` - candidates for the
        physics constraint ``monotonic=`` of :class:`BurnupNN` (e.g. burnup,
        build-up of U-232/U-236/Np-237 without removal).
        """
        if not self.time_column:
            return {}
        out = {}
        g = self.df.groupby(self.case_column, sort=False)
        for c in targets:
            scale = np.nanmax(np.abs(self.df[c].to_numpy(float))) or 1.0
            tol = rtol * scale
            inc = dec = True
            for _, d in g:
                v = d[c].to_numpy(float)
                v = v[np.isfinite(v)]
                if len(v) < 2:
                    continue
                dv = np.diff(v)
                inc &= bool(np.all(dv >= -tol))
                dec &= bool(np.all(dv <= tol))
                if not (inc or dec):
                    break
            if inc and not dec:
                out[c] = "increasing"
            elif dec and not inc:
                out[c] = "decreasing"
        return out

    # ------------------------------------------------------------- writing
    def to_csv(self, path: str, float_format: str = "%.7g") -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.df.to_csv(path, index=False, float_format=float_format)
        return path


def write_template(path: str, features: Sequence[str], targets: Sequence[str],
                   case_column: str = "case_id", time_column: str = "time_days",
                   times: Sequence[float] = (0.0, 30.0, 180.0, 365.0, 730.0),
                   example_values: Optional[dict] = None, n_cases: int = 2) -> str:
    """Write an empty data template with the right columns.

    Feature cells are pre-filled from ``example_values`` (if given); target
    cells are left blank for you to paste results from your code.
    """
    rows = []
    for i in range(n_cases):
        for t in times:
            r = {case_column: f"my_run_{i + 1:02d}", time_column: t}
            for f in features:
                r[f] = (example_values or {}).get(f, np.nan)
            for y in targets:
                r[y] = np.nan
            rows.append(r)
    df = pd.DataFrame(rows, columns=[case_column, time_column, *features, *targets])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.lower().endswith((".xlsx", ".xlsm")):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path
