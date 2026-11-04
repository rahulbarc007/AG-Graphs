"""Command line for the neural-network burnup framework.

    python -m msrnx.burnup_nn generate --cases 400 --out data/physics_dataset.csv
    python -m msrnx.burnup_nn template --out data/my_runs.csv
    python -m msrnx.burnup_nn check    examples/burnup_nn/config.json
    python -m msrnx.burnup_nn train    examples/burnup_nn/config.json
    python -m msrnx.burnup_nn finetune my_hifi_config.json --model runs/demo/model
    python -m msrnx.burnup_nn predict  --model runs/demo/model --input new_designs.csv
    python -m msrnx.burnup_nn info     --model runs/demo/model
"""
from __future__ import annotations

import argparse
import sys


def _ranges(items):
    out = {}
    for it in items or []:
        name, _, rng = it.partition("=")
        lo, _, hi = rng.partition(":")
        out[name.strip()] = (float(lo), float(hi))
    return out


def cmd_generate(a):
    from ..depletion import FeedControl
    from ..ml.design_space import DesignSpace, Parameter
    from .physics_data import (BURNUP_SPACE, COARSE_STEPS_DAYS, DEFAULT_STEPS_DAYS,
                               generate_physics_dataset)
    rng = _ranges(a.range)
    unknown = set(rng) - set(BURNUP_SPACE.names)
    if unknown:
        sys.exit(f"unknown parameters {sorted(unknown)}; available: {BURNUP_SPACE.names}")
    space = DesignSpace([Parameter(p.name, *rng.get(p.name, (p.low, p.high)),
                                   log=p.log, unit=p.unit, label=p.label)
                         for p in BURNUP_SPACE.params])
    steps = COARSE_STEPS_DAYS if a.steps == "coarse" else DEFAULT_STEPS_DAYS
    feed = FeedControl(target_k=1.0) if a.feed else None
    generate_physics_dataset(a.cases, space, steps, seed=a.seed, transport=a.transport,
                             feed=feed, n_jobs=a.jobs, out_csv=a.out)


def cmd_template(a):
    from .dataset import write_template
    from .physics_data import BURNUP_SPACE
    feats = a.features or BURNUP_SPACE.names
    targets = a.targets or ["k_eff", "burnup_MWd_kgHM", "U233_kg", "Pa233_kg", "Xe135_abs_pct"]
    ex = {p.name: p.nominal for p in BURNUP_SPACE.params}
    path = write_template(a.out, feats, targets, a.case_column, a.time_column, example_values=ex)
    print(f"template written to {path}\n  columns: {a.case_column}, {a.time_column}, "
          f"{', '.join(feats)}, {', '.join(targets)}\n  one row per (case, time point); blank "
          "target cells are allowed")


def cmd_check(a):
    from .pipeline import check_data, load_config
    _, msgs, _ = check_data(load_config(a.config))
    if any(l == "error" for l, _ in msgs):
        sys.exit(1)


def cmd_train(a):
    from .pipeline import train
    train(a.config, a.out)


def cmd_finetune(a):
    from .pipeline import fine_tune
    fine_tune(a.config, a.model, a.out, method=a.method, anchor=a.anchor, freeze=a.freeze,
              max_iter=a.iters)


def cmd_predict(a):
    import numpy as np
    from .pipeline import predict
    times = None
    if a.t_max:
        times = np.linspace(0.0, a.t_max, a.n_times)
    crossing = None
    if a.crossing_target:
        crossing = {"target": a.crossing_target, "level": a.level,
                    "burnup_target": a.burnup_target}
    predict(a.model, a.input, a.out, times, a.n_times, crossing)


def cmd_info(a):
    import pandas as pd
    from .model import BurnupNN
    m = BurnupNN.load(a.model)
    print(m.describe())
    print(f"  features: {m.features}\n  targets : {m.targets}\n  training range: {m.domain}")
    if m.metrics.get("test"):
        met = pd.DataFrame(m.metrics["test"]).T
        cols = [c for c in ("rmse", "rmse_pcm", "r2", "nrmse_pct", "coverage95") if c in met]
        print("\n  test accuracy:\n" + met[cols].to_string(float_format=lambda v: f"{v:.4g}"))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m msrnx.burnup_nn",
                                description="Neural-network burnup framework for MSR design")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="built-in MSR-NX depletion runs -> training CSV")
    g.add_argument("--cases", type=int, default=400)
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--transport", default="lattice",
                   choices=["lattice", "lattice-uncalibrated", "core"])
    g.add_argument("--steps", default="default", choices=["default", "coarse"])
    g.add_argument("--jobs", type=int, default=None)
    g.add_argument("--feed", action="store_true",
                   help="criticality-controlled U-233 feed (k_eff held at 1)")
    g.add_argument("--range", action="append", metavar="NAME=LO:HI",
                   help="override a design-space range, e.g. --range uf4_pct=0.3:0.5")
    g.set_defaults(fn=cmd_generate)

    t = sub.add_parser("template", help="write an empty data template (CSV or XLSX)")
    t.add_argument("--out", required=True)
    t.add_argument("--features", nargs="*")
    t.add_argument("--targets", nargs="*")
    t.add_argument("--case-column", default="case_id")
    t.add_argument("--time-column", default="time_days")
    t.set_defaults(fn=cmd_template)

    c = sub.add_parser("check", help="validate the data of a config file")
    c.add_argument("config")
    c.set_defaults(fn=cmd_check)

    tr = sub.add_parser("train", help="train, evaluate, save model, charts and report")
    tr.add_argument("config")
    tr.add_argument("--out", default=None, help="output folder (default: config output_dir)")
    tr.set_defaults(fn=cmd_train)

    f = sub.add_parser("finetune", help="adapt a trained model to new (e.g. high-fidelity) data")
    f.add_argument("config")
    f.add_argument("--model", required=True)
    f.add_argument("--out", default=None)
    f.add_argument("--method", default="auto", choices=["auto", "correction", "weights"],
                   help="correction: additive discrepancy (few runs); weights: fine-tuning "
                        "(more runs); auto: correction below 12 training cases")
    f.add_argument("--anchor", type=float, default=1e-3)
    f.add_argument("--freeze", type=int, default=0)
    f.add_argument("--iters", type=int, default=300)
    f.set_defaults(fn=cmd_finetune)

    pr = sub.add_parser("predict", help="burnup trajectories for new designs")
    pr.add_argument("--model", required=True)
    pr.add_argument("--input", required=True, help="CSV/XLSX: one row per design")
    pr.add_argument("--out", default="burnup_predictions")
    pr.add_argument("--t-max", type=float, default=None)
    pr.add_argument("--n-times", type=int, default=200)
    pr.add_argument("--crossing-target", default=None, help="e.g. k_eff")
    pr.add_argument("--level", type=float, default=1.0)
    pr.add_argument("--burnup-target", default="burnup_MWd_kgHM")
    pr.set_defaults(fn=cmd_predict)

    i = sub.add_parser("info", help="describe a saved model")
    i.add_argument("--model", required=True)
    i.set_defaults(fn=cmd_info)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
