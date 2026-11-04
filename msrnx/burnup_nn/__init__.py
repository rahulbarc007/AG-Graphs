"""Neural-network burnup framework for molten salt reactor design.

Learn burnup trajectories - k_eff, nuclide inventories, breeding ratio, poison
fractions, burnup - as functions of design parameters and time, from *any*
depletion data: the built-in MSR-NX physics runs, your Serpent / SCALE /
OpenMC / in-house results, or both merged.

Quick start (Python)::

    from msrnx.burnup_nn import BurnupDataset, BurnupNN
    data = BurnupDataset.load("physics_dataset.csv") + BurnupDataset.load(
        "my_serpent_runs.xlsx", rename={"BU": "burnup_MWd_kgHM"}, source="serpent")
    train, val, test = data.split(test=0.15, val=0.15)
    model = BurnupNN(features=["uf4_pct", "thf4_pct", "salt_fraction", "power_MW"],
                     targets=["k_eff", "U233_kg", "Pa233_kg"]).fit(train, val)
    print(model.evaluate(test))
    model.predict_trajectory({"uf4_pct": 0.35, "thf4_pct": 10, "salt_fraction": 0.18,
                              "power_MW": 1000})
    model.save("my_model")

Command line: ``python -m msrnx.burnup_nn --help`` (generate / template / check /
train / finetune / predict / info).  Pure NumPy networks - no deep-learning
framework needed.
"""
from .dataset import BurnupDataset, DataError, write_template
from .model import BurnupNN, ColumnTransform, NNConfig, TimeEncoding, learning_curve, load_model
from .nn import MLP, gradient_check
from .physics_data import (BURNUP_SPACE, COARSE_STEPS_DAYS, DEFAULT_STEPS_DAYS, build_case,
                           generate_physics_dataset, run_case)

__all__ = ["BurnupDataset", "DataError", "write_template", "BurnupNN", "NNConfig",
           "TimeEncoding", "ColumnTransform", "load_model", "learning_curve", "MLP",
           "gradient_check",
           "BURNUP_SPACE", "DEFAULT_STEPS_DAYS", "COARSE_STEPS_DAYS", "build_case",
           "generate_physics_dataset", "run_case", "train", "fine_tune", "predict",
           "check_data", "load_config"]


def __getattr__(name):           # lazy: the pipeline pulls in matplotlib
    if name in ("train", "fine_tune", "predict", "check_data", "load_config"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module 'msrnx.burnup_nn' has no attribute '{name}'")
