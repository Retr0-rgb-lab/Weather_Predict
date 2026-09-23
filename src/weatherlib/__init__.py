"""Shared experiment library for the delta-T_max prediction project.

All experiments import from here to keep one pipeline definition
(cleaning, imputation, feature building, split, metrics, baselines, models).
R1: no torch yet (MLP lands in R3, spec v1.1 section 2.3).
"""

from .data import (
    DATA,
    ROOT,
    STATIONS,
    TARGET_STATION,
    TARGET_VAR,
    UPSTREAM,
    clean,
    impute_station_month,
    load_raw,
    split_column,
)
from .dataset import DeltaMaxData, make_delta_max_dataset
from .features import pressure_gradients, season_features
from .metrics import (
    block_bootstrap_ci,
    block_bootstrap_diff_ci,
    corr_safe,
    mae,
    r2,
    rmse,
    skill,
)
from .baselines import monthly_delta_clim, yesterday_delta, zero
from .models import Fitted, fit_linear

__all__ = [
    "ROOT", "DATA", "STATIONS", "UPSTREAM", "TARGET_STATION", "TARGET_VAR",
    "load_raw", "clean", "impute_station_month", "split_column",
    "DeltaMaxData", "make_delta_max_dataset",
    "pressure_gradients", "season_features",
    "mae", "rmse", "r2", "corr_safe", "skill", "block_bootstrap_ci",
    "block_bootstrap_diff_ci",
    "zero", "yesterday_delta", "monthly_delta_clim",
    "fit_linear", "Fitted",
]
