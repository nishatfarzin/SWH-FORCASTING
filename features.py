"""Feature construction, chronological splitting and window building."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

STEP_HOURS = 3
CALM_WIND_MS = 0.1  # below this speed the wind direction is undefined and set to zero signal

FEATURES = ["U", "Wind_Sin", "Wind_Cos", "SST_C", "Hs", "Tp", "Wind_lag3h", "Wind_lag6h", "Hs_lag3h"]


def check_regular(index: pd.DatetimeIndex, step_hours: int = STEP_HOURS) -> None:
    """Raise if the time axis has gaps or duplicates."""
    deltas = np.diff(index.values).astype("timedelta64[h]").astype(int)
    if not np.all(deltas == step_hours):
        raise ValueError(f"time axis is not a regular {step_hours}-hourly grid")


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw columns (u10, v10, sst_k, hs, tp) into the nine model features."""
    check_regular(raw.index)
    out = pd.DataFrame(index=raw.index)
    out["U"] = np.hypot(raw["u10"], raw["v10"])
    theta = np.arctan2(raw["v10"], raw["u10"])
    calm = out["U"] < CALM_WIND_MS
    out["Wind_Sin"] = np.where(calm, 0.0, np.sin(theta))
    out["Wind_Cos"] = np.where(calm, 0.0, np.cos(theta))
    out["SST_C"] = raw["sst_k"] - 273.15
    out["Hs"] = raw["hs"]
    out["Tp"] = raw["tp"]
    out["Wind_lag3h"] = out["U"].shift(1)
    out["Wind_lag6h"] = out["U"].shift(2)
    out["Hs_lag3h"] = out["Hs"].shift(1)
    return out.dropna()[FEATURES]


def chronological_split(df: pd.DataFrame, train_end: int = 2014, val_end: int = 2018):
    """Train up to train_end, validation up to val_end, test afterwards (calendar years)."""
    year = df.index.year
    return df[year <= train_end], df[(year > train_end) & (year <= val_end)], df[year > val_end]


def fit_scaler(train: pd.DataFrame) -> StandardScaler:
    """Fit on the training years only; reuse unchanged for validation and test."""
    return StandardScaler().fit(train.values)


def make_windows(
    features: pd.DataFrame, scaler: StandardScaler, window: int, horizon_steps: int, target: str = "Hs"
):
    """Build (X, y, t_target, y_phys) arrays.

    X has shape (n, window, n_features) and holds scaled inputs ending at time t.
    The target is the scaled Hs at t + horizon_steps. y_phys is the same target in metres,
    and t_target the timestamp of the target. Windows that would cross a gap are dropped.
    """
    check_regular(features.index)
    values = scaler.transform(features.values)
    t_idx = features.columns.get_loc(target)
    hs_phys = features[target].to_numpy()
    n = len(features)
    last = n - horizon_steps
    X, y, ts, yp = [], [], [], []
    for end in range(window - 1, last):
        X.append(values[end - window + 1 : end + 1])
        y.append(values[end + horizon_steps, t_idx])
        ts.append(features.index[end + horizon_steps])
        yp.append(hs_phys[end + horizon_steps])
    return np.asarray(X, np.float32), np.asarray(y, np.float32), pd.DatetimeIndex(ts), np.asarray(yp)


def inverse_hs(
    scaler: StandardScaler, y_scaled: np.ndarray, hs_index: int = FEATURES.index("Hs")
) -> np.ndarray:
    return np.asarray(y_scaled) * scaler.scale_[hs_index] + scaler.mean_[hs_index]
