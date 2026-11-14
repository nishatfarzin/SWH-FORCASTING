"""Synthetic wave series for tests and smoke runs. Not physical; never use for results."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def synthetic_series(years=31, seed=0, xi=-0.05, sigma=0.8, base=1.2) -> pd.Series:
    """3-hourly series: seasonal background plus injected storms with GPD-distributed peaks."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1993-01-01", periods=int(years * 365.25 * 8), freq="3h")
    season = np.sin(2 * np.pi * np.asarray(idx.dayofyear) / 365.25)
    hs = base + 0.4 * season + rng.gamma(2.0, 0.12, len(idx))
    n_storms = int(years * 9)
    centres = rng.choice(len(idx) - 40, n_storms, replace=False) + 20
    peaks = 2.5 + stats.genpareto.rvs(xi, scale=sigma, size=n_storms, random_state=rng)
    shape = np.exp(-0.5 * (np.arange(-16, 17) / 5.0) ** 2)
    for c, pk in zip(centres, peaks):
        hs[c - 16 : c + 17] = np.maximum(hs[c - 16 : c + 17], pk * shape)
    return pd.Series(hs, index=idx)
