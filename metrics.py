"""Forecast scores and paired comparison tools.

All functions work on plain numpy arrays in physical units (metres).
Time series are assumed to be regularly spaced (3-hourly in this study).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def rmse(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def mae(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    return float(np.mean(np.abs(obs - pred)))


def bias(obs, pred):
    """Mean error, prediction minus observation (negative = underprediction)."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    return float(np.mean(pred - obs))


def r2(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def scatter_index(obs, pred):
    """RMSE divided by the mean observation."""
    return rmse(obs, pred) / float(np.mean(obs))


def skill_vs_reference(obs, pred, ref):
    """MSE skill score against a reference forecast (1 = perfect, 0 = no better, <0 = worse)."""
    obs, pred, ref = (np.asarray(a, float) for a in (obs, pred, ref))
    return float(1.0 - np.mean((obs - pred) ** 2) / np.mean((obs - ref) ** 2))


def summary(obs, pred):
    return {
        "rmse": rmse(obs, pred),
        "mae": mae(obs, pred),
        "bias": bias(obs, pred),
        "r2": r2(obs, pred),
        "si": scatter_index(obs, pred),
    }


def upper_tail_scores(obs, pred, quantiles=(0.95, 0.99)):
    """RMSE and bias on the samples whose observation lies above given observed quantiles."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    out = {}
    for q in quantiles:
        mask = obs >= np.quantile(obs, q)
        tag = f"p{int(round(q * 100))}"
        out[f"rmse_{tag}"] = rmse(obs[mask], pred[mask])
        out[f"bias_{tag}"] = bias(obs[mask], pred[mask])
    return out


def peak_error(obs_window, pred_window, step_hours=3, search_hours=30):
    """Peak magnitude and timing error inside one storm window.

    The magnitude error is the observed window maximum minus the predicted maximum
    found within +/- search_hours of the observed peak (positive = underprediction).
    The timing error is the distance in hours between the two maxima. Restricting the
    search avoids matching a distant, unrelated predicted maximum.
    """
    obs_window, pred_window = np.asarray(obs_window, float), np.asarray(pred_window, float)
    i_obs = int(np.argmax(obs_window))
    half = int(search_hours // step_hours)
    lo, hi = max(0, i_obs - half), min(len(pred_window), i_obs + half + 1)
    i_pred = lo + int(np.argmax(pred_window[lo:hi]))
    magnitude = float(obs_window[i_obs] - pred_window[i_pred])
    timing = float(abs(i_pred - i_obs) * step_hours)
    return magnitude, timing


def moving_block_indices(n, block_len, rng):
    """One moving-block bootstrap resample of range(n) (Kuensch, 1989)."""
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
    return idx[:n]


def paired_block_bootstrap(obs, pred_a, pred_b, block_len=56, n_boot=2000, seed=0, score=rmse):
    """Bootstrap the difference score(A) - score(B) on a common test set.

    Blocks keep the serial correlation of the errors; 56 steps = 7 days at 3-hourly spacing.
    Returns the observed difference, a 95% percentile interval and the share of
    resamples in which A is better (lower score) than B.
    """
    obs, pred_a, pred_b = (np.asarray(a, float) for a in (obs, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(obs)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = moving_block_indices(n, block_len, rng)
        diffs[k] = score(obs[idx], pred_a[idx]) - score(obs[idx], pred_b[idx])
    observed = score(obs, pred_a) - score(obs, pred_b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(observed), "ci95": (float(lo), float(hi)), "p_a_better": float(np.mean(diffs < 0))}


def diebold_mariano(obs, pred_a, pred_b, horizon_steps=1, small_sample=True):
    """Diebold-Mariano test on squared errors with a Newey-West (Bartlett) variance.

    horizon_steps is the forecast horizon in time steps; the long-run variance uses
    horizon_steps - 1 autocovariances. A negative statistic means forecast A has
    the smaller squared error. Returns (statistic, two-sided p-value).
    """
    obs, pred_a, pred_b = (np.asarray(a, float) for a in (obs, pred_a, pred_b))
    d = (obs - pred_a) ** 2 - (obs - pred_b) ** 2
    n = len(d)
    d_bar = d.mean()
    lags = max(horizon_steps - 1, 0)
    dc = d - d_bar
    lrv = np.dot(dc, dc) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * np.dot(dc[k:], dc[:-k]) / n
    stat = d_bar / np.sqrt(lrv / n)
    if small_sample:
        h = horizon_steps
        stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        p = 2 * stats.t.sf(abs(stat), df=n - 1)
    else:
        p = 2 * stats.norm.sf(abs(stat))
    return float(stat), float(p)


def paired_tests_from_diff(d, horizon_steps=1, block_len=56, n_boot=2000, seed=0):
    """Test whether a loss-differential series d(t) = loss_A(t) - loss_B(t) has mean zero.

    d is typically the squared-error difference averaged over random seeds, so the test
    asks whether model A differs from model B beyond the time-series noise on the test set.
    Returns the mean difference, a Diebold-Mariano style statistic and p-value (Newey-West
    variance), and a 95% moving-block-bootstrap interval for the mean.
    """
    d = np.asarray(d, float)
    n = len(d)
    mean = d.mean()
    lags = max(horizon_steps - 1, 0)
    dc = d - mean
    lrv = np.dot(dc, dc) / n
    for k in range(1, lags + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * np.dot(dc[k:], dc[:-k]) / n
    stat = mean / np.sqrt(lrv / n)
    p = 2 * stats.norm.sf(abs(stat))
    rng = np.random.default_rng(seed)
    boots = np.array([d[moving_block_indices(n, block_len, rng)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": float(mean), "stat": float(stat), "p": float(p), "ci95": (float(lo), float(hi))}
