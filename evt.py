"""Peaks-over-threshold and block-maxima tools for significant wave height.

Conventions
-----------
* GPD shape is xi (xi < 0: bounded tail). scipy's genpareto uses c = xi.
* GEV shape is xi as well; scipy's genextreme uses c = -xi, handled in the wrappers.
* A "rate" is the number of independent storm peaks above the threshold per year.
* Return level z_T satisfies: rate * P(X > z_T | X > u) = 1 / T.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

XI_EPS = 1e-6


# ---------------------------------------------------------------- declustering
def decluster_runs(series: pd.Series, threshold: float, separation_hours: float = 48.0) -> pd.DataFrame:
    """Group exceedances into storms and keep each storm's maximum.

    Two exceedances belong to the same storm if the time between them does not exceed
    separation_hours (runs declustering). Returns a frame with columns time, hs.
    """
    exceed = series[series > threshold]
    if exceed.empty:
        return pd.DataFrame(columns=["time", "hs"])
    gaps = exceed.index.to_series().diff() > pd.Timedelta(hours=separation_hours)
    cluster_id = gaps.cumsum().to_numpy()
    grouped = exceed.groupby(cluster_id)
    peaks = grouped.idxmax()
    out = pd.DataFrame({"time": peaks.to_numpy(), "hs": grouped.max().to_numpy()})
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- GPD basics
def gpd_loglik(excess, xi, sigma):
    x = np.asarray(excess, float)
    if sigma <= 0:
        return -np.inf
    if abs(xi) < XI_EPS:
        return float(-len(x) * np.log(sigma) - x.sum() / sigma)
    t = 1.0 + xi * x / sigma
    if np.any(t <= 0):
        return -np.inf
    return float(-len(x) * np.log(sigma) - (1.0 / xi + 1.0) * np.log(t).sum())


def fit_gpd(excess):
    """Maximum-likelihood GPD fit to excesses over the threshold (location fixed at 0)."""
    x = np.asarray(excess, float)
    xi, _, sigma = stats.genpareto.fit(x, floc=0.0)
    return float(xi), float(sigma)


def return_level(u, xi, sigma, rate, T):
    """Return level in metres for return period T (years)."""
    m = rate * np.asarray(T, float)
    if abs(xi) < XI_EPS:
        return u + sigma * np.log(m)
    return u + sigma / xi * (m**xi - 1.0)


def exceedance_probability(x, xi, sigma):
    """P(excess > x) for a GPD."""
    x = np.asarray(x, float)
    if abs(xi) < XI_EPS:
        return np.exp(-x / sigma)
    t = np.maximum(1.0 + xi * x / sigma, 0.0)
    with np.errstate(divide="ignore"):
        return np.where(t > 0, t ** (-1.0 / xi), 0.0)


# ---------------------------------------------------------------- threshold diagnostics
def mean_residual_life(values, thresholds):
    """Mean excess and an approximate 95% band for a set of thresholds."""
    v = np.asarray(values, float)
    rows = []
    for u in thresholds:
        ex = v[v > u] - u
        n = len(ex)
        if n < 5:
            continue
        m, s = ex.mean(), ex.std(ddof=1)
        rows.append((u, n, m, m - 1.96 * s / np.sqrt(n), m + 1.96 * s / np.sqrt(n)))
    return pd.DataFrame(rows, columns=["u", "n", "mean_excess", "lo", "hi"])


def parameter_stability(series: pd.Series, quantiles, separation_hours=48.0, years=None):
    """Fit GPD for a grid of threshold quantiles on declustered peaks.

    The standard error of xi uses the usual large-sample formula (1 + xi) / sqrt(n),
    which holds for xi > -0.5.
    """
    if years is None:
        years = (series.index[-1] - series.index[0]).days / 365.25
    rows = []
    for q in quantiles:
        u = float(series.quantile(q))
        peaks = decluster_runs(series, u, separation_hours)
        if len(peaks) < 20:
            continue
        xi, sigma = fit_gpd(peaks["hs"].to_numpy() - u)
        rows.append(
            {
                "quantile": q,
                "u": u,
                "n": len(peaks),
                "rate": len(peaks) / years,
                "xi": xi,
                "sigma": sigma,
                "se_xi": (1 + xi) / np.sqrt(len(peaks)),
                "modified_scale": sigma - xi * u,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- uncertainty
def bootstrap_return_levels(peaks: pd.DataFrame, u: float, n_years: float, T_list, n_boot=1000, seed=0):
    """Year-resampling bootstrap of return levels.

    Whole calendar years are drawn with replacement, so the number of storms per year
    (the rate) and the clustering of storms within a year are both resampled. Years
    without storms stay in the pool. Returns an array (n_boot, len(T_list)).
    """
    rng = np.random.default_rng(seed)
    years = pd.DatetimeIndex(peaks["time"]).year.to_numpy()
    all_years = np.arange(years.min(), years.max() + 1)
    by_year = {y: peaks.loc[years == y, "hs"].to_numpy() for y in all_years}
    T_list = np.asarray(T_list, float)
    out = np.full((n_boot, len(T_list)), np.nan)
    for b in range(n_boot):
        draw = rng.choice(all_years, size=len(all_years), replace=True)
        sample = np.concatenate([by_year[y] for y in draw])
        if len(sample) < 20:
            continue
        try:
            xi, sigma = fit_gpd(sample - u)
        except Exception:
            continue
        rate = len(sample) / len(all_years)
        out[b] = return_level(u, xi, sigma, rate, T_list)
    return out


def profile_ci_xi(excess, level=0.95, grid=None):
    """Profile-likelihood confidence interval for the GPD shape parameter."""
    x = np.asarray(excess, float)
    xi_hat, sigma_hat = fit_gpd(x)
    ll_max = gpd_loglik(x, xi_hat, sigma_hat)
    crit = stats.chi2.ppf(level, 1) / 2.0
    grid = np.linspace(xi_hat - 0.6, xi_hat + 0.6, 241) if grid is None else grid

    def prof(xi):
        res = optimize.minimize_scalar(
            lambda s: -gpd_loglik(x, xi, s), bounds=(1e-4, 10 * x.max()), method="bounded"
        )
        return -res.fun

    vals = np.array([prof(g) for g in grid])
    inside = grid[vals >= ll_max - crit]
    return float(inside.min()), float(inside.max())


def profile_ci_return_level(excess, u, rate, T, level=0.95, n_grid=160):
    """Profile-likelihood interval for the T-year return level, conditional on the rate.

    The scale is expressed through the return level and xi, so that the return level
    itself is the parameter of interest.
    """
    x = np.asarray(excess, float)
    xi_hat, sigma_hat = fit_gpd(x)
    ll_max = gpd_loglik(x, xi_hat, sigma_hat)
    z_hat = float(return_level(u, xi_hat, sigma_hat, rate, T))
    crit = stats.chi2.ppf(level, 1) / 2.0
    m = rate * T

    def sigma_from(z, xi):
        if abs(xi) < XI_EPS:
            return (z - u) / np.log(m)
        return (z - u) * xi / (m**xi - 1.0)

    def prof(z):
        def neg(xi):
            s = sigma_from(z, xi)
            if not np.isfinite(s) or s <= 0:
                return 1e12
            val = -gpd_loglik(x, xi, s)
            return val if np.isfinite(val) else 1e12

        res = optimize.minimize_scalar(neg, bounds=(-0.9, 0.9), method="bounded")
        return -res.fun

    span = max(z_hat - u, 0.5)
    zs = np.linspace(u + 0.3 * span, z_hat + 2.5 * span, n_grid)
    vals = np.array([prof(z) for z in zs])
    inside = zs[vals >= ll_max - crit]
    if len(inside) == 0:
        return z_hat, z_hat
    return float(inside.min()), float(inside.max())


# ---------------------------------------------------------------- two populations
def combined_return_level(u, pop_a, pop_b, T):
    """Return level for two independent storm populations sharing the threshold u.

    pop_a and pop_b are dicts with xi, sigma, rate. The T-year level is where the summed
    annual exceedance rate equals 1/T.
    """

    def annual_rate(z):
        x = z - u
        return sum(p["rate"] * float(exceedance_probability(x, p["xi"], p["sigma"])) for p in (pop_a, pop_b))

    target = 1.0 / T
    lo, hi = u + 1e-6, u + 50.0
    if annual_rate(hi) > target:
        return np.nan
    return float(optimize.brentq(lambda z: annual_rate(z) - target, lo, hi))


# ---------------------------------------------------------------- annual maxima
def annual_maxima(series: pd.Series) -> pd.Series:
    return series.groupby(series.index.year).max()


def fit_gev(maxima):
    c, loc, scale = stats.genextreme.fit(np.asarray(maxima, float))
    return float(-c), float(loc), float(scale)  # xi = -c


def gev_return_level(xi, loc, scale, T):
    y = -np.log(1.0 - 1.0 / np.asarray(T, float))
    if abs(xi) < XI_EPS:
        return loc - scale * np.log(y)
    return loc + scale / xi * (y ** (-xi) - 1.0)


def gev_bootstrap(maxima, T_list, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.asarray(maxima, float)
    out = np.empty((n_boot, len(T_list)))
    for b in range(n_boot):
        xi, loc, scale = fit_gev(rng.choice(m, size=len(m), replace=True))
        out[b] = gev_return_level(xi, loc, scale, T_list)
    return out


# ---------------------------------------------------------------- trend and homogeneity
def mann_kendall(x):
    """Mann-Kendall test with tie correction and Sen's slope (per time step)."""
    x = np.asarray(x, float)
    n = len(x)
    s = 0.0
    for i in range(n - 1):
        s += np.sign(x[i + 1 :] - x[i]).sum()
    _, counts = np.unique(x, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * stats.norm.sf(abs(z))
    slopes = [(x[j] - x[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    return {"S": float(s), "z": float(z), "p": float(p), "sen_slope": float(np.median(slopes))}


def pettitt_test(x):
    """Pettitt change-point test; returns the change index and an approximate p-value."""
    x = np.asarray(x, float)
    n = len(x)
    ranks = stats.rankdata(x)
    u = np.array([2 * ranks[:t].sum() - t * (n + 1) for t in range(1, n)])
    k = np.abs(u).max()
    p = min(1.0, 2.0 * np.exp(-6.0 * k**2 / (n**3 + n**2)))
    return {"change_index": int(np.abs(u).argmax()) + 1, "K": float(k), "p": float(p)}
