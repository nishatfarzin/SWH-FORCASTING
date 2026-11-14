import numpy as np
import pandas as pd
import pytest
from scipy import stats

from waveshelf import evt
from waveshelf.synthetic import synthetic_series


def test_gpd_fit_recovers_parameters():
    x = stats.genpareto.rvs(-0.1, scale=1.2, size=4000, random_state=5)
    xi, sigma = evt.fit_gpd(x)
    assert abs(xi + 0.1) < 0.05 and abs(sigma - 1.2) < 0.1


def test_return_level_matches_survival_definition():
    u, xi, sigma, rate, T = 2.0, -0.08, 0.9, 9.0, 100
    z = evt.return_level(u, xi, sigma, rate, T)
    expected_rate = rate * float(evt.exceedance_probability(z - u, xi, sigma))
    assert np.isclose(expected_rate, 1.0 / T, rtol=1e-9)


def test_return_level_exponential_limit():
    z0 = evt.return_level(2.0, 0.0, 0.9, 9.0, 100)
    z1 = evt.return_level(2.0, 1e-5, 0.9, 9.0, 100)
    assert np.isclose(z0, z1, atol=1e-3)


def test_decluster_one_peak_per_storm():
    idx = pd.date_range("2000-01-01", periods=200, freq="3h")
    hs = pd.Series(1.0, index=idx)
    hs.iloc[10:14] = [3.0, 4.0, 3.5, 3.1]  # storm 1
    hs.iloc[60:63] = [3.2, 5.0, 3.3]  # storm 2, far away
    peaks = evt.decluster_runs(hs, 3.0, separation_hours=48)
    assert list(peaks["hs"]) == [4.0, 5.0]


def test_profile_interval_contains_estimate_and_truth():
    x = stats.genpareto.rvs(-0.1, scale=1.0, size=300, random_state=7)
    xi_hat, _ = evt.fit_gpd(x)
    lo, hi = evt.profile_ci_xi(x)
    assert lo < xi_hat < hi
    assert lo < -0.1 < hi


def test_profile_ci_return_level_brackets_point_estimate():
    x = stats.genpareto.rvs(-0.05, scale=0.9, size=300, random_state=8)
    u, rate, T = 2.5, 9.0, 100
    xi, sigma = evt.fit_gpd(x)
    z = evt.return_level(u, xi, sigma, rate, T)
    lo, hi = evt.profile_ci_return_level(x, u, rate, T)
    assert lo < z < hi


def test_year_bootstrap_interval_is_sensible():
    series = synthetic_series()
    u = float(series.quantile(0.93))
    peaks = evt.decluster_runs(series, u)
    xi, sigma = evt.fit_gpd(peaks["hs"].to_numpy() - u)
    rate = len(peaks) / 31
    point = evt.return_level(u, xi, sigma, rate, 100)
    boot = evt.bootstrap_return_levels(peaks, u, 31, [100], n_boot=200, seed=1)
    lo, hi = np.nanpercentile(boot[:, 0], [2.5, 97.5])
    assert lo < point < hi and (hi - lo) > 0.2


def test_combined_return_level_between_single_populations():
    a = {"xi": -0.05, "sigma": 0.8, "rate": 8.0}
    b = {"xi": -0.10, "sigma": 1.0, "rate": 1.0}
    z_ab = evt.combined_return_level(2.5, a, b, 100)
    z_a = evt.return_level(2.5, a["xi"], a["sigma"], a["rate"], 100)
    assert z_ab > z_a  # adding a population cannot lower the level
    z_single = evt.combined_return_level(2.5, a, {**b, "rate": 0.0}, 100)
    assert np.isclose(z_single, z_a, atol=1e-6)


def test_gev_annual_maxima_roundtrip():
    maxima = stats.genextreme.rvs(0.1, loc=7.0, scale=1.0, size=2000, random_state=3)  # xi = -0.1
    xi, loc, scale = evt.fit_gev(maxima)
    assert abs(xi + 0.1) < 0.05
    z100 = evt.gev_return_level(xi, loc, scale, 100)
    true = evt.gev_return_level(-0.1, 7.0, 1.0, 100)
    assert abs(z100 - true) < 0.3


def test_mann_kendall_detects_trend_and_not_noise():
    rng = np.random.default_rng(9)
    trend = np.arange(31) * 0.1 + rng.normal(scale=0.2, size=31)
    noise = rng.normal(size=31)
    assert evt.mann_kendall(trend)["p"] < 0.01
    assert evt.mann_kendall(noise)["p"] > 0.05
    assert evt.mann_kendall(trend)["sen_slope"] == pytest.approx(0.1, abs=0.03)


def test_pettitt_finds_step_change():
    x = np.concatenate(
        [np.random.default_rng(1).normal(0, 0.3, 15), np.random.default_rng(2).normal(2, 0.3, 16)]
    )
    res = evt.pettitt_test(x)
    assert res["p"] < 0.01 and abs(res["change_index"] - 15) <= 1
