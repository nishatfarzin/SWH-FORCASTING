import numpy as np

from waveshelf import metrics as m


def test_basic_scores():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    pred = obs + 0.5
    assert np.isclose(m.rmse(obs, pred), 0.5)
    assert np.isclose(m.mae(obs, pred), 0.5)
    assert np.isclose(m.bias(obs, pred), 0.5)
    assert np.isclose(m.scatter_index(obs, pred), 0.5 / 2.5)


def test_skill_score_against_itself_is_zero():
    rng = np.random.default_rng(1)
    obs = rng.normal(size=500)
    pred = obs + rng.normal(scale=0.3, size=500)
    assert np.isclose(m.skill_vs_reference(obs, pred, pred), 0.0)


def test_peak_error_sign_and_timing():
    obs = np.array([0, 1, 5, 2, 0], float)
    pred = np.array([0, 1, 2, 4, 0], float)  # peak too low and one step late
    mag, timing = m.peak_error(obs, pred, step_hours=3)
    assert np.isclose(mag, 1.0)
    assert timing == 3.0


def test_block_bootstrap_detects_better_model():
    rng = np.random.default_rng(2)
    obs = np.sin(np.linspace(0, 60, 3000)) + 2
    good = obs + rng.normal(scale=0.05, size=obs.size)
    bad = obs + rng.normal(scale=0.30, size=obs.size)
    res = m.paired_block_bootstrap(obs, good, bad, block_len=40, n_boot=300, seed=3)
    assert res["diff"] < 0 and res["ci95"][1] < 0 and res["p_a_better"] > 0.99


def test_diebold_mariano_sign():
    rng = np.random.default_rng(4)
    obs = rng.normal(size=2000)
    good = obs + rng.normal(scale=0.1, size=obs.size)
    bad = obs + rng.normal(scale=0.5, size=obs.size)
    stat, p = m.diebold_mariano(obs, good, bad, horizon_steps=4)
    assert stat < 0 and p < 0.01


def test_paired_tests_from_diff_detects_shift():
    rng = np.random.default_rng(11)
    d = rng.normal(loc=0.05, scale=0.2, size=3000)
    res = m.paired_tests_from_diff(d, horizon_steps=4, block_len=40, n_boot=300, seed=1)
    assert res["mean_diff"] > 0 and res["p"] < 0.01
    assert res["ci95"][0] > 0
    null = m.paired_tests_from_diff(rng.normal(size=3000), horizon_steps=4, block_len=40, n_boot=300, seed=1)
    assert null["p"] > 0.01
