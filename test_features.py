import numpy as np
import pandas as pd
import pytest

from waveshelf import baselines as b
from waveshelf import features as f


def raw_frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1993-01-01", periods=n, freq="3h")
    return pd.DataFrame(
        {
            "u10": rng.normal(size=n),
            "v10": rng.normal(size=n),
            "sst_k": 300 + rng.normal(size=n),
            "hs": 1.5 + np.sin(np.arange(n) / 20) + rng.normal(scale=0.05, size=n),
            "tp": 8 + rng.normal(size=n),
        },
        index=idx,
    )


def test_feature_columns_and_calm_wind():
    raw = raw_frame()
    raw.loc[raw.index[5], ["u10", "v10"]] = 0.0
    feats = f.build_features(raw)
    assert list(feats.columns) == f.FEATURES
    row = feats.loc[raw.index[5]]
    assert row["Wind_Sin"] == 0 and row["Wind_Cos"] == 0
    assert not feats.isna().any().any()


def test_gap_is_rejected():
    raw = raw_frame().drop(index=pd.date_range("1993-01-03", periods=2, freq="3h"))
    with pytest.raises(ValueError):
        f.build_features(raw)


def test_target_alignment_and_scaler_uses_training_only():
    feats = f.build_features(raw_frame())
    train, val, test = f.chronological_split(feats, train_end=1993, val_end=1994)
    scaler = f.fit_scaler(train)
    assert np.allclose(scaler.mean_, train.mean().to_numpy())
    X, y, t, y_phys = f.make_windows(train, scaler, window=16, horizon_steps=4)
    # the physical target must equal the series value at the target time
    assert np.allclose(y_phys, train.loc[t, "Hs"].to_numpy())
    # and the last input step must be exactly horizon_steps*3 h before the target
    first_end = train.index[15]
    assert t[0] - first_end == pd.Timedelta(hours=12)
    # inverse transform recovers metres
    assert np.allclose(f.inverse_hs(scaler, y), y_phys, atol=1e-4)


def test_persistence_returns_last_hs():
    feats = f.build_features(raw_frame())
    train, *_ = f.chronological_split(feats, 1993, 1994)
    scaler = f.fit_scaler(train)
    X, y, t, _ = f.make_windows(train, scaler, 16, 2)
    assert np.allclose(b.persistence(X), X[:, -1, f.FEATURES.index("Hs")])


def test_ridge_baseline_beats_persistence_on_smooth_signal():
    feats = f.build_features(raw_frame(n=6000))
    train, val, test = f.chronological_split(feats, 1993, 1994)
    scaler = f.fit_scaler(train)
    Xtr, ytr, *_ = f.make_windows(train, scaler, 16, 4)
    Xte, yte, *_ = f.make_windows(test, scaler, 16, 4)
    model = b.fit_ridge(Xtr, ytr)
    err_ridge = np.mean((b.predict_ridge(model, Xte) - yte) ** 2)
    err_pers = np.mean((b.persistence(Xte) - yte) ** 2)
    assert err_ridge < err_pers
