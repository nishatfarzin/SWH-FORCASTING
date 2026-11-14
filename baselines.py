"""Reference forecasts: persistence, ridge autoregression and gradient-boosted trees."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .features import FEATURES

HS_INDEX = FEATURES.index("Hs")


def persistence(X_scaled: np.ndarray) -> np.ndarray:
    """Forecast = Hs at the end of the input window (scaled units)."""
    return X_scaled[:, -1, HS_INDEX]


def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


def fit_ridge(X_train, y_train, alpha: float = 1.0) -> Ridge:
    """Linear model on the flattened window (all features, all lags)."""
    return Ridge(alpha=alpha).fit(_flatten(X_train), y_train)


def predict_ridge(model: Ridge, X) -> np.ndarray:
    return model.predict(_flatten(X))


def fit_xgb(X_train, y_train, X_val, y_val, seed: int = 0):
    """Gradient-boosted trees with early stopping on the validation years."""
    from xgboost import XGBRegressor  # imported here so the rest works without xgboost

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        early_stopping_rounds=30,
        n_jobs=-1,
    )
    model.fit(_flatten(X_train), y_train, eval_set=[(_flatten(X_val), y_val)], verbose=False)
    return model


def predict_xgb(model, X) -> np.ndarray:
    return model.predict(_flatten(X))
