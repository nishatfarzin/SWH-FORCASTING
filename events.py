"""Link storm peaks to tropical cyclone tracks (IBTrACS) and split them into populations."""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, float)) for a in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def label_cyclone_peaks(
    peaks: pd.DataFrame,
    tracks: pd.DataFrame,
    station_lat: float,
    station_lon: float,
    max_distance_km: float = 800.0,
    window_hours: float = 48.0,
) -> pd.DataFrame:
    """Flag each storm peak that has a cyclone track point nearby in space and time.

    tracks needs columns time, lat, lon (and optionally sid for the storm id). A peak is
    cyclone-related when at least one track point lies within max_distance_km of the
    station and within +/- window_hours of the peak. Both limits are analyst choices;
    run the analysis for several values and report the sensitivity.
    """
    out = peaks.copy()
    t = pd.to_datetime(tracks["time"]).to_numpy()
    dist = haversine_km(station_lat, station_lon, tracks["lat"].to_numpy(), tracks["lon"].to_numpy())
    near = dist <= max_distance_km
    t_near = t[near]
    sid_near = tracks["sid"].to_numpy()[near] if "sid" in tracks else np.full(near.sum(), "")
    window = np.timedelta64(int(window_hours * 3600), "s")
    flags, ids = [], []
    for pt in pd.to_datetime(out["time"]).to_numpy():
        hit = np.abs(t_near - pt) <= window
        flags.append(bool(hit.any()))
        ids.append(sid_near[hit][0] if hit.any() else "")
    out["cyclone"] = flags
    out["storm_id"] = ids
    return out
