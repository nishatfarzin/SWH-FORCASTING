"""Collocate multi-mission altimeter Hs with the station reanalysis series and score the match-ups.

Steps (same choices as the first analysis):
 1. read every mission file (.nc, variable swh_adjusted) in --alt-dir, tag it with the mission name, apply three checks
    (inside the bounding box, 0 <= swh <= 20 m, no duplicate time-mission rows);
 2. for each station keep the points within 0.25 degrees (as a circle, ~27.8 km) of the requested position;
 3. match each point to the nearest reanalysis time within +/-1.5 h;
 4. group consecutive points of one mission with gaps <= 10 min into a pass and take the median per pass;
 5. report bias (altimeter minus reanalysis), RMSE, MAE, correlation and scatter index over the passes.

Usage: python scripts/altimetry_matchup.py --alt-dir <folder with mission .nc files> --stations-dir data/stations --out results/altimetry
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from common import load_stations, read_station

from waveshelf.events import haversine_km

BBOX = {"lon_min": 88.0, "lon_max": 92.5, "lat_min": 18.0, "lat_max": 22.5}
RADIUS_DEG, WINDOW_H, PASS_GAP_MIN = 0.25, 1.5, 10


def load_mission(path):
    mission = os.path.basename(path).replace(".nc", "")
    ds = xr.open_dataset(path)
    lat = next(n for n in ("lat", "latitude") if n in ds.coords or n in ds.data_vars)
    lon = next(n for n in ("lon", "longitude") if n in ds.coords or n in ds.data_vars)
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(ds["time"].values),
            "lat": ds[lat].values,
            "lon": ds[lon].values,
            "swh": ds["swh_adjusted"].values,
            "mission": mission,
        }
    )
    ds.close()
    return df


def merge_and_check(alt_dir):
    files = sorted(glob.glob(os.path.join(alt_dir, "*.nc")))
    combined = pd.concat([load_mission(f) for f in files], ignore_index=True)
    inside = combined["lon"].between(BBOX["lon_min"] - 0.01, BBOX["lon_max"] + 0.01) & combined[
        "lat"
    ].between(BBOX["lat_min"] - 0.01, BBOX["lat_max"] + 0.01)
    combined = combined[inside]
    combined = combined[combined["swh"].between(0, 20)]
    combined = combined.drop_duplicates(subset=["time", "mission"]).sort_values("time").reset_index(drop=True)
    print(f"{len(files)} mission files, {len(combined)} points after checks")
    print(combined["mission"].value_counts().to_string())
    return combined


def match_station(alt, lat, lon, reanalysis_hs):
    near = alt[
        alt["lat"].between(lat - RADIUS_DEG, lat + RADIUS_DEG)
        & alt["lon"].between(lon - RADIUS_DEG, lon + RADIUS_DEG)
    ].copy()
    radius_km = float(haversine_km(lat, lon, lat + RADIUS_DEG, lon))
    near = near[haversine_km(lat, lon, near["lat"].to_numpy(), near["lon"].to_numpy()) <= radius_km]
    near = near.sort_values(["mission", "time"]).reset_index(drop=True)
    times = reanalysis_hs.index.to_numpy()
    idx = np.searchsorted(times, near["time"].to_numpy())
    best, best_dt = np.full(len(near), -1), np.full(len(near), np.inf)
    for shift in (-1, 0):
        cand = np.clip(idx + shift, 0, len(times) - 1)
        dt = np.abs((times[cand] - near["time"].to_numpy()) / np.timedelta64(1, "h"))
        better = dt < best_dt
        best[better], best_dt[better] = cand[better], dt[better]
    ok = best_dt <= WINDOW_H
    out = near[ok].copy()
    out["reanalysis_hs"] = reanalysis_hs.to_numpy()[best[ok]]
    return out.rename(columns={"time": "altimetry_time", "swh": "altimetry_swh"})


def to_passes(matched):
    m = matched.sort_values(["mission", "altimetry_time"]).reset_index(drop=True)
    new_pass = (m["mission"] != m["mission"].shift()) | (
        m["altimetry_time"].diff() > pd.Timedelta(minutes=PASS_GAP_MIN)
    )
    m["pass_id"] = new_pass.cumsum()
    return (
        m.groupby("pass_id")
        .agg(
            altimetry_swh=("altimetry_swh", "median"),
            reanalysis_hs=("reanalysis_hs", "median"),
            n_points_in_pass=("altimetry_swh", "size"),
        )
        .reset_index()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt-dir", required=True)
    ap.add_argument("--stations-dir", default="data/stations")
    ap.add_argument("--out", default="results/altimetry")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    alt = merge_and_check(args.alt_dir)
    rows = []
    for sid, st in load_stations().items():
        hs = read_station(Path(args.stations_dir) / f"{sid}.csv")["hs"]
        matched = match_station(alt, st["lat"], st["lon"], hs)
        passes = to_passes(matched)
        passes.to_parquet(out / f"{sid}_matched_passes.parquet", index=False)
        resid = passes["altimetry_swh"] - passes["reanalysis_hs"]
        rmse = float(np.sqrt((resid**2).mean()))
        rows.append(
            {
                "station": sid,
                "n_raw_points": len(matched),
                "n_independent_passes": len(passes),
                "bias_m": resid.mean(),
                "rmse_m": rmse,
                "mae_m": resid.abs().mean(),
                "correlation": passes["altimetry_swh"].corr(passes["reanalysis_hs"]),
                "si": rmse / passes["reanalysis_hs"].mean(),
            }
        )
        print(sid, rows[-1])
    pd.DataFrame(rows).to_csv(out / "table_altimetry_validation.csv", index=False)


if __name__ == "__main__":
    main()
