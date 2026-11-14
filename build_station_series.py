"""Cut the downloaded fields down to one CSV per station.

Uses the nearest *ocean* grid cell for the waves (cells that are land in the wave model are NaN)
and writes the cell actually used to data/stations/grid_cells.csv. Report that table in the paper.
Usage: python scripts/build_station_series.py --raw data/raw --out data/stations
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from common import load_stations


def nearest_ocean_cell(ds, var, lat, lon):
    """Nearest cell with valid data; returns (lat, lon) of that cell."""
    valid = ds[var].isel(time=0).notnull()
    lat2, lon2 = np.meshgrid(ds["latitude"].values, ds["longitude"].values, indexing="ij")
    dist = np.hypot(lat2 - lat, (lon2 - lon) * np.cos(np.radians(lat)))
    dist = np.where(valid.values, dist, np.inf)
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    return float(ds["latitude"].values[i]), float(ds["longitude"].values[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/stations")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    waves = xr.open_dataset(Path(args.raw) / "cmems_waves.nc")
    era5 = xr.open_mfdataset(str(Path(args.raw) / "era5_*.nc"))
    if "valid_time" in era5.dims:
        era5 = era5.rename({"valid_time": "time"})
    cells = []
    for sid, st in load_stations().items():
        wlat, wlon = nearest_ocean_cell(waves, "VHM0", st["lat"], st["lon"])
        w = waves.sel(latitude=wlat, longitude=wlon)
        e = era5.sel(latitude=st["lat"], longitude=st["lon"], method="nearest")
        df = pd.DataFrame(
            {
                "u10": e["u10"].to_series(),
                "v10": e["v10"].to_series(),
                "sst_k": e["sst"].to_series(),
                "hs": w["VHM0"].to_series(),
                "tp": w["VTPK"].to_series(),
            }
        )
        df.index.name = "time"
        df.dropna().to_csv(out / f"{sid}.csv")
        cells.append(
            {
                "station": sid,
                "requested_lat": st["lat"],
                "requested_lon": st["lon"],
                "wave_cell_lat": wlat,
                "wave_cell_lon": wlon,
            }
        )
        print(sid, "->", wlat, wlon)
    pd.DataFrame(cells).to_csv(out / "grid_cells.csv", index=False)


if __name__ == "__main__":
    main()
