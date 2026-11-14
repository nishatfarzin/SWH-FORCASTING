"""Data audit: continuity, descriptive statistics and the grid cells actually used.

Usage: python scripts/audit_stations.py --data data/stations --out results/tables \
          [--wave-nc /content/drive/MyDrive/wave_cmems_nbob.nc] [--wind-nc <one ERA5 wind file>]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from common import load_stations, read_station


def wind_from_direction(u, v):
    """Meteorological direction the wind blows FROM, degrees clockwise from north; vector mean."""
    return float((np.degrees(np.arctan2(-u.mean(), -v.mean())) + 360) % 360)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stations")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--wave-nc", default=None)
    ap.add_argument("--wind-nc", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stations = load_stations()

    rows = []
    for sid in stations:
        df = read_station(Path(args.data) / f"{sid}.csv")
        steps = pd.Series(df.index).diff().dropna()
        gaps = int((steps != pd.Timedelta(hours=3)).sum())
        speed = np.hypot(df["u10"], df["v10"])
        rows.append(
            {
                "station": sid,
                "start": df.index[0],
                "end": df.index[-1],
                "n": len(df),
                "time_gaps": gaps,
                "hs_mean": df["hs"].mean(),
                "hs_p90": df["hs"].quantile(0.90),
                "hs_p99": df["hs"].quantile(0.99),
                "hs_max": df["hs"].max(),
                "tp_mean": df["tp"].mean(),
                "wind_mean_ms": speed.mean(),
                "wind_p99_ms": speed.quantile(0.99),
                "wind_from_deg": wind_from_direction(df["u10"], df["v10"]),
            }
        )
    table = pd.DataFrame(rows)
    num = table.select_dtypes("number").columns
    table[num] = table[num].round(3)
    table.to_csv(out / "station_descriptive_stats.csv", index=False)
    print(table.to_string(index=False))

    if args.wave_nc or args.wind_nc:
        import xarray as xr

        cells = []
        for sid, st in stations.items():
            rec = {"station": sid, "requested_lat": st["lat"], "requested_lon": st["lon"]}
            for tag, path in (("wave", args.wave_nc), ("era5", args.wind_nc)):
                if not path:
                    continue
                ds = xr.open_dataset(path)
                pt = ds.sel(latitude=st["lat"], longitude=st["lon"], method="nearest")
                rec[f"{tag}_cell_lat"] = float(pt["latitude"])
                rec[f"{tag}_cell_lon"] = float(pt["longitude"])
                ds.close()
            cells.append(rec)
        cells = pd.DataFrame(cells)
        cells.to_csv(out / "grid_cells.csv", index=False)
        print(cells.to_string(index=False))


if __name__ == "__main__":
    main()
