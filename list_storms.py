"""List storms that came within a distance of the stations, to choose and check the storm windows.

Usage: python scripts/list_storms.py --tracks data/ibtracs_nio.csv --from 2019 --to 2023 --max-km 800
Prints one line per storm: name, dates and the closest approach (km) to every station.
Use the dates to fill `storms_in_test_period` in config/params.yaml.
"""

import argparse
from pathlib import Path

import pandas as pd
from common import load_stations

from waveshelf.events import haversine_km


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="data/ibtracs_nio.csv")
    ap.add_argument("--from", dest="y0", type=int, default=2019)
    ap.add_argument("--to", dest="y1", type=int, default=2023)
    ap.add_argument("--max-km", type=float, default=800.0)
    ap.add_argument("--out", default="results/tables/storms_listed.csv")
    args = ap.parse_args()

    tr = pd.read_csv(args.tracks, parse_dates=["time"])
    tr = tr[(tr["time"].dt.year >= args.y0) & (tr["time"].dt.year <= args.y1)]
    stations = load_stations()
    rows = []
    for sid, g in tr.groupby("sid"):
        rec = {
            "sid": sid,
            "name": g["name"].iloc[0] if "name" in g else "",
            "start": g["time"].min().date(),
            "end": g["time"].max().date(),
        }
        if "wmo_wind" in g:
            rec["max_wind_kt"] = g["wmo_wind"].max()
        close = []
        for st, info in stations.items():
            d = haversine_km(info["lat"], info["lon"], g["lat"].to_numpy(), g["lon"].to_numpy())
            rec[f"min_km_{st}"] = round(float(d.min()))
            close.append(d.min())
        if min(close) <= args.max_km:
            rows.append(rec)
    out = pd.DataFrame(rows).sort_values("start")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
