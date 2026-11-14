"""Relative depth, linear shoaling and Hs/depth ratios for the five stations.

Combines the station statistics (mean peak period), the GEBCO depths in config/stations.yaml and the
100-year return levels from the EVT run. Writes results/tables/depth_diagnostics.csv.

Usage: python scripts/depth_diagnostics.py --stats results/tables/station_descriptive_stats.csv \
          --evt results/evt/evt_summary.csv --out results/tables
"""

import argparse
from pathlib import Path

import pandas as pd
from common import load_stations

from waveshelf import dispersion as wd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="results/tables/station_descriptive_stats.csv")
    ap.add_argument("--evt", default="results/evt/evt_summary.csv")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    stats = pd.read_csv(args.stats).set_index("station")
    evt = pd.read_csv(args.evt).set_index("station")
    rows = []
    for sid, st in load_stations().items():
        tp = float(stats.loc[sid, "tp_mean"])
        rec = {"station": sid, "tp_mean_s": tp, "deep_water_wavelength_m": wd.deep_water_wavelength(tp)}
        for tag, depth in (("mean", st["cell_mean_depth_m"]), ("min", st["cell_min_depth_m"])):
            rec[f"depth_{tag}_m"] = depth
            rec[f"h_over_L0_{tag}"] = wd.relative_depth(tp, depth)
            rec[f"Ks_{tag}"] = wd.shoaling_coefficient(tp, depth)
            rec[f"hs100_over_depth_{tag}"] = float(evt.loc[sid, "rl100"]) / depth
        rows.append(rec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "depth_diagnostics.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
