"""Figure 1: study area with bathymetry (GEBCO), the five stations and the tracks of the storms analysed.

Usage: python scripts/make_map_figure.py --gebco gebco_nbob.nc --tracks data/ibtracs_nio.csv --out figures
GEBCO: download the NetCDF for 17-23 N, 88-93 E from download.gebco.net (unzip it first; use the file
whose name does not contain 'tid' or 'sub_ice'). Only the coastline and depth contours are drawn, no national borders.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from common import load_stations

plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.dpi": 300})
STORM_SIDS = {  # IBTrACS ids of the storms used in the paper (see results/tables/storms_listed.csv)
    "Fani": "2019116N02090",
    "Bulbul": "2019302N11118",
    "Amphan": "2020136N10088",
    "Yaas": "2021143N15090",
    "Sitrang": "2022295N13093",
    "Mocha": "2023129N08091",
    "Midhili": "2023317N10094",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gebco", required=True)
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--out", default="figures")
    ap.add_argument(
        "--extent",
        nargs=4,
        type=float,
        default=[88.0, 93.0, 17.0, 23.0],
        metavar=("LON0", "LON1", "LAT0", "LAT1"),
    )
    args = ap.parse_args()
    lon0, lon1, lat0, lat1 = args.extent

    g = xr.open_dataset(args.gebco)
    g = g.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in g.dims})
    z = g["elevation"].sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
    depth = (-z).where(z < 0)

    fig, ax = plt.subplots(figsize=(5.6, 6.0))
    ax.set_facecolor("#d9d9d9")
    levels = [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000]
    cf = ax.contourf(z["lon"], z["lat"], depth, levels=levels, cmap="Blues", extend="max")
    ax.contour(z["lon"], z["lat"], z, levels=[0], colors="k", linewidths=0.6)
    cs = ax.contour(z["lon"], z["lat"], depth, levels=[10, 50, 200, 1000], colors="0.35", linewidths=0.4)
    ax.clabel(cs, fmt="%d m", fontsize=6)
    cb = fig.colorbar(cf, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Depth (m)")

    if args.tracks and Path(args.tracks).exists():
        tr = pd.read_csv(args.tracks, parse_dates=["time"])
        cmap = plt.get_cmap("tab10")
        for i, (name, sid) in enumerate(STORM_SIDS.items()):
            t = tr[tr["sid"] == sid].sort_values("time")
            if len(t):
                ax.plot(t["lon"], t["lat"], "-", lw=1.1, color=cmap(i), label=name)
        ax.legend(
            loc="lower right",
            fontsize=6.5,
            frameon=True,
            framealpha=0.85,
            title="Storm tracks",
            title_fontsize=7,
        )

    for sid, st in load_stations().items():
        ax.plot(st["wave_cell_lon"], st["wave_cell_lat"], "o", mfc="#d62728", mec="k", ms=6, zorder=5)
        ax.annotate(
            sid,
            (st["wave_cell_lon"], st["wave_cell_lat"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            weight="bold",
            zorder=6,
        )
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect(1 / np.cos(np.radians((lat0 + lat1) / 2)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "Figure_1.pdf", bbox_inches="tight")
    fig.savefig(out / "Figure_1.png", bbox_inches="tight")
    print("saved Figure_1")


if __name__ == "__main__":
    main()
