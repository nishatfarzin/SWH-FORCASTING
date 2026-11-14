"""Figure 1 (no bathymetry file needed): coastline, the five wave-model cells, station depths and storm tracks.

Needs the optional packages `basemap` and `basemap-data-hires` (pip install basemap basemap-data-hires).
Usage: python scripts/make_map_coastline.py --tracks data/ibtracs_nio.csv --out figures
Depths are taken from config/stations.yaml (GEBCO cell means). For a version with depth contours use make_map_figure.py.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from common import load_stations
from mpl_toolkits.basemap import Basemap

plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.dpi": 300})
STORM_SIDS = {
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
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--out", default="figures")
    ap.add_argument(
        "--extent",
        nargs=4,
        type=float,
        default=[87.5, 93.5, 16.5, 23.5],
        metavar=("LON0", "LON1", "LAT0", "LAT1"),
    )
    args = ap.parse_args()
    lon0, lon1, lat0, lat1 = args.extent

    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    m = Basemap(
        llcrnrlon=lon0,
        llcrnrlat=lat0,
        urcrnrlon=lon1,
        urcrnrlat=lat1,
        resolution="h",
        projection="cyl",
        ax=ax,
    )
    m.drawmapboundary(fill_color="#dbe9f6")
    m.fillcontinents(color="#e9e4d4", lake_color="#dbe9f6", zorder=1)
    m.drawcoastlines(linewidth=0.5, color="0.25", zorder=2)
    m.drawparallels(range(17, 24), labels=[1, 0, 0, 0], linewidth=0.25, color="0.6", fontsize=8)
    m.drawmeridians(range(88, 94), labels=[0, 0, 0, 1], linewidth=0.25, color="0.6", fontsize=8)

    if args.tracks and Path(args.tracks).exists():
        tr = pd.read_csv(args.tracks, parse_dates=["time"])
        cmap = plt.get_cmap("tab10")
        for i, (name, sid) in enumerate(STORM_SIDS.items()):
            t = tr[tr["sid"] == sid].sort_values("time")
            if len(t):
                ax.plot(
                    t["lon"],
                    t["lat"],
                    "-",
                    lw=1.2,
                    color=cmap(i),
                    label=f"{name} ({t['time'].iloc[0].year})",
                    zorder=3,
                )
        ax.legend(
            loc="lower right",
            fontsize=6.8,
            frameon=True,
            framealpha=0.9,
            title="Storm tracks",
            title_fontsize=7.5,
        )

    for sid, st in load_stations().items():
        ax.plot(st["wave_cell_lon"], st["wave_cell_lat"], "o", mfc="#d62728", mec="k", ms=6.5, zorder=6)
        depth = st["cell_mean_depth_m"]
        label = f"{sid} ({depth:,.0f} m)"
        dx, dy = (-8, 6) if sid == "P5" else ((9, 5) if sid == "P4" else (7, 4))
        ax.annotate(
            label,
            (st["wave_cell_lon"], st["wave_cell_lat"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            weight="bold",
            ha="right" if dx < 0 else "left",
            zorder=7,
        )
    ax.set_xlabel("Longitude (°E)", labelpad=14)
    ax.set_ylabel("Latitude (°N)", labelpad=24)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "Figure_1.pdf", bbox_inches="tight")
    fig.savefig(out / "Figure_1.png", bbox_inches="tight")
    print("saved Figure_1")


if __name__ == "__main__":
    main()
