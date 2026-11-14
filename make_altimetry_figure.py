"""Figure 6: altimeter vs reanalysis Hs at the five stations (pass medians), with the statistics in each panel.

Reads <dir>/<station>_matched_passes.parquet written by scripts/altimetry_matchup.py
(columns altimetry_swh, reanalysis_hs). Usage: python scripts/make_altimetry_figure.py --passes <dir> --out figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import load_stations

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", required=True)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    stations = list(load_stations())
    fig, axes = plt.subplots(1, len(stations), figsize=(2.5 * len(stations), 2.7), sharex=True, sharey=True)
    for ax, sid in zip(axes, stations):
        df = pd.read_parquet(Path(args.passes) / f"{sid}_matched_passes.parquet")
        x, y = df["reanalysis_hs"].to_numpy(), df["altimetry_swh"].to_numpy()
        r = np.corrcoef(x, y)[0, 1]
        resid = y - x
        ax.plot(x, y, ".", ms=2.5, alpha=0.5, color="#1f77b4")
        lim = [0, max(x.max(), y.max()) * 1.05]
        ax.plot(lim, lim, "k", lw=0.7)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(sid)
        ax.text(
            0.04,
            0.96,
            f"n = {len(df)}\nr = {r:.2f}\nbias = {resid.mean():+.2f} m\nRMSE = {np.sqrt((resid ** 2).mean()):.2f} m",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
        )
        ax.set_xlabel("Reanalysis $H_s$ (m)")
    axes[0].set_ylabel("Altimeter $H_s$ (m)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "Figure_6.pdf", bbox_inches="tight")
    fig.savefig(out / "Figure_6.png", bbox_inches="tight")
    print("saved Figure_6")


if __name__ == "__main__":
    main()
