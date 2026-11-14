"""Download the IBTrACS North Indian Ocean file and keep the columns used here.

Output: data/ibtracs_nio.csv with sid, name, time, lat, lon (and wmo_wind if present), 1993-2023.
If the download fails (no internet, URL moved), download ibtracs.NI.list.v04r01.csv by hand from
https://www.ncei.noaa.gov/products/international-best-track-archive and pass it with --csv.
"""

import argparse
from pathlib import Path

import pandas as pd

URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=URL, help="URL or local path of the IBTrACS NI csv")
    ap.add_argument("--out", default="data/ibtracs_nio.csv")
    args = ap.parse_args()

    want = {"SID", "NAME", "ISO_TIME", "LAT", "LON", "WMO_WIND"}
    df = pd.read_csv(args.csv, skiprows=[1], usecols=lambda c: c in want, low_memory=False)
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    for col in ("LAT", "LON", "WMO_WIND"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ISO_TIME", "LAT", "LON"])
    df = df[(df["ISO_TIME"] >= "1993-01-01") & (df["ISO_TIME"] < "2024-01-01")]
    df = df.rename(
        columns={
            "SID": "sid",
            "NAME": "name",
            "ISO_TIME": "time",
            "LAT": "lat",
            "LON": "lon",
            "WMO_WIND": "wmo_wind",
        }
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{df['sid'].nunique()} storms, {len(df)} track points -> {args.out}")


if __name__ == "__main__":
    main()
