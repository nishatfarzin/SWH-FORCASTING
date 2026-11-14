"""Turn the first-stage station files (P1_merged_raw.parquet ...) into the CSV layout used here.

Input columns : time, u10, v10, sst, VHM0, VTPK        (ERA5 wind and SST, CMEMS Hs and Tp)
Output columns: time, u10, v10, sst_k, hs, tp          (one CSV per station in --out)

Usage: python scripts/convert_thesis_parquet.py --src /content/drive/MyDrive/wave_thesis/data/processed --out data/stations
"""

import argparse
from pathlib import Path

import pandas as pd
from common import load_stations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/stations")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for sid in load_stations():
        df = pd.read_parquet(Path(args.src) / f"{sid}_merged_raw.parquet")
        df = df.rename(columns={"sst": "sst_k", "VHM0": "hs", "VTPK": "tp"})
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates("time")
        if df["sst_k"].median() < 100:  # already in degrees C
            df["sst_k"] = df["sst_k"] + 273.15
            print(f"{sid}: SST looked like degrees C, converted to K")
        full = pd.date_range(df["time"].min(), df["time"].max(), freq="3h")
        missing = len(full.difference(pd.DatetimeIndex(df["time"])))
        nans = int(df[["u10", "v10", "sst_k", "hs", "tp"]].isna().sum().sum())
        print(
            f"{sid}: {len(df)} rows, {df['time'].min()} to {df['time'].max()}, missing steps {missing}, NaN values {nans}"
        )
        df[["time", "u10", "v10", "sst_k", "hs", "tp"]].to_csv(
            out / f"{sid}.csv", index=False, float_format="%.5f"
        )


if __name__ == "__main__":
    main()
