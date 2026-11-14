"""Show which LSTM runs are finished and what is still missing for a planned design.

Usage: python scripts/lstm_status.py --lstm results/lstm --seeds 0 1 2
Prints a station x horizon x loss grid of finished seeds and the number of runs still to do.
"""

import argparse
from pathlib import Path

import pandas as pd
from common import load_params, load_stations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm", default="results/lstm")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--losses", nargs="*", default=["mse", "huber", "focal_mse", "focal_mse_norm", "target_weighted_mse"]
    )
    args = ap.parse_args()
    prm = load_params()
    horizons = prm["forecast"]["horizons_hours"]
    path = Path(args.lstm) / "lstm_metrics.csv"
    done = (
        pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["station", "horizon_h", "loss", "seed"])
    )
    done = done[done["seed"].isin(args.seeds) & done["loss"].isin(args.losses)]
    rows, missing = [], 0
    for sid in load_stations():
        for h in horizons:
            sub = done[(done.station == sid) & (done.horizon_h == h)]
            row = {"station": sid, "horizon_h": h}
            for loss in args.losses:
                n = int((sub.loss == loss).sum())
                row[loss] = f"{n}/{len(args.seeds)}"
                missing += len(args.seeds) - n
            rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nfinished: {len(done)}   still to do for seeds {args.seeds}: {missing}")


if __name__ == "__main__":
    main()
