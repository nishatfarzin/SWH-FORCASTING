"""Forecast skill inside named storm windows, for every baseline and every LSTM run.

Reads: results/baselines/predictions, results/lstm/predictions, windows in config/params.yaml.
Writes: storm_metrics.csv (one row per run), storm_summary.csv (mean and SD over seeds) and
storm_table3.csv (averaged over stations and horizons: the table used in the paper).
Usage: python scripts/storm_eval.py --baselines results/baselines --lstm results/lstm --out results/storms
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from common import load_params, load_stations, read_pred

from waveshelf import metrics as mt


def window_scores(frame, col, start, end):
    t = frame["time"]
    sel = frame[(t >= pd.Timestamp(start)) & (t < pd.Timestamp(end) + pd.Timedelta(days=1))]
    if len(sel) < 4:
        return None
    obs, pred = sel["obs"].to_numpy(), sel[col].to_numpy()
    mag, timing = mt.peak_error(obs, pred)
    return {
        "n": len(sel),
        "rmse": mt.rmse(obs, pred),
        "peak_error": mag,
        "timing_h": timing,
        "obs_peak": float(obs.max()),
        "pred_peak": float(obs.max() - mag),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="results/baselines")
    ap.add_argument("--lstm", default="results/lstm")
    ap.add_argument("--out", default="results/storms")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="use only these seeds")
    args = ap.parse_args()
    prm = load_params()
    storms = prm["storms_in_test_period"]
    horizons = prm["forecast"]["horizons_hours"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lstm_metrics = Path(args.lstm) / "lstm_metrics.csv"
    runs = (
        pd.read_csv(lstm_metrics)[["station", "horizon_h", "loss", "seed"]]
        if lstm_metrics.exists()
        else pd.DataFrame()
    )
    if args.seeds is not None and len(runs):
        runs = runs[runs["seed"].isin(args.seeds)]
    rows = []
    for sid in load_stations():
        for h in horizons:
            base = read_pred(Path(args.baselines) / "predictions" / f"{sid}_h{h}")
            sources = []
            if base is not None:
                for col in [c for c in base.columns if c not in ("time", "obs")]:
                    sources.append((base, col, col, np.nan))
            if len(runs):
                for _, r in runs[(runs.station == sid) & (runs.horizon_h == h)].iterrows():
                    fr = read_pred(
                        Path(args.lstm) / "predictions" / f"{sid}_h{h}_{r['loss']}_s{int(r['seed'])}"
                    )
                    if fr is not None:
                        sources.append((fr, "pred", f"lstm_{r['loss']}", int(r["seed"])))
            for name, (a, b) in storms.items():
                for frame, col, model, seed in sources:
                    sc = window_scores(frame, col, a, b)
                    if sc:
                        rows.append(
                            {
                                "storm": name,
                                "station": sid,
                                "horizon_h": h,
                                "model": model,
                                "seed": seed,
                                **sc,
                            }
                        )
    res = pd.DataFrame(rows)
    res.to_csv(out / "storm_metrics.csv", index=False)
    if res.empty:
        print("no storm windows overlap the predictions; check the dates in config/params.yaml")
        return
    keys = ["storm", "station", "horizon_h", "model"]
    summ = (
        res.groupby(keys)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            peak_error_mean=("peak_error", "mean"),
            peak_error_sd=("peak_error", "std"),
            timing_h=("timing_h", "mean"),
            obs_peak=("obs_peak", "first"),
            n_runs=("rmse", "size"),
        )
        .reset_index()
    )
    summ.to_csv(out / "storm_summary.csv", index=False)
    t3 = (
        summ.groupby(["storm", "model"])[["rmse_mean", "peak_error_mean", "timing_h"]]
        .mean()
        .round(3)
        .reset_index()
    )
    t3.to_csv(out / "storm_table3.csv", index=False)
    print(t3.to_string(index=False))


if __name__ == "__main__":
    main()
