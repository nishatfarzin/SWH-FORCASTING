"""Persistence, ridge and XGBoost forecasts for every station and horizon.

Usage: python scripts/run_baselines.py --data data/stations --out results/baselines
Expects data/stations/<ID>.csv with columns time,u10,v10,sst_k,hs,tp.
"""

import argparse
from pathlib import Path

import pandas as pd
from common import load_params, load_stations, read_station, save_pred

from waveshelf import baselines as bl
from waveshelf import features as ft
from waveshelf import metrics as mt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stations")
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--skip-xgb", action="store_true")
    args = ap.parse_args()

    prm = load_params()
    out = Path(args.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in args.stations or load_stations():
        feats = ft.build_features(read_station(Path(args.data) / f"{sid}.csv"))
        train, val, test = ft.chronological_split(
            feats, prm["data"]["train_end_year"], prm["data"]["val_end_year"]
        )
        scaler = ft.fit_scaler(train)
        for h in prm["forecast"]["horizons_hours"]:
            steps = h // prm["data"]["step_hours"]
            w = prm["forecast"]["window_steps"]
            Xtr, ytr, _, _ = ft.make_windows(train, scaler, w, steps)
            Xva, yva, _, _ = ft.make_windows(val, scaler, w, steps)
            Xte, yte, tte, obs = ft.make_windows(test, scaler, w, steps)
            preds = {"persistence": bl.persistence(Xte)}
            preds["ridge"] = bl.predict_ridge(bl.fit_ridge(Xtr, ytr), Xte)
            if not args.skip_xgb:
                preds["xgboost"] = bl.predict_xgb(bl.fit_xgb(Xtr, ytr, Xva, yva), Xte)
            frame = pd.DataFrame({"time": tte, "obs": obs})
            for name, p in preds.items():
                phys = ft.inverse_hs(scaler, p)
                frame[name] = phys
                rows.append(
                    {
                        "station": sid,
                        "horizon_h": h,
                        "model": name,
                        **mt.summary(obs, phys),
                        **mt.upper_tail_scores(obs, phys),
                    }
                )
            save_pred(frame, out / "predictions" / f"{sid}_h{h}")
            print(f"{sid} {h} h done ({len(obs)} test samples)")
    pd.DataFrame(rows).to_csv(out / "baseline_metrics.csv", index=False)


if __name__ == "__main__":
    main()
