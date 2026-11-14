"""Train LSTM forecasters for several losses and random seeds (needs PyTorch, a GPU helps).

Usage examples
  python scripts/run_lstm.py --preset core  --out /content/drive/MyDrive/wave_paper/outputs/lstm
  python scripts/run_lstm.py --stations P1 --horizons 6 --losses huber --seeds 0      # one quick test run

Presets (losses x seeds):
  quick : mse, huber, focal_mse x 3 seeds
  core  : mse, huber, focal_mse, focal_mse_norm, target_weighted_mse x 5 seeds
  full  : all six losses x 10 seeds (as in config/params.yaml)

Runs that already exist in lstm_metrics.csv are skipped, so the job can be restarted after a disconnect.
--time-budget-min stops cleanly after that many minutes so you can run in several sittings.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from common import load_params, load_stations, read_station, save_pred

from waveshelf import features as ft
from waveshelf import metrics as mt
from waveshelf.lstm import predict, train_model

PRESETS = {
    "quick": (["mse", "huber", "focal_mse"], [0, 1, 2]),
    "core": (["mse", "huber", "focal_mse", "focal_mse_norm", "target_weighted_mse"], [0, 1, 2, 3, 4]),
    "full": (None, None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stations")
    ap.add_argument("--out", default="results/lstm")
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--horizons", nargs="*", type=int, default=None)
    ap.add_argument("--losses", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--time-budget-min", type=float, default=None)
    args = ap.parse_args()

    params = load_params()
    prm, dprm = params["forecast"], params["data"]
    losses, seeds = prm["losses"], prm["seeds"]
    if args.preset and PRESETS[args.preset][0]:
        losses, seeds = PRESETS[args.preset]
    losses = args.losses or losses
    seeds = args.seeds if args.seeds is not None else seeds
    horizons = args.horizons or prm["horizons_hours"]
    batch_size = args.batch_size or prm["batch_size"]

    out = Path(args.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "history").mkdir(parents=True, exist_ok=True)
    metrics_path = out / "lstm_metrics.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    done = {(r["station"], int(r["horizon_h"]), r["loss"], int(r["seed"])) for r in rows}

    stations = args.stations or list(load_stations())
    total = len(stations) * len(horizons) * len(losses) * len(seeds)
    t0 = time.time()
    n_new = 0
    print(f"{total} runs planned, {len(done)} already finished")

    for sid in stations:
        feats = ft.build_features(read_station(Path(args.data) / f"{sid}.csv"))
        train, val, test = ft.chronological_split(feats, dprm["train_end_year"], dprm["val_end_year"])
        scaler = ft.fit_scaler(train)
        for h in horizons:
            steps = h // dprm["step_hours"]
            Xtr, ytr, _, _ = ft.make_windows(train, scaler, prm["window_steps"], steps)
            Xva, yva, _, _ = ft.make_windows(val, scaler, prm["window_steps"], steps)
            Xte, yte, tte, obs = ft.make_windows(test, scaler, prm["window_steps"], steps)
            for loss in losses:
                for seed in seeds:
                    if (sid, h, loss, seed) in done:
                        continue
                    if args.time_budget_min and (time.time() - t0) / 60 > args.time_budget_min:
                        print(
                            f"time budget reached after {n_new} new runs; run the same command again to continue"
                        )
                        return
                    t1 = time.time()
                    model, hist = train_model(
                        Xtr,
                        ytr,
                        Xva,
                        yva,
                        loss,
                        seed,
                        max_epochs=prm["max_epochs"],
                        batch_size=batch_size,
                        lr=prm["lr"],
                    )
                    pred = ft.inverse_hs(scaler, predict(model, Xte))
                    rows.append(
                        {
                            "station": sid,
                            "horizon_h": h,
                            "loss": loss,
                            "seed": seed,
                            "epochs": len(hist),
                            "minutes": round((time.time() - t1) / 60, 2),
                            **mt.summary(obs, pred),
                            **mt.upper_tail_scores(obs, pred),
                        }
                    )
                    stem = f"{sid}_h{h}_{loss}_s{seed}"
                    save_pred(
                        pd.DataFrame({"time": tte, "obs": obs, "pred": pred}), out / "predictions" / stem
                    )
                    (out / "history" / f"{stem}.json").write_text(json.dumps(hist))
                    pd.DataFrame(rows).to_csv(metrics_path, index=False)
                    n_new += 1
                    print(
                        f"[{len(done) + n_new}/{total}] {stem}  rmse={rows[-1]['rmse']:.4f}  "
                        f"epochs={len(hist)}  {rows[-1]['minutes']} min"
                    )


if __name__ == "__main__":
    main()
