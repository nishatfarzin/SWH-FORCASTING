"""Create synthetic data so the whole pipeline can be smoke-tested without the real files.

  python scripts/make_demo_data.py                       # five synthetic station CSVs in data/stations_demo
  python scripts/make_demo_data.py --fake-lstm           # also fake per-seed LSTM outputs (needs baselines run first)

The numbers mean nothing physically. Never use them for results.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from common import ROOT, load_stations, read_pred, save_pred

from waveshelf import metrics as mt
from waveshelf.synthetic import synthetic_series


def make_stations(out=ROOT / "data" / "stations_demo"):
    Path(out).mkdir(parents=True, exist_ok=True)
    for k, sid in enumerate(load_stations()):
        hs = synthetic_series(years=31, seed=k, base=1.4 - 0.15 * k)
        rng = np.random.default_rng(100 + k)
        n = len(hs)
        wind = np.clip(3 + 2.5 * hs.to_numpy() + rng.normal(0, 1.0, n), 0, None)
        ang = rng.uniform(0, 2 * np.pi, n)
        pd.DataFrame(
            {
                "time": hs.index,
                "u10": wind * np.cos(ang),
                "v10": wind * np.sin(ang),
                "sst_k": 301 + rng.normal(0, 0.5, n),
                "hs": hs.to_numpy(),
                "tp": 6 + 1.5 * hs.to_numpy() + rng.normal(0, 0.5, n),
            }
        ).to_csv(Path(out) / f"{sid}.csv", index=False)
    print("synthetic stations written to", out)


def fake_lstm(
    baselines=ROOT / "results_demo" / "baselines",
    out=ROOT / "results_demo" / "lstm",
    seeds=(0, 1, 2),
    losses=("mse", "huber", "focal_mse"),
):
    out = Path(out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in load_stations():
        for h in (6, 12, 24):
            base = read_pred(Path(baselines) / "predictions" / f"{sid}_h{h}")
            for loss in losses:
                for seed in seeds:
                    rng = np.random.default_rng(hash((sid, h, loss, seed)) % 2**32)
                    pred = base["ridge"].to_numpy() + rng.normal(
                        0, 0.03 + (0.02 if loss == "focal_mse" else 0), len(base)
                    )
                    save_pred(
                        pd.DataFrame({"time": base["time"], "obs": base["obs"], "pred": pred}),
                        out / "predictions" / f"{sid}_h{h}_{loss}_s{seed}",
                    )
                    rows.append(
                        {
                            "station": sid,
                            "horizon_h": h,
                            "loss": loss,
                            "seed": seed,
                            "epochs": 20,
                            "minutes": 0.0,
                            **mt.summary(base["obs"], pred),
                            **mt.upper_tail_scores(base["obs"], pred),
                        }
                    )
    pd.DataFrame(rows).to_csv(out / "lstm_metrics.csv", index=False)
    print("fake LSTM outputs written to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake-lstm", action="store_true")
    args = ap.parse_args()
    if args.fake_lstm:
        fake_lstm()
    else:
        make_stations()
