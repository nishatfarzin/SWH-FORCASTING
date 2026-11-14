"""Seed-averaged forecast tables and paired tests.

Writes into --out:
  table2_forecast_skill.csv  mean and SD over seeds for every station, horizon and model
  loss_pairs.csv             Huber vs every other loss: mean RMSE difference, seed wins, DM and block-bootstrap results
  vs_baselines.csv           LSTM-Huber vs persistence, ridge and XGBoost: skill score and paired tests
  loss_pair_counts.csv       how many of the station-horizon pairs each comparison won
Usage: python scripts/aggregate_forecast.py --baselines results/baselines --lstm results/lstm --out results/tables
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from common import load_params, load_stations, read_pred

from waveshelf import metrics as mt


def load_seed_stack(lstm_dir, sid, h, loss, seeds):
    """Return times, obs and an array (n_seeds, n_time) of predictions."""
    preds, obs, times = [], None, None
    for s in seeds:
        fr = read_pred(Path(lstm_dir) / "predictions" / f"{sid}_h{h}_{loss}_s{s}")
        if fr is None:
            continue
        preds.append(fr["pred"].to_numpy())
        obs, times = fr["obs"].to_numpy(), fr["time"]
    return times, obs, (np.vstack(preds) if preds else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="results/baselines")
    ap.add_argument("--lstm", default="results/lstm")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--reference-loss", default="huber")
    ap.add_argument(
        "--seeds", nargs="*", type=int, default=None, help="use only these seeds (same set for all stations)"
    )
    args = ap.parse_args()
    prm = load_params()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step_h = prm["data"]["step_hours"]
    block = prm["forecast"]["block_length_steps"]

    lm = pd.read_csv(Path(args.lstm) / "lstm_metrics.csv")
    if args.seeds is not None:
        lm = lm[lm["seed"].isin(args.seeds)]
    bm = pd.read_csv(Path(args.baselines) / "baseline_metrics.csv")
    cols = ["rmse", "mae", "r2", "si", "bias", "rmse_p95", "rmse_p99", "bias_p95", "bias_p99"]

    # ---- Table 2: mean and SD over seeds
    g = lm.groupby(["station", "horizon_h", "loss"])[cols].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index().rename(columns={"loss": "model"})
    g["model"] = "lstm_" + g["model"]
    g["n_seeds"] = lm.groupby(["station", "horizon_h", "loss"]).size().to_numpy()
    base = bm.rename(columns={c: f"{c}_mean" for c in cols})
    for c in cols:
        base[f"{c}_std"] = np.nan
    base["n_seeds"] = 1
    t2 = pd.concat(
        [
            base[
                ["station", "horizon_h", "model"]
                + [f"{c}_{s}" for c in cols for s in ("mean", "std")]
                + ["n_seeds"]
            ],
            g,
        ],
        ignore_index=True,
    )
    t2.sort_values(["station", "horizon_h", "model"]).round(4).to_csv(
        out / "table2_forecast_skill.csv", index=False
    )

    # ---- paired comparisons between losses
    ref = args.reference_loss
    pair_rows = []
    losses = sorted(set(lm["loss"]) - {ref})
    for sid in load_stations():
        for h in prm["forecast"]["horizons_hours"]:
            seeds_ref = sorted(lm[(lm.station == sid) & (lm.horizon_h == h) & (lm.loss == ref)]["seed"])
            t_a, obs, pa = load_seed_stack(args.lstm, sid, h, ref, seeds_ref)
            if pa is None:
                continue
            for other in losses:
                seeds_o = sorted(lm[(lm.station == sid) & (lm.horizon_h == h) & (lm.loss == other)]["seed"])
                common = sorted(set(seeds_ref) & set(seeds_o))
                if not common:
                    continue
                _, _, pa_c = load_seed_stack(args.lstm, sid, h, ref, common)
                _, _, pb_c = load_seed_stack(args.lstm, sid, h, other, common)
                rmse_a = np.sqrt(((pa_c - obs) ** 2).mean(axis=1))
                rmse_b = np.sqrt(((pb_c - obs) ** 2).mean(axis=1))
                d = (((pa_c - obs) ** 2) - ((pb_c - obs) ** 2)).mean(
                    axis=0
                )  # seed-averaged loss differential
                tst = mt.paired_tests_from_diff(d, horizon_steps=h // step_h, block_len=block)
                pair_rows.append(
                    {
                        "station": sid,
                        "horizon_h": h,
                        "ref": ref,
                        "other": other,
                        "n_seeds": len(common),
                        "rmse_ref": rmse_a.mean(),
                        "rmse_other": rmse_b.mean(),
                        "rmse_diff_ref_minus_other": (rmse_a - rmse_b).mean(),
                        "seeds_ref_better": int((rmse_a < rmse_b).sum()),
                        "dm_stat": tst["stat"],
                        "dm_p": tst["p"],
                        "boot_lo": tst["ci95"][0],
                        "boot_hi": tst["ci95"][1],
                    }
                )
    pairs = pd.DataFrame(pair_rows)
    pairs.round(5).to_csv(out / "loss_pairs.csv", index=False)
    if len(pairs):
        counts = (
            pairs.groupby("other")
            .agg(
                pairs=("station", "size"),
                ref_better=("rmse_diff_ref_minus_other", lambda x: int((x < 0).sum())),
                ref_better_p05=("dm_p", lambda x: int(((x < 0.05)).sum())),
            )
            .reset_index()
        )
        counts.to_csv(out / "loss_pair_counts.csv", index=False)
        print(counts.to_string(index=False))

    # ---- LSTM vs baselines
    vb = []
    for sid in load_stations():
        for h in prm["forecast"]["horizons_hours"]:
            bp = read_pred(Path(args.baselines) / "predictions" / f"{sid}_h{h}")
            seeds_ref = sorted(lm[(lm.station == sid) & (lm.horizon_h == h) & (lm.loss == ref)]["seed"])
            if bp is None or not seeds_ref:
                continue
            _, obs, pa = load_seed_stack(args.lstm, sid, h, ref, seeds_ref)
            for col in [c for c in bp.columns if c not in ("time", "obs")]:
                pred_b = bp[col].to_numpy()
                if len(pred_b) != len(obs):
                    continue
                d = (((pa - obs) ** 2).mean(axis=0)) - (pred_b - obs) ** 2
                tst = mt.paired_tests_from_diff(d, horizon_steps=h // step_h, block_len=block)
                vb.append(
                    {
                        "station": sid,
                        "horizon_h": h,
                        "baseline": col,
                        "rmse_lstm": float(np.sqrt(((pa - obs) ** 2).mean())),
                        "rmse_baseline": mt.rmse(obs, pred_b),
                        "mse_skill": 1 - float(((pa - obs) ** 2).mean() / ((pred_b - obs) ** 2).mean()),
                        "dm_stat": tst["stat"],
                        "dm_p": tst["p"],
                        "boot_lo": tst["ci95"][0],
                        "boot_hi": tst["ci95"][1],
                    }
                )
    pd.DataFrame(vb).round(5).to_csv(out / "vs_baselines.csv", index=False)
    print(f"tables written to {out}")


if __name__ == "__main__":
    main()
