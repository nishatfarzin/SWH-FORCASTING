"""Draw the data-driven figures from the CSV results (Figs 2, 3, 4, 5 and 7).

Fig 1 (map) and Fig 6 (altimetry scatter) need data that is not produced by the scripts here
(bathymetry/coastline and the altimeter match-ups) and are made separately.
Usage: python scripts/make_figures.py --results results --out figures
Each figure is saved as PDF (vector, for submission) and PNG (300 dpi, for checking).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import load_params, load_stations, read_pred
from scipy import stats

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "figure.dpi": 120,
    }
)
COLORS = {
    "persistence": "#9e9e9e",
    "ridge": "#8c6d31",
    "xgboost": "#5b8c5a",
    "lstm_huber": "#1f77b4",
    "lstm_focal_mse": "#d62728",
    "lstm_mse": "#7f7f7f",
    "lstm_focal_mse_norm": "#ff9896",
    "lstm_target_weighted_mse": "#9467bd",
}


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig2_rmse(res, out):
    f = res / "tables" / "table2_forecast_skill.csv"
    if not f.exists():
        return print("skip Fig 2 (no table2_forecast_skill.csv)")
    t = pd.read_csv(f)
    models = [
        m
        for m in ["persistence", "ridge", "xgboost", "lstm_mse", "lstm_huber", "lstm_focal_mse"]
        if m in set(t["model"])
    ]
    stations = list(load_stations())
    horizons = sorted(t["horizon_h"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(2.9 * len(horizons), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    w = 0.8 / len(models)
    for ax, h in zip(axes, horizons):
        for i, m in enumerate(models):
            sub = t[(t.horizon_h == h) & (t.model == m)].set_index("station").reindex(stations)
            ax.bar(
                np.arange(len(stations)) + i * w,
                sub["rmse_mean"],
                w,
                yerr=sub["rmse_std"],
                color=COLORS.get(m, "k"),
                label=m.replace("lstm_", "LSTM ").replace("_", "-"),
                error_kw={"lw": 0.7, "capsize": 1.5},
            )
        ax.set_xticks(np.arange(len(stations)) + w * (len(models) - 1) / 2)
        ax.set_xticklabels(stations)
        ax.set_title(f"{h} h ahead")
    axes[0].set_ylabel("Test RMSE (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(models),
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, 1.08),
    )
    save(fig, out, "Figure_2")


def fig3_storms(res, lstm_dir, out, storm_names, station, horizon, seeds=None):
    prm = load_params()
    storms = prm["storms_in_test_period"]
    lm_path = Path(lstm_dir) / "lstm_metrics.csv"
    if not lm_path.exists():
        return print("skip Fig 3 (no lstm_metrics.csv)")
    lm = pd.read_csv(lm_path)
    if seeds is not None:
        lm = lm[lm["seed"].isin(seeds)]
    fig, axes = plt.subplots(1, len(storm_names), figsize=(3.4 * len(storm_names), 2.8), squeeze=False)
    for ax, name in zip(axes[0], storm_names):
        a, b = storms[name]
        lo, hi = pd.Timestamp(a) - pd.Timedelta(days=2), pd.Timestamp(b) + pd.Timedelta(days=3)
        obs_done = False
        for loss in ("huber", "focal_mse"):
            seeds = sorted(
                lm[(lm.station == station) & (lm.horizon_h == horizon) & (lm.loss == loss)]["seed"]
            )
            frames = [
                read_pred(Path(lstm_dir) / "predictions" / f"{station}_h{horizon}_{loss}_s{s}") for s in seeds
            ]
            frames = [f for f in frames if f is not None]
            if not frames:
                continue
            t = frames[0]["time"]
            m = (t >= lo) & (t <= hi)
            P = np.vstack([f["pred"].to_numpy() for f in frames])[:, m.to_numpy()]
            if not obs_done:
                ax.plot(t[m], frames[0]["obs"][m], "k", lw=1.4, label="Reanalysis")
                obs_done = True
            c = COLORS[f"lstm_{loss}"]
            ax.plot(t[m], P.mean(axis=0), color=c, lw=1.0, label=f"LSTM {loss.replace('_', '-')} (seed mean)")
            ax.fill_between(t[m], P.min(axis=0), P.max(axis=0), color=c, alpha=0.2, lw=0)
        ax.set_title(f"{name}, {station}, {horizon} h")
        ax.set_ylabel("$H_s$ (m)")
        ax.tick_params(axis="x", rotation=30)
    axes[0][0].legend(fontsize=6.5, frameon=False)
    save(fig, out, "Figure_3")


def fig4_return_levels(res, out):
    f = res / "evt" / "evt_summary.csv"
    if not f.exists():
        return print("skip Fig 4 (no evt_summary.csv)")
    s = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = np.arange(len(s))
    ax.bar(
        x - 0.27,
        s["rl100"],
        0.26,
        yerr=[s["rl100"] - s["rl100_boot_lo"], s["rl100_boot_hi"] - s["rl100"]],
        color="#1f77b4",
        label="POT-GPD, pooled",
        error_kw={"lw": 0.8, "capsize": 2},
    )
    if "rl100_combined" in s:
        ax.bar(x, s["rl100_combined"], 0.26, color="#d62728", label="POT-GPD, cyclone + other")
    ax.bar(
        x + 0.27,
        s["gev_rl100"],
        0.26,
        yerr=[s["gev_rl100"] - s["gev_rl100_lo"], s["gev_rl100_hi"] - s["gev_rl100"]],
        color="#7f7f7f",
        label="GEV, annual maxima",
        error_kw={"lw": 0.8, "capsize": 2},
    )
    if "record_max" in s:
        ax.plot(x, s["record_max"], "k_", ms=14, mew=1.6, label="1993-2023 maximum")
    ax.set_xticks(x)
    ax.set_xticklabels(s["station"])
    ax.set_ylabel("100-year $H_s$ (m)")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    save(fig, out, "Figure_4")


def fig5_diagnostics(res, data_dir, out, stations=("P1", "P5")):
    evt_dir = res / "evt"
    summ = evt_dir / "evt_summary.csv"
    if not summ.exists():
        return print("skip Fig 5 (no EVT output)")
    s = pd.read_csv(summ).set_index("station")
    fig, axes = plt.subplots(len(stations), 3, figsize=(8.2, 2.5 * len(stations)), squeeze=False)
    for r, sid in enumerate(stations):
        mrl, stab = pd.read_csv(evt_dir / f"{sid}_mrl.csv"), pd.read_csv(evt_dir / f"{sid}_stability.csv")
        a = axes[r][0]
        a.plot(mrl["u"], mrl["mean_excess"], "k", lw=1)
        a.fill_between(mrl["u"], mrl["lo"], mrl["hi"], color="k", alpha=0.15, lw=0)
        a.set_xlabel("Threshold (m)")
        a.set_ylabel("Mean excess (m)")
        a.set_title(f"{sid}: mean residual life")
        b = axes[r][1]
        b.errorbar(stab["u"], stab["xi"], yerr=1.96 * stab["se_xi"], fmt="o-", ms=3, lw=0.8, color="#1f77b4")
        b.axvline(s.loc[sid, "u"], color="#d62728", ls="--", lw=0.8)
        b.axhline(0, color="k", lw=0.5)
        b.set_xlabel("Threshold (m)")
        b.set_ylabel(r"Shape $\xi$")
        b.set_title("Parameter stability")
        c = axes[r][2]
        peaks = pd.read_csv(evt_dir / f"{sid}_peaks.csv")
        u, xi, sigma = s.loc[sid, "u"], s.loc[sid, "xi"], s.loc[sid, "sigma"]
        x = np.sort(peaks["hs"].to_numpy() - u)
        p = (np.arange(1, len(x) + 1) - 0.5) / len(x)
        model = stats.genpareto.ppf(p, xi, scale=sigma)
        c.plot(model, x, "o", ms=2.5, color="#1f77b4")
        lim = [0, max(model.max(), x.max()) * 1.05]
        c.plot(lim, lim, "k", lw=0.7)
        c.set_xlabel("GPD quantile (m)")
        c.set_ylabel("Observed excess (m)")
        c.set_title("QQ plot")
    fig.tight_layout()
    save(fig, out, "Figure_5")


def fig7_synthesis(res, out):
    st = load_stations()
    depth = pd.Series({k: v.get("depth_plot_m", np.nan) for k, v in st.items()})
    t2f, evf, stf = (
        res / "tables" / "table2_forecast_skill.csv",
        res / "evt" / "evt_summary.csv",
        res / "storms" / "storm_summary.csv",
    )
    if not (t2f.exists() and evf.exists()):
        return print("skip Fig 7 (need table2 and evt_summary)")
    t2 = pd.read_csv(t2f)
    ev = pd.read_csv(evf).set_index("station")
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.7))
    h = max(t2["horizon_h"])
    sub = t2[(t2.horizon_h == h) & (t2.model == "lstm_huber")].set_index("station").reindex(list(st))
    axes[0].errorbar(depth, sub["si_mean"], yerr=sub["si_std"], fmt="o-", color="#1f77b4", capsize=2)
    axes[0].set_ylabel(f"Scatter index ({h} h, Huber)")
    axes[0].set_title("(a)", loc="left")
    if stf.exists():
        ss = pd.read_csv(stf)
        name = sorted(ss["storm"].unique())[0]
        sp = (
            ss[(ss.storm == name) & (ss.horizon_h == h) & (ss.model == "lstm_huber")]
            .set_index("station")
            .reindex(list(st))
        )
        axes[1].plot(depth, 100 * sp["peak_error_mean"] / sp["obs_peak"], "o-", color="#d62728")
        axes[1].set_ylabel(f"Peak underprediction (%), {name}")
        axes[1].set_title("(b)", loc="left")
    axes[2].errorbar(
        depth,
        ev["rl100"].reindex(list(st)),
        yerr=[ev["rl100"] - ev["rl100_boot_lo"], ev["rl100_boot_hi"] - ev["rl100"]],
        fmt="o-",
        color="k",
        capsize=2,
    )
    axes[2].set_ylabel("100-year $H_s$ (m)")
    axes[2].set_title("(c)", loc="left")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Model-cell depth (m)")
        ax.invert_xaxis()
        for sid in st:
            off = {"P5": (7, -11), "P4": (-7, 7)}.get(sid, (0, 7))
            ax.annotate(
                sid,
                (depth[sid], ax.get_lines()[0].get_ydata()[list(st).index(sid)] if ax.get_lines() else 0),
                textcoords="offset points",
                xytext=off,
                ha="center",
                fontsize=7,
                color="0.35",
            )
    fig.tight_layout()
    save(fig, out, "Figure_7")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--storms", nargs="*", default=["Amphan", "Mocha"])
    ap.add_argument("--storm-station", default="P1")
    ap.add_argument("--storm-horizon", type=int, default=24)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    fig2_rmse(res, out)
    fig3_storms(res, res / "lstm", out, args.storms, args.storm_station, args.storm_horizon, args.seeds)
    fig4_return_levels(res, out)
    fig5_diagnostics(res, None, out)
    fig7_synthesis(res, out)


if __name__ == "__main__":
    main()
