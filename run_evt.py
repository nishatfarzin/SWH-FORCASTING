"""Extreme-value analysis of station Hs: diagnostics, sensitivity, return levels, cross-checks.

Usage:
  python scripts/run_evt.py --data data/stations --out results/evt [--tracks data/ibtracs_nio.csv]
The optional tracks file needs columns sid,time,lat,lon (a subset of IBTrACS, North Indian Ocean).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from common import load_params, load_stations, read_station

from waveshelf import events as ev
from waveshelf import evt


def fit_block(series, q, sep, T_list, years):
    u = float(series.quantile(q))
    peaks = evt.decluster_runs(series, u, sep)
    xi, sigma = evt.fit_gpd(peaks["hs"].to_numpy() - u)
    rate = len(peaks) / years
    rl = {T: float(evt.return_level(u, xi, sigma, rate, T)) for T in T_list}
    return u, peaks, xi, sigma, rate, rl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stations")
    ap.add_argument("--out", default="results/evt")
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--cyclone-km", type=float, default=None, help="override cyclone distance limit (km)")
    ap.add_argument("--cyclone-hours", type=float, default=None, help="override cyclone time window (h)")
    args = ap.parse_args()

    prm = load_params()["evt"]
    n_boot = args.n_boot or prm["n_boot"]
    km = args.cyclone_km or prm["cyclone_distance_km"]
    hrs = args.cyclone_hours or prm["cyclone_window_hours"]
    T_list = prm["return_periods"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stations = load_stations()
    tracks = pd.read_csv(args.tracks, parse_dates=["time"]) if args.tracks else None
    summary = []

    for sid in args.stations or stations:
        hs = read_station(Path(args.data) / f"{sid}.csv")["hs"]
        years = len(hs.index.year.unique())
        q0, sep0 = prm["threshold_quantile"], prm["separation_hours"]

        # 1. threshold diagnostics
        evt.mean_residual_life(hs.to_numpy(), np.quantile(hs, np.linspace(0.80, 0.99, 40))).to_csv(
            out / f"{sid}_mrl.csv", index=False
        )
        evt.parameter_stability(hs, np.linspace(0.88, 0.98, 21), sep0, years).to_csv(
            out / f"{sid}_stability.csv", index=False
        )

        # 2. sensitivity to threshold and declustering window
        sens = []
        for q in prm["quantile_grid"]:
            for sep in prm["separation_grid_hours"]:
                u, peaks, xi, sigma, rate, rl = fit_block(hs, q, sep, T_list, years)
                sens.append(
                    {
                        "quantile": q,
                        "separation_h": sep,
                        "u": u,
                        "n": len(peaks),
                        "rate": rate,
                        "xi": xi,
                        "sigma": sigma,
                        **{f"rl{T}": v for T, v in rl.items()},
                    }
                )
        pd.DataFrame(sens).to_csv(out / f"{sid}_sensitivity.csv", index=False)

        # 3. main fit with year-bootstrap and profile-likelihood intervals
        u, peaks, xi, sigma, rate, rl = fit_block(hs, q0, sep0, T_list, years)
        boot = evt.bootstrap_return_levels(peaks, u, years, T_list, n_boot=n_boot, seed=1)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        excess = peaks["hs"].to_numpy() - u
        xi_lo, xi_hi = evt.profile_ci_xi(excess)
        rl100_pl = evt.profile_ci_return_level(excess, u, rate, 100)
        rows = [{"T": T, "rl": rl[T], "boot_lo": lo[i], "boot_hi": hi[i]} for i, T in enumerate(T_list)]
        pd.DataFrame(rows).to_csv(out / f"{sid}_return_levels.csv", index=False)
        peaks.to_csv(out / f"{sid}_peaks.csv", index=False)

        rec = {
            "station": sid,
            "cyclone_km": km,
            "cyclone_hours": hrs,
            "quantile": q0,
            "u": u,
            "n_storms": len(peaks),
            "rate_per_year": rate,
            "xi": xi,
            "xi_lo": xi_lo,
            "xi_hi": xi_hi,
            "sigma": sigma,
            "rl100": rl[100],
            "rl100_boot_lo": lo[T_list.index(100)],
            "rl100_boot_hi": hi[T_list.index(100)],
            "rl100_profile_lo": rl100_pl[0],
            "rl100_profile_hi": rl100_pl[1],
        }

        # 4. cyclone / non-cyclone populations
        if tracks is not None:
            st = stations[sid]
            lab = ev.label_cyclone_peaks(peaks, tracks, st["lat"], st["lon"], km, hrs)
            lab.to_csv(out / f"{sid}_peaks_labelled.csv", index=False)
            pops = {}
            for name, mask in (("cyclone", lab["cyclone"]), ("other", ~lab["cyclone"])):
                ex = lab.loc[mask, "hs"].to_numpy() - u
                if len(ex) >= 15:
                    x_i, s_i = evt.fit_gpd(ex)
                    pops[name] = {"xi": x_i, "sigma": s_i, "rate": len(ex) / years, "n": len(ex)}
            rec.update({f"n_{k}": v["n"] for k, v in pops.items()})
            for k, v in pops.items():
                rec[f"xi_{k}"] = v["xi"]
                # asymptotic theory is poor for xi < -0.5 or very few storms: flag instead of trusting the fit
                rec[f"{k}_fit_unreliable"] = bool(v["xi"] < -0.5 or v["n"] < 30)
            if len(pops) == 2:
                for T in (50, 100):
                    rec[f"rl{T}_combined"] = evt.combined_return_level(u, pops["cyclone"], pops["other"], T)

        # 5. annual maxima GEV cross-check, trend and homogeneity
        am = evt.annual_maxima(hs)
        g_xi, g_loc, g_scale = evt.fit_gev(am.to_numpy())
        g_boot = evt.gev_bootstrap(am.to_numpy(), [100], n_boot=n_boot, seed=2)
        g_lo, g_hi = np.percentile(g_boot[:, 0], [2.5, 97.5])
        mk = evt.mann_kendall(am.to_numpy())
        pt = evt.pettitt_test(am.to_numpy())
        rec.update(
            {
                "gev_xi": g_xi,
                "gev_rl100": float(evt.gev_return_level(g_xi, g_loc, g_scale, 100)),
                "gev_rl100_lo": g_lo,
                "gev_rl100_hi": g_hi,
                "mk_z": mk["z"],
                "mk_p": mk["p"],
                "sen_slope_m_per_yr": mk["sen_slope"],
                "pettitt_year": int(am.index[pt["change_index"] - 1]),
                "pettitt_p": pt["p"],
                "record_max": float(hs.max()),
            }
        )
        summary.append(rec)
        print(f"{sid}: u={u:.2f} n={len(peaks)} xi={xi:.3f} RL100={rl[100]:.2f} (GEV {rec['gev_rl100']:.2f})")

    pd.DataFrame(summary).to_csv(out / "evt_summary.csv", index=False)


if __name__ == "__main__":
    main()
