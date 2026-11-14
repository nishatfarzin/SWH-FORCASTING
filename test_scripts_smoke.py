"""Run the analysis scripts end to end on a small synthetic record (no PyTorch needed)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    out = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_baselines_storms_and_evt_on_small_series(tmp_path):
    sys.path.insert(0, str(ROOT / "src"))
    from waveshelf.synthetic import synthetic_series

    data = tmp_path / "stations"
    data.mkdir()
    for k, sid in enumerate(["P1", "P2", "P3", "P4", "P5"]):
        hs = synthetic_series(years=31, seed=k)
        n = len(hs)
        rng = np.random.default_rng(k)
        wind = np.clip(3 + 2.5 * hs.to_numpy() + rng.normal(0, 1, n), 0, None)
        ang = rng.uniform(0, 2 * np.pi, n)
        pd.DataFrame(
            {
                "time": hs.index,
                "u10": wind * np.cos(ang),
                "v10": wind * np.sin(ang),
                "sst_k": 301.0,
                "hs": hs.to_numpy(),
                "tp": 6 + 1.5 * hs.to_numpy(),
            }
        ).to_csv(data / f"{sid}.csv", index=False)
    base = tmp_path / "baselines"
    run("scripts/run_baselines.py", "--data", data, "--out", base, "--stations", "P1", "--skip-xgb")
    table = pd.read_csv(base / "baseline_metrics.csv")
    assert set(table["model"]) == {"persistence", "ridge"} and len(table) == 6
    # ridge should beat persistence at the longest horizon on this smooth synthetic record
    t24 = table[table.horizon_h == 24].set_index("model")["rmse"]
    assert t24["ridge"] <= t24["persistence"] + 1e-6

    evt_out = tmp_path / "evt"
    run("scripts/run_evt.py", "--data", data, "--out", evt_out, "--stations", "P1", "--n-boot", 30)
    summary = pd.read_csv(evt_out / "evt_summary.csv")
    assert summary.loc[0, "rl100"] > summary.loc[0, "u"]
    assert summary.loc[0, "xi_lo"] < summary.loc[0, "xi"] < summary.loc[0, "xi_hi"]
