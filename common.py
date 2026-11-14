"""Small helpers shared by the run scripts."""

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_params():
    with open(ROOT / "config" / "params.yaml") as fh:
        return yaml.safe_load(fh)


def load_stations():
    with open(ROOT / "config" / "stations.yaml") as fh:
        return yaml.safe_load(fh)["stations"]


def read_station(path) -> pd.DataFrame:
    """Read one station CSV (time, u10, v10, sst_k, hs, tp) onto a UTC time index."""
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()
    return df


def read_pred(stem):
    """Read a prediction file written by the run scripts (.csv.gz or .csv); returns None if missing."""
    from pathlib import Path

    for ext in (".csv.gz", ".csv"):
        f = Path(str(stem) + ext)
        if f.exists():
            return pd.read_csv(f, parse_dates=["time"])
    return None


def save_pred(frame, stem):
    """Write predictions gzip-compressed with 4 decimals (keeps 900 runs well under 100 MB)."""
    frame.to_csv(str(stem) + ".csv.gz", index=False, float_format="%.4f", compression="gzip")
