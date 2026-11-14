"""Download ERA5 winds/SST and the CMEMS wave reanalysis at the station grid cells.

Credentials: ~/.cdsapirc (Climate Data Store) and `copernicusmarine login` (Copernicus Marine).
Record the dataset ids, product versions and download date in the README after running.
The CMEMS dataset id below must be checked against the product page before use.
"""

import argparse
from pathlib import Path

from common import load_stations

CMEMS_DATASET_ID = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"  # confirm on the product page; record the version
CMEMS_VARIABLES = ["VHM0", "VTPK"]
ERA5_VARIABLES = ["10m_u_component_of_wind", "10m_v_component_of_wind", "sea_surface_temperature"]


def era5(out_dir, years, box):
    import cdsapi

    client = cdsapi.Client()
    for year in years:
        target = Path(out_dir) / f"era5_{year}.nc"
        if target.exists():
            continue
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ERA5_VARIABLES,
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
                "area": box,
                "format": "netcdf",
            },
            str(target),
        )


def cmems(out_dir, start, end, box):
    import copernicusmarine as cm

    north, west, south, east = box
    cm.subset(
        dataset_id=CMEMS_DATASET_ID,
        variables=CMEMS_VARIABLES,
        minimum_longitude=west,
        maximum_longitude=east,
        minimum_latitude=south,
        maximum_latitude=north,
        start_datetime=start,
        end_datetime=end,
        output_directory=str(out_dir),
        output_filename="cmems_waves.nc",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--first-year", type=int, default=1993)
    ap.add_argument("--last-year", type=int, default=2023)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    st = load_stations().values()
    box = [
        max(s["lat"] for s in st) + 0.5,
        min(s["lon"] for s in st) - 0.5,
        min(s["lat"] for s in st) - 0.5,
        max(s["lon"] for s in st) + 0.5,
    ]  # N, W, S, E
    era5(args.out, range(args.first_year, args.last_year + 1), box)
    cmems(args.out, f"{args.first_year}-01-01T00:00:00", f"{args.last_year}-12-31T21:00:00", box)


if __name__ == "__main__":
    main()
