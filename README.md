# Cross-shelf wave forecast skill and extreme wave heights: northern Bay of Bengal

Code behind the paper *Cross-shelf variability of wave forecast skill and extreme significant wave height in the
northern Bay of Bengal: implications for coastal and offshore engineering* (Farzin, Tahura and Rifat).

The study uses a 31-year (1993-2023) ERA5 and CMEMS wave reanalysis at five stations on one transect, from open deep
water to a shallow estuary. It does two things with the same data:

1. forecasts significant wave height (Hs) 6, 12 and 24 h ahead with LSTM networks, compared with persistence, ridge
   regression and XGBoost, and compares several training losses (MSE, MAE, Huber, Focal-MSE and variants);
2. estimates 50- and 100-year return levels of Hs with peaks-over-threshold analysis (generalised Pareto), separately
   for cyclone and non-cyclone storms, with sensitivity checks, profile-likelihood and bootstrap intervals, and an
   annual-maxima GEV cross-check.

## Layout

```
config/           stations.yaml (positions, wave-model cells, GEBCO depths), params.yaml (splits, seeds, EVT settings)
src/waveshelf/    library code: features, baselines, lstm (PyTorch), metrics, events (IBTrACS), evt, dispersion, synthetic
scripts/          one script per analysis step (see "Running it")
tests/            unit and smoke tests on synthetic data
notebooks/        Google Colab version of the whole pipeline
results/          result tables from the authors' run (CSV) that feed every number in the paper
figures/          the six figures of the paper (PDF and PNG)
docs/             DATA.md (sources, file formats) and REPRODUCE.md
references/       reference list for the DOI check
data/             IBTrACS subset; station series are not stored here (see docs/DATA.md)
```

## Setup

```
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate waveshelf
pip install -e .                         # optional; the scripts add src/ to the path themselves
python -m pytest                         # PyTorch tests are skipped if torch is not installed
```

## Data

Nothing is stored in the repository except configuration. Record the details below after you download, and keep them
with the Zenodo archive.

| Item | Source | Identifier / version | Downloaded |
|---|---|---|---|
| 10 m wind, sea-surface temperature | ERA5 hourly data on single levels, Copernicus Climate Data Store | DOI 10.24381/cds.adbb2d47 | (date) |
| Hs, Tp | CMEMS global ocean waves reanalysis (WAVERYS) | dataset id and DOI: (fill in from the product page) | (date) |
| Cyclone tracks | IBTrACS v4, NOAA NCEI | DOI 10.25921/82ty-9e16 | (date) |
| Along-track Hs | multi-mission Sea State CCI level 3 via the Copernicus Marine Service | (fill in) | (date) |

Credentials are needed for the Climate Data Store (`~/.cdsapirc`) and Copernicus Marine (`copernicusmarine login`).

## Running it

Everything below is run from the repository root. `Cross_shelf_wave_pipeline_Colab.ipynb`

```
# 1  station files (either from your own extraction ...)
python scripts/convert_thesis_parquet.py --src <folder with P1_merged_raw.parquet ...> --out data/stations
#    ... or from fresh downloads
python scripts/download_data.py && python scripts/build_station_series.py

# 2  data audit: gaps, descriptive statistics, grid cells used
python scripts/audit_stations.py --data data/stations --out results/tables --wave-nc <wave.nc> --wind-nc <era5 wind.nc>

# 3  baselines: persistence, ridge, XGBoost
python scripts/run_baselines.py --data data/stations --out results/baselines

# 4  cyclone tracks and storm windows
python scripts/make_ibtracs_subset.py --out data/ibtracs_nio.csv
python scripts/list_storms.py --tracks data/ibtracs_nio.csv --from 2019 --to 2023

# 5  extreme values (cyclone/non-cyclone, thresholds, intervals, GEV, trend and change-point tests)
python scripts/run_evt.py --data data/stations --out results/evt --tracks data/ibtracs_nio.csv

# 5b depth regime: relative depth, linear shoaling coefficient, Hs/depth (needs the EVT and audit outputs)
python scripts/depth_diagnostics.py

# 6  LSTM (GPU recommended); restartable, writes after every run
python scripts/run_lstm.py --data data/stations --out results/lstm --preset core --time-budget-min 80

# 7  storm windows, seed-averaged tables with paired tests, figures
python scripts/storm_eval.py
python scripts/aggregate_forecast.py
python scripts/make_figures.py
```

Presets for `run_lstm.py`: `quick` (3 losses x 3 seeds), `core` (5 losses x 5 seeds), `full` (6 losses x 10 seeds).
Predictions are stored gzip-compressed, so even the full preset stays well below 1 GB.
Figures 1 (map) and 6 (altimetry scatter) need bathymetry and altimeter match-ups and are produced separately.

To try the pipeline without any real data, create synthetic data (not physical, never for results):

```
python scripts/make_demo_data.py
python scripts/run_baselines.py --data data/stations_demo --out results_demo/baselines --skip-xgb
python scripts/make_demo_data.py --fake-lstm
python scripts/storm_eval.py --baselines results_demo/baselines --lstm results_demo/lstm --out results_demo/storms
python scripts/aggregate_forecast.py --baselines results_demo/baselines --lstm results_demo/lstm --out results_demo/tables
```

## Choices worth knowing about

* **Chronological split**: train 1993-2014, validation 2015-2018, test 2019-2023. Scalers are fitted on the training
  years only. Storms that fall in the training years (for example Sidr, Aila) are not used as independent tests.
* **Declustering**: exceedances less than 48 h apart belong to one storm (runs declustering); the storm maximum is kept.
* **Bootstrap**: return-level intervals resample whole calendar years, so the storm rate is resampled too.
* **Focal-MSE**: weights are (|r| / max|r|)^2 with the batch maximum detached, so they never exceed 1. The
  `focal_mse_norm` variant rescales them to a mean of 1 to remove the change in effective step size.
* **Cyclone labelling**: a storm peak counts as cyclone-related if an IBTrACS point lies within a chosen distance and time
  window. Both limits are analyst choices and are varied in the sensitivity analysis.

## Reproducing the paper

Every table and figure number is produced from a CSV under `results/`. The Zenodo record for the release that matches
the paper contains the extracted station series and these CSV files, so the figures can be rebuilt without repeating the
downloads or the training.

## Citation and licence

See `CITATION.cff`. Code is released under the MIT licence (`LICENSE`).

## Archive
[![tests](https://github.com/[USER]/cross-shelf-wave-hs-bay-of-bengal/actions/workflows/tests.yml/badge.svg)](https://github.com/[USER]/cross-shelf-wave-hs-bay-of-bengal/actions)

