# Data

Nothing large is stored in this repository. The inputs are public; the derived station series are archived
on Zenodo together with the code release.

## Source data (public)

| Item | Source | DOI |
|---|---|---|
| 10 m wind, sea-surface temperature | ERA5 hourly data on single levels (Copernicus Climate Change Service) | 10.24381/cds.adbb2d47 |
| Significant wave height, peak period | Global Ocean Waves Reanalysis, product GLOBAL_MULTIYEAR_WAV_001_032 (Copernicus Marine Service) | 10.48670/moi-00022 |
| Along-track significant wave height | Product WAVE_GLO_PHY_SWH_L3_MY_014_005 (Copernicus Marine Service) | 10.48670/moi-00176 |
| Cyclone tracks | IBTrACS v04r01 (NOAA NCEI) | 10.25921/82ty-9e16 |
| Bathymetry | GEBCO 2026 Grid | 10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa |

ERA5 and CMEMS data were downloaded in August 2026. The Copernicus licences allow free use and redistribution
of derived data with acknowledgement; the acknowledgement text is in the manuscript.

## Derived data in the Zenodo record

`station_series.zip` contains `P1.csv` ... `P5.csv` (3-hourly, 1993-01-01 to 2023-12-31, 90,576 rows each):

| Column | Meaning | Unit |
|---|---|---|
| time | UTC time stamp | |
| u10, v10 | eastward and northward 10 m wind (ERA5, nearest cell) | m/s |
| sst_k | sea-surface temperature (ERA5) | K |
| hs | significant wave height (WAVERYS cell given in `config/stations.yaml`) | m |
| tp | peak wave period (WAVERYS) | s |

`lstm_predictions.zip` holds the per-run test-set predictions (gzip-compressed CSV, one file per station, lead
time, loss and seed). They are needed only to redraw Figure 3 and to recompute the storm tables.

Checksums of the archive files are listed in `SHA256SUMS.txt` (create with `sha256sum *.zip > SHA256SUMS.txt`).
