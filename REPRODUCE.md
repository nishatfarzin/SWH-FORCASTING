# Reproducing the results

## Option A: redraw tables and figures from the stored results (minutes)

```
pip install -r requirements.txt
python scripts/make_figures.py --results results --out figures      # Figures 2, 4, 5, 6 (Figure 3 needs the predictions archive)
```

## Option B: rerun everything (hours; a GPU is needed for the LSTM runs)

1. Get the station series: unzip `station_series.zip` from the Zenodo record into `data/stations/`, or rebuild
   them with `scripts/download_data.py` and `scripts/build_station_series.py` (credentials needed).
2. Run the steps in `README.md` in order. `scripts/run_lstm.py --preset core` trains 375 models
   (5 stations x 3 lead times x 5 losses x 5 seeds); on a Colab T4 a run takes about 3 minutes.
   The script writes after every run and can be restarted.
3. `notebooks/Cross_shelf_wave_pipeline_Colab.ipynb` runs the same steps in Google Colab with all outputs on Drive.

## Notes

* Random seeds 0-4 are used for every configuration. GPU runs are deterministic for a given PyTorch and CUDA version
  but may differ slightly between versions.
* The unit tests (`python -m pytest`) run on synthetic data and need neither the real data nor a GPU.
