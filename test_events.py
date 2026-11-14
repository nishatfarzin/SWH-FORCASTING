import pandas as pd

from waveshelf import events as ev


def test_haversine_known_distance():
    # one degree of latitude is about 111.2 km
    assert abs(float(ev.haversine_km(20.0, 90.0, 21.0, 90.0)) - 111.2) < 0.5


def test_cyclone_labelling_space_and_time():
    peaks = pd.DataFrame(
        {
            "time": pd.to_datetime(["2020-05-20 12:00", "2020-07-10 00:00", "2021-05-25 06:00"]),
            "hs": [8.5, 4.0, 9.5],
        }
    )
    tracks = pd.DataFrame(
        {
            "time": pd.to_datetime(["2020-05-20 00:00", "2020-05-20 06:00", "2021-05-25 00:00"]),
            "lat": [20.0, 21.0, 5.0],
            "lon": [88.5, 88.5, 70.0],
            "sid": ["AMPHAN", "AMPHAN", "FAR"],
        }
    )
    out = ev.label_cyclone_peaks(peaks, tracks, 18.5, 89.5, max_distance_km=500, window_hours=24)
    assert list(out["cyclone"]) == [True, False, False]
    assert out.loc[0, "storm_id"] == "AMPHAN"
