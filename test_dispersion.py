import numpy as np

from waveshelf import dispersion as wd


def test_deep_water_limit():
    k = wd.wavenumber(10.0, 5000.0)
    assert np.isclose(k, (2 * np.pi / 10.0) ** 2 / wd.G, rtol=1e-6)
    assert np.isclose(wd.shoaling_coefficient(10.0, 5000.0), 1.0, atol=1e-6)


def test_shallow_water_limit():
    k = wd.wavenumber(100.0, 2.0)  # very long wave in 2 m of water
    assert np.isclose(k, (2 * np.pi / 100.0) / np.sqrt(wd.G * 2.0), rtol=1e-3)
    # in shallow water Ks grows as h^(-1/4)
    assert wd.shoaling_coefficient(100.0, 1.0) > wd.shoaling_coefficient(100.0, 4.0)


def test_shoaling_has_minimum_in_intermediate_depth():
    ks = [wd.shoaling_coefficient(11.0, h) for h in (500, 100, 50, 30, 15, 8, 3)]
    assert min(ks) < 0.95 < 1.0 < ks[-1]
    assert np.isclose(min(ks), 0.91, atol=0.03)


def test_relative_depth_classes():
    assert wd.relative_depth(10.0, 500.0) > 0.5
    assert 0.05 < wd.relative_depth(10.0, 15.0) < 0.5
