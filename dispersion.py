"""Linear wave theory helpers: wavenumber, group velocity, relative depth and shoaling coefficient."""

from __future__ import annotations

import numpy as np
from scipy import optimize

G = 9.81  # m s^-2


def wavenumber(period_s: float, depth_m: float) -> float:
    """Solve the linear dispersion relation  w^2 = g k tanh(k h)  for k (rad/m)."""
    w = 2.0 * np.pi / period_s
    k_deep = w**2 / G
    k_shallow = w / np.sqrt(G * depth_m)

    def residual(k):
        return w**2 - G * k * np.tanh(k * depth_m)

    # residual() decreases with k; the root is never below either limiting value, so bracket upwards from there
    lo = max(k_deep, k_shallow)
    hi = 2.0 * lo
    while residual(hi) > 0.0:
        hi *= 2.0
    return float(optimize.brentq(residual, lo, hi))


def group_velocity(period_s: float, depth_m: float) -> float:
    k = wavenumber(period_s, depth_m)
    w = 2.0 * np.pi / period_s
    n = 0.5 * (1.0 + 2.0 * k * depth_m / np.sinh(2.0 * k * depth_m))
    return float(n * w / k)


def deep_water_wavelength(period_s: float) -> float:
    return G * period_s**2 / (2.0 * np.pi)


def relative_depth(period_s: float, depth_m: float) -> float:
    """h / L0, with L0 the deep-water wavelength (deep water > 0.5, shallow < 0.05)."""
    return depth_m / deep_water_wavelength(period_s)


def shoaling_coefficient(period_s: float, depth_m: float) -> float:
    """Ks = sqrt(cg0 / cg): change in wave height from deep water to depth h for normally incident,
    non-dissipating linear waves (no refraction, currents or breaking)."""
    cg0 = G * period_s / (4.0 * np.pi)
    return float(np.sqrt(cg0 / group_velocity(period_s, depth_m)))
