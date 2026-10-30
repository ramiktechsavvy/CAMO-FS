"""Pareto hypervolume and spread diagnostics for normalized minimization objectives."""

from __future__ import annotations

import numpy as np


def hypervolume_3d(points: np.ndarray, reference: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """Exact dominated volume for 3D minimization points bounded by reference."""
    values = np.asarray(points, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        return 0.0
    values = np.clip(values, 0.0, ref)
    xs = np.unique(np.r_[values[:, 0], ref[0]])
    volume = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        active = values[values[:, 0] <= left + 1e-15, 1:]
        if len(active) == 0 or right <= left:
            continue
        ys = np.unique(np.r_[active[:, 0], ref[1]])
        area = 0.0
        for low_y, high_y in zip(ys[:-1], ys[1:]):
            z_values = active[active[:, 0] <= low_y + 1e-15, 1]
            if len(z_values):
                area += (high_y - low_y) * (ref[2] - min(ref[2], float(z_values.min())))
        volume += (right - left) * area
    return float(max(0.0, volume))


def nearest_neighbor_spacing(points: np.ndarray) -> float:
    """Coefficient of variation of nearest-neighbor distances (lower is more even)."""
    values = np.asarray(points, dtype=float)
    if len(values) < 2:
        return 0.0
    delta = values[:, None, :] - values[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    mean = float(nearest.mean())
    return float(nearest.std(ddof=0) / mean) if mean > 0 else 0.0

