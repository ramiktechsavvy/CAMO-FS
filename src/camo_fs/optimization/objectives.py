"""Chromosome validation and CAMO-FS objective composition."""

from __future__ import annotations

import numpy as np


def validate_chromosome(chromosome: np.ndarray, n_features: int = 261) -> np.ndarray:
    values = np.asarray(chromosome)
    if values.ndim != 1 or len(values) != n_features:
        raise ValueError(f"Chromosome must be a one-dimensional vector of length {n_features}.")
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Chromosome entries must be binary.")
    result = values.astype(bool, copy=True)
    if not result.any():
        raise ValueError("All-zero chromosomes are infeasible.")
    return result


def objective_values(macro_f1: float, selected_count: int, eci: float, n_features: int = 261) -> tuple[float, float, float]:
    if not 0 <= macro_f1 <= 1:
        raise ValueError("Macro-F1 must be in [0, 1].")
    if not 1 <= selected_count <= n_features:
        raise ValueError("Selected feature count is outside feasible bounds.")
    if not 0 <= eci <= 1:
        raise ValueError("ECI must be in [0, 1].")
    return (float(1 - macro_f1), float(selected_count / n_features), float(1 - eci))

