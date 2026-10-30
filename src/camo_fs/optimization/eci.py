"""Explanation Consistency Index utilities using precomputed SHAP rankings."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


REPRESENTATIONS = ("EARLY", "TRANSITION", "LATE")


def load_fold_rankings(results_dir: Path, feature_names: list[str]) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Read saved SHAP fold ranks and align them to canonical feature order."""
    ranking_vectors: list[np.ndarray] = []
    metadata: pd.DataFrame | None = None
    for fold in range(1, 6):
        path = results_dir / f"fold_{fold}_feature_importance.csv"
        frame = pd.read_csv(path)
        required = {"feature_name", "rank", "representation", "channel_id", "sensor", "measurement_name"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} lacks required columns: {sorted(required - set(frame.columns))}")
        if frame["feature_name"].duplicated().any() or set(frame.feature_name) != set(feature_names):
            raise ValueError(f"{path} does not contain exactly the canonical features.")
        aligned = frame.set_index("feature_name").loc[feature_names]
        ranks = aligned["rank"].to_numpy(dtype=float)
        if not np.isfinite(ranks).all():
            raise ValueError(f"{path} contains missing or nonfinite ranks.")
        ranking_vectors.append(ranks)
        if metadata is None:
            metadata = aligned.reset_index()[["feature_name", "representation", "channel_id", "sensor", "measurement_name"]]
    assert metadata is not None
    return ranking_vectors, metadata


def eci_from_rank_vectors(rank_vectors: list[np.ndarray], selected_indices: np.ndarray | list[int]) -> float:
    """ECI=(mean pairwise Kendall tau + 1)/2 for selected feature indices.

    A singleton has no estimable pairwise order; it is assigned ECI=1 because
    its ordering cannot disagree. This edge convention is documented explicitly.
    """
    selected = np.asarray(selected_indices, dtype=np.int64)
    if selected.size == 0:
        raise ValueError("ECI is undefined for an empty feature subset.")
    if np.unique(selected).size != selected.size or np.any(selected < 0):
        raise ValueError("Selected feature indices must be unique and nonnegative.")
    if selected.size < 2:
        return 1.0
    taus: list[float] = []
    for first, second in combinations(rank_vectors, 2):
        if np.max(selected) >= len(first) or np.max(selected) >= len(second):
            raise ValueError("Selected feature index is outside the ranking vector.")
        tau = kendalltau(first[selected], second[selected], variant="b").statistic
        taus.append(float(tau) if np.isfinite(tau) else 0.0)
    return float(np.clip((np.mean(taus) + 1.0) / 2.0, 0.0, 1.0))


def eci_by_representation(
    rank_vectors: list[np.ndarray], feature_names: list[str], representations: list[str], selected_indices: np.ndarray | None = None
) -> dict[str, float]:
    """Compute ECI within each operational representation."""
    if len(feature_names) != len(representations):
        raise ValueError("Feature names and representation labels differ in length.")
    selected_set = set(range(len(feature_names))) if selected_indices is None else set(map(int, selected_indices))
    result: dict[str, float] = {}
    for representation in REPRESENTATIONS:
        indices = np.asarray(
            [i for i, rep in enumerate(representations) if rep == representation and i in selected_set],
            dtype=np.int64,
        )
        result[representation] = eci_from_rank_vectors(rank_vectors, indices) if len(indices) else float("nan")
    return result

