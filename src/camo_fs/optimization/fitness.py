"""Cached XGBoost fitness for evolutionary search and final full-CV evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from .eci import eci_from_rank_vectors
from .nsga2 import chromosome_id
from .objectives import objective_values, validate_chromosome


@dataclass
class Fitness:
    macro_f1: float
    eci: float
    objectives: tuple[float, float, float]
    fit_seconds: float


class CachedFitnessEvaluator:
    """One fixed training-only holdout surrogate, memoized by binary genotype."""

    def __init__(self, X: np.ndarray, y_codes: np.ndarray, classes: np.ndarray, feature_names: list[str], rank_vectors: list[np.ndarray], model_config: dict, n_features_total: int):
        self.X = X
        self.y = y_codes
        self.classes = classes
        self.feature_names = feature_names
        self.rank_vectors = rank_vectors
        self.model_config = dict(model_config)
        self.n_features_total = n_features_total
        self.cache: dict[str, Fitness] = {}
        self.fit_count = 0
        self.fit_seconds = 0.0

    def evaluate(self, chromosome: np.ndarray) -> Fitness:
        bits = validate_chromosome(chromosome, self.n_features_total)
        key = chromosome_id(bits)
        if key in self.cache:
            return self.cache[key]
        selected = np.flatnonzero(bits)
        started = time.perf_counter()
        model = XGBClassifier(num_class=len(self.classes), **self.model_config)
        model.fit(self.X["train"][:, selected], self.y["train"], verbose=False)
        prediction = model.predict(self.X["validation"][:, selected]).astype(np.int64)
        macro_f1 = float(f1_score(self.y["validation"], prediction, labels=np.arange(len(self.classes)), average="macro", zero_division=0))
        eci = eci_from_rank_vectors(self.rank_vectors, selected)
        objectives = objective_values(macro_f1, len(selected), eci, self.n_features_total)
        elapsed = time.perf_counter() - started
        result = Fitness(macro_f1, eci, objectives, elapsed)
        self.cache[key] = result
        self.fit_count += 1
        self.fit_seconds += elapsed
        return result

