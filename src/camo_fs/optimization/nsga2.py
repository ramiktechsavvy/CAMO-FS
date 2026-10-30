"""Small, deterministic NSGA-II primitives and weighted solution selection."""

from __future__ import annotations

import hashlib

import numpy as np

from .objectives import validate_chromosome


def chromosome_id(chromosome: np.ndarray) -> str:
    packed = np.packbits(np.asarray(chromosome, dtype=np.uint8)).tobytes()
    return hashlib.sha256(packed).hexdigest()[:16]


def dominates(first: np.ndarray, second: np.ndarray) -> bool:
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Objectives must be a finite 2D array.")
    n = len(values)
    dominates_set: list[list[int]] = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    first_front: list[int] = []
    for p in range(n):
        for q in range(p + 1, n):
            if dominates(values[p], values[q]):
                dominates_set[p].append(q)
                domination_count[q] += 1
            elif dominates(values[q], values[p]):
                dominates_set[q].append(p)
                domination_count[p] += 1
        if domination_count[p] == 0:
            first_front.append(p)
    fronts = [first_front]
    current = first_front
    while current:
        next_front: list[int] = []
        for p in current:
            for q in dominates_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        current = next_front
    return fronts


def crowding_distance(objectives: np.ndarray, front: list[int]) -> dict[int, float]:
    values = np.asarray(objectives, dtype=float)
    distances = {int(index): 0.0 for index in front}
    if len(front) <= 2:
        return {int(index): float("inf") for index in front}
    for objective in range(values.shape[1]):
        ordered = sorted(front, key=lambda index: values[index, objective])
        low, high = values[ordered[0], objective], values[ordered[-1], objective]
        distances[ordered[0]] = distances[ordered[-1]] = float("inf")
        if high == low:
            continue
        for pos in range(1, len(ordered) - 1):
            index = ordered[pos]
            if np.isfinite(distances[index]):
                distances[index] += (values[ordered[pos + 1], objective] - values[ordered[pos - 1], objective]) / (high - low)
    return distances


def environmental_selection(population: np.ndarray, objectives: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    fronts = non_dominated_sort(objectives)
    chosen: list[int] = []
    for front in fronts:
        remaining = size - len(chosen)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            chosen.extend(front)
        else:
            distances = crowding_distance(objectives, front)
            chosen.extend(sorted(front, key=lambda index: distances[index], reverse=True)[:remaining])
            break
    indices = np.asarray(chosen, dtype=np.int64)
    return population[indices].copy(), np.asarray(objectives)[indices].copy()


def guided_initial_population(size: int, priors: np.ndarray, rng: np.random.Generator, guided_fraction: float = 0.5) -> np.ndarray:
    prior = np.asarray(priors, dtype=float)
    if prior.ndim != 1 or not len(prior) or not np.isfinite(prior).all() or np.any((prior < 0) | (prior > 1)):
        raise ValueError("SHAP priors must be a finite vector bounded by [0,1].")
    guided_count = int(round(size * guided_fraction))
    chromosomes = np.zeros((size, len(prior)), dtype=np.uint8)
    for i in range(size):
        if i < guided_count:
            probability = np.clip(0.02 + 0.65 * prior, 0.02, 0.67)
            chromosomes[i] = rng.random(len(prior)) < probability
        else:
            density = rng.uniform(0.05, 0.95)
            chromosomes[i] = rng.random(len(prior)) < density
        if not chromosomes[i].any():
            chromosomes[i, int(np.argmax(prior))] = 1
    return chromosomes


def uniform_crossover(first: np.ndarray, second: np.ndarray, rng: np.random.Generator, probability: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    a, b = validate_chromosome(first, len(first)), validate_chromosome(second, len(second))
    if rng.random() >= probability:
        return a.astype(np.uint8), b.astype(np.uint8)
    swap = rng.random(len(a)) < 0.5
    child_a, child_b = a.copy(), b.copy()
    child_a[swap], child_b[swap] = b[swap], a[swap]
    if not child_a.any():
        child_a[int(np.argmax(a | b))] = 1
    if not child_b.any():
        child_b[int(np.argmax(a | b))] = 1
    return child_a.astype(np.uint8), child_b.astype(np.uint8)


def mutation_probabilities(chromosome: np.ndarray, priors: np.ndarray, base_probability: float = 0.03) -> np.ndarray:
    bits = np.asarray(chromosome, dtype=bool)
    prior = np.asarray(priors, dtype=float)
    if bits.shape != prior.shape or np.any((prior < 0) | (prior > 1)) or not 0 <= base_probability <= 1:
        raise ValueError("Mutation inputs are invalid or have inconsistent shapes.")
    return np.where(bits, base_probability * (1 - prior), base_probability * prior)


def guided_mutation(chromosome: np.ndarray, priors: np.ndarray, rng: np.random.Generator, base_probability: float = 0.03) -> np.ndarray:
    bits = validate_chromosome(chromosome, len(chromosome)).astype(np.uint8)
    probabilities = mutation_probabilities(bits, priors, base_probability)
    flip = rng.random(len(bits)) < probabilities
    bits[flip] = 1 - bits[flip]
    if not bits.any():
        bits[int(np.argmax(priors))] = 1
    return bits


def weighted_distance_selection(objectives: np.ndarray, weights: tuple[float, float, float] = (0.5, 0.2, 0.3)) -> tuple[int, np.ndarray, np.ndarray]:
    values = np.asarray(objectives, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite N x 3 objective matrix.")
    if weight.shape != (3,) or np.any(weight < 0) or not np.isclose(weight.sum(), 1.0):
        raise ValueError("Three nonnegative weights summing to one are required.")
    low, high = values.min(axis=0), values.max(axis=0)
    span = high - low
    normalized = np.divide(values - low, span, out=np.zeros_like(values), where=span > 0)
    # Weighted Euclidean distance: sqrt(sum_i w_i * normalized_objective_i^2).
    distances = np.sqrt(np.sum(weight[None, :] * normalized**2, axis=1))
    return int(np.argmin(distances)), normalized, distances


def maintain_archive(archive: dict[str, tuple[np.ndarray, np.ndarray]], chromosomes: np.ndarray, objectives: np.ndarray, max_size: int = 500) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    combined = dict(archive)
    for chromosome, objective in zip(chromosomes, objectives):
        combined[chromosome_id(chromosome)] = (np.asarray(chromosome, dtype=np.uint8).copy(), np.asarray(objective, dtype=float).copy())
    keys = list(combined)
    all_objectives = np.vstack([combined[key][1] for key in keys])
    front_indices = non_dominated_sort(all_objectives)[0]
    selected = {keys[i]: combined[keys[i]] for i in front_indices}
    if len(selected) > max_size:
        keep_keys = list(selected)
        values = np.vstack([selected[k][1] for k in keep_keys])
        distances = crowding_distance(values, list(range(len(keep_keys))))
        chosen = sorted(range(len(keep_keys)), key=lambda i: distances[i], reverse=True)[:max_size]
        selected = {keep_keys[i]: selected[keep_keys[i]] for i in chosen}
    return selected

