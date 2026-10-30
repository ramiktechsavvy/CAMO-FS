"""Run the complete training-only CAMO-FS NSGA-II optimization.

Run from the project root:
    python src/camo_fs/optimization/run_camo_fs.py

Evolution uses a documented low-cost surrogate. The final selected chromosome is
then evaluated by stratified five-fold CV over every saved training observation.
The saved test partition is never used in optimization or final CV.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from camo_fs.optimization.eci import eci_by_representation, eci_from_rank_vectors, load_fold_rankings
from camo_fs.optimization.fitness import CachedFitnessEvaluator
from camo_fs.optimization.indicators import hypervolume_3d, nearest_neighbor_spacing
from camo_fs.optimization.nsga2 import (
    chromosome_id,
    crowding_distance,
    environmental_selection,
    guided_initial_population,
    guided_mutation,
    maintain_archive,
    non_dominated_sort,
    uniform_crossover,
    weighted_distance_selection,
)
from camo_fs.optimization.objectives import validate_chromosome


DATA_PATH = ROOT / "data/processed/camo_temporal_261.parquet"
METADATA_PATH = ROOT / "data/processed/camo_temporal_feature_metadata.csv"
TRAIN_PATH = ROOT / "data/processed/train_indices.npy"
TEST_PATH = ROOT / "data/processed/test_indices.npy"
SPLIT_INFO_PATH = ROOT / "data/processed/baseline_split.json"
BASELINE_CONFIG_PATH = ROOT / "results/baseline/configuration.json"
SHAP_DIR = ROOT / "results/shap"
SHAP_IMPORTANCE_PATH = SHAP_DIR / "shap_feature_importance.csv"
OUT = ROOT / "results/camo_fs"
ECI_OUT = ROOT / "results/eci"
DOCS = ROOT / "docs"

SEED = 42
POPULATION_SIZE = 60
GENERATIONS = 40
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.03
GUIDED_INITIALIZATION_FRACTION = 0.5
ARCHIVE_MAX_SIZE = 500
WEIGHTS = (0.5, 0.2, 0.3)

# Explicit search-only surrogate. The full baseline config is restored for final CV.
SEARCH_SAMPLE_SIZE = 10_000
SEARCH_VALIDATION_FRACTION = 0.20
SEARCH_ESTIMATORS = 30


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def establish_eci(feature_names: list[str], representations: list[str]) -> tuple[list[np.ndarray], dict[str, Any]]:
    rank_vectors, rank_metadata = load_fold_rankings(SHAP_DIR, feature_names)
    overall = eci_from_rank_vectors(rank_vectors, np.arange(len(feature_names)))
    stage_values = eci_by_representation(rank_vectors, feature_names, representations)
    pairwise_rows = []
    from itertools import combinations
    from scipy.stats import kendalltau

    for first_index, second_index in combinations(range(len(rank_vectors)), 2):
        tau = float(kendalltau(rank_vectors[first_index], rank_vectors[second_index], variant="b").statistic)
        record: dict[str, Any] = {"fold_a": first_index + 1, "fold_b": second_index + 1, "kendall_tau": tau, "eci_pair": (tau + 1) / 2}
        for representation in ("EARLY", "TRANSITION", "LATE"):
            indices = np.flatnonzero(np.asarray(representations) == representation)
            stage_tau = float(kendalltau(rank_vectors[first_index][indices], rank_vectors[second_index][indices], variant="b").statistic)
            record[f"{representation.lower()}_kendall_tau"] = stage_tau
        pairwise_rows.append(record)
    ECI_OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairwise_rows).to_csv(ECI_OUT / "pairwise_kendall_tau.csv", index=False)
    result = {
        "definition": "ECI = (mean_pairwise_Kendall_tau + 1) / 2",
        "fold_count": 5,
        "pair_count": len(pairwise_rows),
        "overall_eci": overall,
        "representation_eci": stage_values,
        "operational_representation_labels_only": True,
        "fold_rankings_source": "results/shap/fold_1_feature_importance.csv through fold_5_feature_importance.csv",
        "rank_ties": "Kendall tau-b; saved fold ranks are ordinal SHAP-importance rankings.",
        "singleton_subset_convention": "ECI=1 because no ranking disagreement can be observed for fewer than two selected features.",
    }
    (ECI_OUT / "eci_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    eci_lines = [
        "# Explanation Consistency Index (ECI)", "",
        "ECI was calculated from the five saved SHAP fold rankings; SHAP was not recomputed.", "",
        "For each of the 10 fold pairs, Kendall tau-b is computed over the selected features' ordinal SHAP-importance ranks. ECI is `(mean pairwise Kendall tau + 1) / 2`, which maps concordance to [0, 1]. Ties are handled with tau-b. For a subset with fewer than two features, ECI is defined as 1 because no ordering disagreement is estimable.", "",
        f"Overall ECI: **{overall:.6f}**", "",
        "| Operational representation | ECI |",
        "|---|---:|",
    ]
    for rep in ("EARLY", "TRANSITION", "LATE"):
        eci_lines.append(f"| {rep} | {stage_values[rep]:.6f} |")
    eci_lines += ["", "These labels refer to July 5, date-to-date change, and July 14 representations. They are not validated biological growth stages.", "", "Pairwise coefficients are saved in `results/eci/pairwise_kendall_tau.csv`; summary values are in `results/eci/eci_summary.json`."]
    (DOCS / "eci_analysis.md").write_text("\n".join(eci_lines) + "\n", encoding="utf-8")
    return rank_vectors, result


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    required_paths = [DATA_PATH, METADATA_PATH, TRAIN_PATH, TEST_PATH, SPLIT_INFO_PATH, BASELINE_CONFIG_PATH, SHAP_IMPORTANCE_PATH]
    required_paths += [SHAP_DIR / f"fold_{i}_feature_importance.csv" for i in range(1, 6)]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required verified Step 0–4 artifact is missing: {path}")
    df = pd.read_parquet(DATA_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    feature_names = [column for column in df.columns if column != "label"]
    if len(df) != 325_834 or len(feature_names) != 261:
        raise ValueError(f"Expected the verified 325,834 x (261 + label) dataset; found {df.shape}.")
    if metadata.feature_name.tolist() != feature_names or metadata.feature_name.duplicated().any():
        raise ValueError("Canonical feature metadata order does not match the temporal dataset.")
    if metadata.representation.value_counts().to_dict() != {"EARLY": 87, "TRANSITION": 87, "LATE": 87}:
        raise ValueError("Expected 87 features in each operational representation.")
    labels = df.label.to_numpy(dtype=np.int64)
    classes = np.sort(np.unique(labels))
    if len(classes) != 7 or not np.isfinite(df[feature_names].to_numpy(dtype=np.float32, copy=False)).all():
        raise ValueError("Dataset class count or feature finiteness check failed.")
    train_ids = np.load(TRAIN_PATH, allow_pickle=False).astype(np.int64)
    test_ids = np.load(TEST_PATH, allow_pickle=False).astype(np.int64)
    split_info = json.loads(SPLIT_INFO_PATH.read_text(encoding="utf-8"))
    if len(train_ids) != split_info["train_rows"] or len(test_ids) != split_info["test_rows"] or np.intersect1d(train_ids, test_ids).size:
        raise ValueError("Saved split files are inconsistent or overlapping.")
    if sha256_file(DATA_PATH) != split_info.get("dataset_sha256"):
        raise ValueError("Temporal dataset hash does not match the saved split metadata.")
    baseline = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    baseline_feature_order = baseline.get("dataset", {}).get("feature_names")
    if baseline_feature_order != feature_names:
        raise ValueError("Saved baseline feature ordering differs from the canonical order.")
    shap_table = pd.read_csv(SHAP_IMPORTANCE_PATH)
    if shap_table.feature_name.tolist() != feature_names:
        # Saved global importance is rank-sorted; align by name, never reorder the data.
        if set(shap_table.feature_name) != set(feature_names) or shap_table.feature_name.duplicated().any():
            raise ValueError("SHAP prior file is missing or duplicating feature names.")
        shap_table = shap_table.set_index("feature_name").loc[feature_names].reset_index()
    if len(shap_table) != 261 or not shap_table.normalized_shap_prior.between(0, 1).all():
        raise ValueError("SHAP prior values are invalid.")
    return df, metadata, feature_names, metadata.representation.tolist(), labels, classes, train_ids, split_info, shap_table


def population_fitness(population: np.ndarray, evaluator: CachedFitnessEvaluator) -> tuple[np.ndarray, list[float], list[float]]:
    evaluated = [evaluator.evaluate(chromosome) for chromosome in population]
    objectives = np.asarray([item.objectives for item in evaluated], dtype=float)
    return objectives, [item.macro_f1 for item in evaluated], [item.eci for item in evaluated]


def tournament(population: np.ndarray, objectives: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    fronts = non_dominated_sort(objectives)
    rank = np.empty(len(population), dtype=int)
    crowding: dict[int, float] = {}
    for front_rank, front in enumerate(fronts):
        rank[front] = front_rank
        crowding.update(crowding_distance(objectives, front))

    def select_one() -> int:
        a, b = rng.integers(0, len(population), size=2)
        if rank[a] < rank[b]:
            return int(a)
        if rank[b] < rank[a]:
            return int(b)
        if crowding[int(a)] > crowding[int(b)]:
            return int(a)
        if crowding[int(b)] > crowding[int(a)]:
            return int(b)
        return int(a if rng.random() < 0.5 else b)

    return population[select_one()].copy()


def final_five_fold_cv(X: np.ndarray, y_codes: np.ndarray, train_ids: np.ndarray, selected: np.ndarray, model_config: dict[str, Any], classes: np.ndarray) -> dict[str, Any]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(train_ids), -1, dtype=np.int64)
    fold_rows = []
    fit_seconds = 0.0
    for fold_number, (fit_pos, val_pos) in enumerate(splitter.split(train_ids, y_codes[train_ids]), start=1):
        fit_ids, val_ids = train_ids[fit_pos], train_ids[val_pos]
        model = XGBClassifier(num_class=len(classes), **model_config)
        started = time.perf_counter()
        model.fit(X[fit_ids][:, selected], y_codes[fit_ids], verbose=False)
        elapsed = time.perf_counter() - started
        fit_seconds += elapsed
        predictions = model.predict(X[val_ids][:, selected]).astype(np.int64)
        oof[val_pos] = predictions
        score = float(f1_score(y_codes[val_ids], predictions, labels=np.arange(len(classes)), average="macro", zero_division=0))
        fold_rows.append({"fold": fold_number, "train_rows": len(fit_ids), "validation_rows": len(val_ids), "macro_f1": score, "fit_seconds": elapsed})
        print(f"Final full-training CV fold {fold_number}/5: Macro-F1={score:.6f}, fit={elapsed:.1f}s", flush=True)
    if np.any(oof < 0):
        raise ValueError("Out-of-fold predictions are incomplete.")
    pooled = float(f1_score(y_codes[train_ids], oof, labels=np.arange(len(classes)), average="macro", zero_division=0))
    return {"folds": fold_rows, "mean_fold_macro_f1": float(np.mean([row["macro_f1"] for row in fold_rows])), "std_fold_macro_f1": float(np.std([row["macro_f1"] for row in fold_rows], ddof=1)), "pooled_oof_macro_f1": pooled, "fit_seconds": fit_seconds}


def main() -> None:
    total_started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    ECI_OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "solutions").mkdir(parents=True, exist_ok=True)
    df, metadata, features, representations, labels, classes, train_ids, split_info, shap_table = load_inputs()
    test_ids = np.load(TEST_PATH, allow_pickle=False)
    rank_vectors, eci_summary = establish_eci(features, representations)

    # Encode only for XGBoost's multiclass interface; original integer labels are retained in files.
    y_codes = np.searchsorted(classes, labels).astype(np.int32)
    all_X = df[features].to_numpy(dtype=np.float32, copy=False)
    selected_sample_ids = train_test_split(
        train_ids,
        test_size=SEARCH_SAMPLE_SIZE,
        random_state=SEED,
        stratify=y_codes[train_ids],
    )[1]
    search_train_ids, search_validation_ids = train_test_split(
        selected_sample_ids,
        test_size=SEARCH_VALIDATION_FRACTION,
        random_state=SEED,
        stratify=y_codes[selected_sample_ids],
    )
    np.savez_compressed(
        OUT / "search_sample_indices.npz",
        sampled_training_ids=selected_sample_ids,
        fit_ids=search_train_ids,
        validation_ids=search_validation_ids,
    )
    search_X = {"train": all_X[search_train_ids], "validation": all_X[search_validation_ids]}
    search_y = {"train": y_codes[search_train_ids], "validation": y_codes[search_validation_ids]}
    baseline_configuration = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))["model"]["configuration"]
    full_model_config = dict(baseline_configuration)
    full_model_config.pop("num_class", None)
    search_model_config = dict(full_model_config)
    search_model_config["n_estimators"] = SEARCH_ESTIMATORS
    priors = shap_table.set_index("feature_name").loc[features, "normalized_shap_prior"].to_numpy(dtype=float)
    evaluator = CachedFitnessEvaluator(search_X, search_y, classes, features, rank_vectors, search_model_config, len(features))
    rng = np.random.default_rng(SEED)
    population = guided_initial_population(POPULATION_SIZE, priors, rng, GUIDED_INITIALIZATION_FRACTION)
    archive: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    search_started = time.perf_counter()

    for generation in range(1, GENERATIONS + 1):
        objectives, macro_scores, eci_values = population_fitness(population, evaluator)
        archive = maintain_archive(archive, population, objectives, ARCHIVE_MAX_SIZE)
        best_macro = max(macro_scores)
        smallest = int(min(np.count_nonzero(chromosome) for chromosome in population))
        best_eci = max(eci_values)
        if generation == GENERATIONS:
            print(
                f"Generation {generation}/{GENERATIONS} | best Macro-F1={best_macro:.5f} | "
                f"smallest subset={smallest} | best ECI={best_eci:.5f} | Pareto archive={len(archive)}",
                flush=True,
            )
            break
        offspring = []
        while len(offspring) < POPULATION_SIZE:
            parent_a = tournament(population, objectives, rng)
            parent_b = tournament(population, objectives, rng)
            child_a, child_b = uniform_crossover(parent_a, parent_b, rng, CROSSOVER_PROBABILITY)
            offspring.append(guided_mutation(child_a, priors, rng, MUTATION_PROBABILITY))
            if len(offspring) < POPULATION_SIZE:
                offspring.append(guided_mutation(child_b, priors, rng, MUTATION_PROBABILITY))
        offspring_arr = np.asarray(offspring, dtype=np.uint8)
        offspring_objectives, _, _ = population_fitness(offspring_arr, evaluator)
        # Archive every evaluated offspring before elitist survivor selection.
        archive = maintain_archive(archive, offspring_arr, offspring_objectives, ARCHIVE_MAX_SIZE)
        combined = np.vstack((population, offspring_arr))
        combined_objectives = np.vstack((objectives, offspring_objectives))
        population, _ = environmental_selection(combined, combined_objectives, POPULATION_SIZE)
        print(
            f"Generation {generation}/{GENERATIONS} | best Macro-F1={best_macro:.5f} | "
            f"smallest subset={smallest} | best ECI={best_eci:.5f} | Pareto archive={len(archive)}",
            flush=True,
        )

    search_seconds = time.perf_counter() - search_started
    archive_items = list(archive.items())
    if not archive_items:
        raise RuntimeError("NSGA-II produced an empty Pareto archive.")
    archive_ids = [key for key, _ in archive_items]
    archive_chromosomes = np.vstack([value[0] for _, value in archive_items])
    archive_objectives = np.vstack([value[1] for _, value in archive_items])
    chosen_position, normalized_objectives, weighted_distances = weighted_distance_selection(archive_objectives, WEIGHTS)
    selected_chromosome = validate_chromosome(archive_chromosomes[chosen_position], len(features))
    selected_indices = np.flatnonzero(selected_chromosome)
    selected_names = [features[i] for i in selected_indices]
    selected_eci = eci_from_rank_vectors(rank_vectors, selected_indices)

    # Full required final evaluation: all saved training observations, proper 5-fold CV, full baseline rounds.
    final_started = time.perf_counter()
    final_cv = final_five_fold_cv(all_X, y_codes, train_ids, selected_indices, full_model_config, classes)
    final_cv_seconds = time.perf_counter() - final_started

    pareto_rows = []
    for pos, (solution_id, (chromosome, objective)) in enumerate(archive_items):
        indices = np.flatnonzero(chromosome)
        fit = evaluator.cache[solution_id]
        row = {
            "solution_id": solution_id,
            "selected_feature_count": int(len(indices)),
            "f1_classification_error": float(objective[0]),
            "f2_subset_ratio": float(objective[1]),
            "f3_explanation_instability": float(objective[2]),
            "macro_f1": fit.macro_f1,
            "ECI": fit.eci,
            "hypervolume_contribution": np.nan,
            "fitness_fidelity": "search_surrogate_10k_holdout_30_trees",
            "final_full_cv_macro_f1": final_cv["pooled_oof_macro_f1"] if solution_id == archive_ids[chosen_position] else np.nan,
            "final_full_cv_classification_error": 1 - final_cv["pooled_oof_macro_f1"] if solution_id == archive_ids[chosen_position] else np.nan,
        }
        pareto_rows.append(row)
        selected_feature_data = metadata.iloc[indices].copy()
        selected_feature_data.insert(0, "feature_index_zero_based", indices)
        selected_feature_data.to_csv(OUT / "solutions" / f"{solution_id}_features.csv", index=False)
    total_hv = hypervolume_3d(archive_objectives, (1.0, 1.0, 1.0))
    contributions = []
    for pos in range(len(archive_objectives)):
        contributions.append(max(0.0, total_hv - hypervolume_3d(np.delete(archive_objectives, pos, axis=0), (1.0, 1.0, 1.0))))
    for row, contribution in zip(pareto_rows, contributions):
        row["hypervolume_contribution"] = float(contribution)
    pareto_table = pd.DataFrame(pareto_rows).sort_values(["f1_classification_error", "f2_subset_ratio", "f3_explanation_instability"])
    pareto_table.to_csv(OUT / "pareto_front.csv", index=False)

    selected_feature_table = metadata.iloc[selected_indices].copy()
    prior_lookup = shap_table.set_index("feature_name")
    selected_feature_table.insert(0, "feature_index_zero_based", selected_indices)
    selected_feature_table["mean_abs_shap"] = selected_feature_table.feature_name.map(prior_lookup.mean_abs_shap)
    selected_feature_table["normalized_shap_prior"] = selected_feature_table.feature_name.map(prior_lookup.normalized_shap_prior)
    selected_feature_table.to_csv(OUT / "final_selected_features.csv", index=False)
    representation_counts = selected_feature_table.representation.value_counts().reindex(["EARLY", "TRANSITION", "LATE"], fill_value=0).to_dict()
    final_pooled_f1 = final_cv["pooled_oof_macro_f1"]
    full_objectives = (1 - final_pooled_f1, len(selected_indices) / len(features), 1 - selected_eci)
    final_solution = {
        "solution_id": archive_ids[chosen_position],
        "selected_feature_names": selected_names,
        "selected_feature_indices_zero_based": selected_indices.astype(int).tolist(),
        "selected_feature_count": int(len(selected_indices)),
        "representation_counts": {key: int(value) for key, value in representation_counts.items()},
        "macro_f1": final_pooled_f1,
        "mean_fold_macro_f1": final_cv["mean_fold_macro_f1"],
        "std_fold_macro_f1": final_cv["std_fold_macro_f1"],
        "classification_error": float(1 - final_pooled_f1),
        "ECI": selected_eci,
        "subset_ratio": float(len(selected_indices) / len(features)),
        "explanation_instability": float(1 - selected_eci),
        "weighted_distance_to_ideal": float(weighted_distances[chosen_position]),
        "weighted_distance_normalized_surrogate_objectives": normalized_objectives[chosen_position].tolist(),
        "final_full_training_cv_objectives": list(map(float, full_objectives)),
        "random_seed": SEED,
        "final_evaluation": "stratified 5-fold CV over the complete saved training partition; test partition unused",
        "final_cv": final_cv,
        "optimizer_configuration": {
            "population_size": POPULATION_SIZE,
            "generations": GENERATIONS,
            "crossover": "uniform",
            "crossover_probability": CROSSOVER_PROBABILITY,
            "mutation": "SHAP-guided; selected drop pm*(1-pi_j), unselected add pm*pi_j",
            "base_mutation_probability": MUTATION_PROBABILITY,
            "guided_initialization_fraction": GUIDED_INITIALIZATION_FRACTION,
            "weights": {"classification_error": 0.5, "subset_size": 0.2, "explanation_instability": 0.3},
            "selection_name": "weighted distance-to-ideal; described as knee-point selection only as paper terminology, not geometric knee detection",
            "pareto_archive_cap": ARCHIVE_MAX_SIZE,
            "search_surrogate": {"stratified_train_only_rows": SEARCH_SAMPLE_SIZE, "holdout_fraction": SEARCH_VALIDATION_FRACTION, "n_estimators": SEARCH_ESTIMATORS, "full_baseline_other_parameters": True},
        },
    }
    (OUT / "final_solution.json").write_text(json.dumps(final_solution, indent=2) + "\n", encoding="utf-8")

    runtime_seconds = time.perf_counter() - total_started
    config = {
        "experiment": "CAMO-FS NSGA-II",
        "dataset_path": str(DATA_PATH.relative_to(ROOT)),
        "dataset_sha256": split_info["dataset_sha256"],
        "feature_count": len(features),
        "feature_order": features,
        "target": "label",
        "classes_original": classes.astype(int).tolist(),
        "train_rows": len(train_ids),
        "test_rows_loaded_only_for_split_integrity_check": len(test_ids),
        "test_set_used_for_optimization_or_final_cv": False,
        "split_indices_reused": True,
        "shap_prior_path": str(SHAP_IMPORTANCE_PATH.relative_to(ROOT)),
        "shap_fold_rankings_reused": True,
        "full_model_config": full_model_config | {"num_class": len(classes)},
        "search_model_config": search_model_config | {"num_class": len(classes)},
        "optimizer": {"population_size": POPULATION_SIZE, "generations": GENERATIONS, "seed": SEED, "crossover_probability": CROSSOVER_PROBABILITY, "mutation_probability": MUTATION_PROBABILITY, "guided_initialization_fraction": GUIDED_INITIALIZATION_FRACTION, "archive_max_size": ARCHIVE_MAX_SIZE},
        "objective_definitions": {"f1": "1 - Macro-F1", "f2": "selected_features / 261", "f3": "1 - ECI_subset"},
        "search_approximation": "Each candidate gets one deterministic stratified holdout evaluation on 10,000 sampled saved-training rows (8,000 fit / 2,000 validation) and 30 boosting rounds. Other XGBoost baseline parameters are unchanged. Candidate fitness is memoized by packed-chromosome SHA-256. The final weighted-distance choice is evaluated separately with 5-fold CV on all saved training rows and 300 rounds.",
        "search_sample_indices_path": str((OUT / "search_sample_indices.npz").relative_to(ROOT)),
        "search_sample_class_counts": {str(int(c)): int(np.count_nonzero(y_codes[selected_sample_ids] == np.searchsorted(classes, c))) for c in classes},
        "search_unique_model_fits": evaluator.fit_count,
        "search_fit_seconds_sum": evaluator.fit_seconds,
        "search_wall_seconds": search_seconds,
        "final_full_5fold_cv_seconds": final_cv_seconds,
        "total_wall_seconds": runtime_seconds,
        "pareto_archive_size": len(archive_items),
        "pareto_hypervolume": total_hv,
        "hypervolume_reference_point": [1.0, 1.0, 1.0],
        "hypervolume_reference_justification": "All three minimization objectives are naturally bounded in [0,1]: classification error, feature subset ratio, and 1-ECI. The reference point is the worst feasible bound.",
        "pareto_nearest_neighbor_spacing_cv": nearest_neighbor_spacing(archive_objectives),
        "software": {name: package_version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "scipy")},
        "python": platform.python_version(),
    }
    (OUT / "configuration.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (OUT / "runtime.json").write_text(json.dumps({"total_seconds": runtime_seconds, "search_seconds": search_seconds, "final_full_training_5fold_cv_seconds": final_cv_seconds, "candidate_model_fit_count": evaluator.fit_count, "candidate_fit_seconds_sum": evaluator.fit_seconds}, indent=2) + "\n", encoding="utf-8")
    summary = {
        "overall_eci": eci_summary["overall_eci"],
        "representation_eci": eci_summary["representation_eci"],
        "optimization_runtime_seconds": search_seconds,
        "total_runtime_seconds": runtime_seconds,
        "pareto_archive_size": len(archive_items),
        "selected_solution_id": archive_ids[chosen_position],
        "selected_feature_count": int(len(selected_indices)),
        "representation_counts": final_solution["representation_counts"],
        "final_macro_f1_pooled_oof": final_pooled_f1,
        "final_macro_f1_mean_fold": final_cv["mean_fold_macro_f1"],
        "final_eci": selected_eci,
        "subset_ratio": final_solution["subset_ratio"],
        "weighted_distance": final_solution["weighted_distance_to_ideal"],
        "search_unique_model_fits": evaluator.fit_count,
        "pareto_hypervolume": total_hv,
    }
    (OUT / "optimization_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_methodology_doc(config, eci_summary, final_solution, summary)
    print("CAMO-FS optimization complete")
    print(json.dumps(summary, indent=2))


def write_methodology_doc(config: dict[str, Any], eci: dict[str, Any], final: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# CAMO-FS optimization methodology", "",
        "This implementation uses 261 predictors in fixed metadata order and the saved train/test split. EARLY (July 5), TRANSITION (July 14 minus July 5), and LATE (July 14) are operational temporal representations, not verified biological stages.", "",
        "## Chromosome and feasibility", "",
        "Each candidate is a 261-bit vector, where 1 selects a feature. Empty vectors are rejected/repaired; indices are zero-based into the canonical feature order. The target label is excluded from the chromosome.", "",
        "## SHAP-guided initialization and variation", "",
        "Half of the initial population is sampled independently using probabilities `clip(0.02 + 0.65*pi_j, 0.02, 0.67)`. The other half uses independently sampled chromosome densities between 0.05 and 0.95. This mixture prevents identical initialization and retains broad subset-size diversity. Uniform crossover is applied to a parent pair with probability 0.9. Mutation uses the requested per-bit probabilities: selected bits leave at `pm*(1-pi_j)` and unselected bits enter at `pm*pi_j`, with `pm=0.03`.", "",
        "## Objectives and ECI", "",
        "The objectives are minimized: `f1=1-Macro-F1`, `f2=selected_count/261`, and `f3=1-ECI_subset`. ECI uses the five existing SHAP feature-rank files and averages the 10 pairwise Kendall tau-b coefficients before mapping to [0,1]. For a candidate, each fold ranking is restricted to its selected features before Kendall tau is calculated. A subset with fewer than two features is assigned ECI=1 by convention because its ranking agreement is not estimable.", "",
        f"Full-feature ECI: {eci['overall_eci']:.6f}; representation ECI: {json.dumps(eci['representation_eci'])}.", "",
        "## NSGA-II and archive", "",
        "The implementation uses pairwise dominance sorting, rank/crowding tournament selection, uniform crossover, guided mutation, and elitist environmental selection from parent-plus-offspring populations. A nondominated archive is updated at every generation, de-duplicates chromosomes, removes dominated points, and is capped at 500 using crowding distance to preserve objective-space diversity. Population size is 60 for 40 generations with seed 42.", "",
        "## Search fitness and computational compromise", "",
        "A timing probe with 8,000 training rows, 100 features, and 100 boosting rounds took about 0.82 seconds for one fit. The requested naive workload can approach 2,400 unique candidates × 5 fits = 12,000 full-data fits; extrapolating the existing full baseline's roughly 129-second fit suggests many days of computation. Therefore evolutionary candidate fitness uses a fixed, stratified 10,000-row sample drawn only from the saved training partition, split 80/20 into 8,000 fit and 2,000 validation rows, and 30 boosting rounds. All other baseline XGBoost settings are unchanged. Candidate scores are approximate; caching avoids repeated chromosome fits. The chosen final solution is then evaluated with the full baseline's 300 rounds and stratified 5-fold CV over all 260,667 saved training rows. The test set is not used.", "",
        "The search holdout's rare-class support is limited, and 30 rounds can underfit relative to the 300-round baseline; these surrogate scores can alter the approximate Pareto front and weighted selection. The final CV result is the reported model estimate, but the selected subset is conditional on the surrogate search.", "",
        "## Pareto indicators", "",
        "Hypervolume is exact in three dimensions with reference point (1,1,1). This is the worst feasible bound for each normalized minimization objective: classification error, subset ratio, and explanation instability each lie in [0,1]. Hypervolume contributions are leave-one-solution-out differences. Spread is reported as the coefficient of variation of each point's nearest-neighbor Euclidean distance in objective space (lower means more even spacing).", "",
        "## Final selection", "",
        "The archive objectives are min-max normalized. The selected point minimizes weighted distance to ideal (0,0,0), using `sqrt(0.5*x1^2 + 0.2*x2^2 + 0.3*x3^2)`. The paper calls its selection a knee point, but this code performs weighted distance-to-ideal selection; it does not detect a geometric knee. The selected subset has " + str(final["selected_feature_count"]) + " features (" + json.dumps(final["representation_counts"]) + "), subset ratio " + f"{final['subset_ratio']:.6f}, pooled full-CV Macro-F1 {final['macro_f1']:.6f}, and ECI {final['ECI']:.6f}.", "",
        "## Limitations", "",
        "The optimizer uses a single saved SHAP prior and fold-ranking set. Fold ECI ranks do not include refitting the model per fold; they measure consistency among previously generated SHAP rankings. Candidate search fitness is an explicitly approximate holdout surrogate. Random row splits do not establish spatially independent generalization. Representation labels remain operational dates/differences, not phenological stages.", "",
        "Run from the repository root with `python src/camo_fs/optimization/run_camo_fs.py`. Results and selected feature lists are saved under `results/camo_fs/`; full ECI calculations are under `results/eci/`.",
    ]
    (DOCS / "camo_fs_methodology.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
