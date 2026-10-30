"""Final one-pass test evaluation and selector comparison for CAMO-FS.

Run from the project root with:
    python src/camo_fs/experiments/final_evaluation.py

All selectors consume training rows only. Each new selector is fitted and its
selected features are saved before the one final test evaluation for that method.
Existing method checkpoints are reused on restart to avoid repeated test use.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import sys
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from camo_fs.optimization.eci import eci_from_rank_vectors, load_fold_rankings
from camo_fs.optimization.fitness import CachedFitnessEvaluator
from camo_fs.optimization.nsga2 import (
    chromosome_id,
    crowding_distance,
    environmental_selection,
    maintain_archive,
    non_dominated_sort,
    uniform_crossover,
    weighted_distance_selection,
)
from camo_fs.optimization.objectives import objective_values


DATA_PATH = ROOT / "data/processed/camo_temporal_261.parquet"
METADATA_PATH = ROOT / "data/processed/camo_temporal_feature_metadata.csv"
TRAIN_PATH = ROOT / "data/processed/train_indices.npy"
TEST_PATH = ROOT / "data/processed/test_indices.npy"
SPLIT_INFO_PATH = ROOT / "data/processed/baseline_split.json"
BASELINE_DIR = ROOT / "results/baseline"
SHAP_DIR = ROOT / "results/shap"
CAMO_DIR = ROOT / "results/camo_fs"
OUT = ROOT / "results/final"
METHOD_DIR = OUT / "method_results"
SELECTION_DIR = OUT / "selections"
FIGURES = OUT / "figures"
SEED = 42
K = 5
SELECTION_SAMPLE_SIZE = 60_000
SEARCH_SAMPLE_SIZE = 10_000
SEARCH_ESTIMATORS = 30
GA_POPULATION = 20
GA_GENERATIONS = 10
NSGA_POPULATION = 20
NSGA_GENERATIONS = 10

CROP_NAMES = {1: "Corn", 2: "Peas", 3: "Canola", 4: "Soybeans", 5: "Oats", 6: "Wheat", 7: "Broadleaf"}
METHOD_ORDER = ["All_261", "Mutual_Information_5", "RFE_5", "RandomForest_5", "SingleObjectiveGA_5", "ContextFreeNSGAII_5", "CAMO_FS_5"]


def pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> dict[str, Any]:
    df = pd.read_parquet(DATA_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    features = [column for column in df.columns if column != "label"]
    labels_original = df.label.to_numpy(dtype=np.int64)
    classes = np.sort(np.unique(labels_original))
    train_ids = np.load(TRAIN_PATH, allow_pickle=False).astype(np.int64)
    test_ids = np.load(TEST_PATH, allow_pickle=False).astype(np.int64)
    split = json.loads(SPLIT_INFO_PATH.read_text(encoding="utf-8"))
    if len(df) != 325_834 or len(features) != 261 or len(classes) != 7:
        raise ValueError(f"Unexpected data dimensions or classes: {df.shape}, {classes}.")
    if metadata.feature_name.tolist() != features:
        raise ValueError("Feature metadata order does not match the dataset.")
    if sha256_array(train_ids) != sha256_array(np.load(TRAIN_PATH, allow_pickle=False)):
        raise ValueError("Train index integrity check failed.")
    if np.intersect1d(train_ids, test_ids).size or len(train_ids) != split["train_rows"] or len(test_ids) != split["test_rows"]:
        raise ValueError("The saved split is inconsistent or overlapping.")
    if sha256_file(DATA_PATH) != split.get("dataset_sha256"):
        raise ValueError("Temporal dataset checksum differs from the saved split metadata.")
    audit = json.loads((ROOT / "docs/data_audit.json").read_text(encoding="utf-8"))
    raw_zip = ROOT / "data/raw/crop+mapping+using+fused+optical+radar+data+set.zip"
    if sha256_file(raw_zip) != audit.get("zip_sha256"):
        raise ValueError("Raw UCI ZIP checksum differs from the Step 0 audit.")
    baseline_config = json.loads((BASELINE_DIR / "configuration.json").read_text(encoding="utf-8"))
    if baseline_config["dataset"]["feature_names"] != features:
        raise ValueError("The saved baseline uses another feature order.")
    config = dict(baseline_config["model"]["configuration"])
    config.pop("num_class", None)
    y_codes = np.searchsorted(classes, labels_original).astype(np.int32)
    X = df[features].to_numpy(dtype=np.float32, copy=False)
    ranks, _ = load_fold_rankings(SHAP_DIR, features)
    shap_importance = pd.read_csv(SHAP_DIR / "shap_feature_importance.csv")
    shap_importance = shap_importance.set_index("feature_name").loc[features].reset_index()
    if len(shap_importance) != len(features) or not np.isfinite(X).all():
        raise ValueError("Feature matrix or SHAP table validation failed.")
    return {
        "X": X,
        "labels_original": labels_original,
        "y_codes": y_codes,
        "classes": classes,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "split": split,
        "features": features,
        "metadata": metadata,
        "ranks": ranks,
        "shap_importance": shap_importance,
        "model_config": config,
    }


def classwise_rows(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> list[dict[str, Any]]:
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(len(classes)), zero_division=0)
    return [
        {"class_label": int(label), "class_name": CROP_NAMES.get(int(label), str(label)), "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
        for i, label in enumerate(classes)
    ]


def evaluate_test_once(method: str, selected_names: list[str], data: dict[str, Any], selection_seconds: float, selection_details: dict[str, Any]) -> dict[str, Any]:
    """Fit final baseline-config XGBoost on all training rows, predict saved test once."""
    method_path = METHOD_DIR / f"{method}.json"
    selected_path = SELECTION_DIR / f"{method}.json"
    prediction_path = METHOD_DIR / f"{method}_test_predictions.npz"
    if method_path.exists() and selected_path.exists():
        record = json.loads(method_path.read_text(encoding="utf-8"))
        saved_selection = json.loads(selected_path.read_text(encoding="utf-8"))
        if saved_selection["selected_features"] != selected_names:
            raise ValueError(f"Checkpoint feature list mismatch for {method}; refusing another test evaluation.")
        if not prediction_path.exists():
            raise FileNotFoundError(f"Completed test metrics exist without saved predictions for {method}.")
        return record

    # Persist selection before exposing test labels to the final evaluator.
    write_json(selected_path, {
        "method": method,
        "selected_features": selected_names,
        "selected_feature_indices_zero_based": [data["features"].index(name) for name in selected_names],
        "selection_runtime_seconds": selection_seconds,
        "selection_details": selection_details,
        "training_indices_sha256": sha256_array(data["train_ids"]),
        "test_indices_sha256": sha256_array(data["test_ids"]),
        "test_features_or_labels_used_for_selection": False,
    })
    selected_indices = np.asarray([data["features"].index(name) for name in selected_names], dtype=np.int64)
    if len(selected_indices) != K and method != "All_261":
        raise ValueError(f"{method} selected {len(selected_indices)} features, expected {K}.")
    model = XGBClassifier(num_class=len(data["classes"]), **data["model_config"])
    fit_started = time.perf_counter()
    model.fit(data["X"][data["train_ids"]][:, selected_indices], data["y_codes"][data["train_ids"]], verbose=False)
    fit_seconds = time.perf_counter() - fit_started
    # Test labels are accessed only here, after the selector has finished and selection was checkpointed.
    predict_started = time.perf_counter()
    y_true = data["y_codes"][data["test_ids"]]
    y_pred = model.predict(data["X"][data["test_ids"]][:, selected_indices]).astype(np.int32)
    evaluation_seconds = time.perf_counter() - predict_started
    metric_values = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.arange(7), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=np.arange(7), average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    report_rows = classwise_rows(y_true, y_pred, data["classes"])
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(7))
    cm_names = [CROP_NAMES[int(label)] for label in data["classes"]]
    cm_df = pd.DataFrame(cm, index=cm_names, columns=cm_names)
    cm_df.index.name = "true_class"
    cm_df.to_csv(METHOD_DIR / f"{method}_confusion_matrix.csv")
    pd.DataFrame(report_rows).to_csv(METHOD_DIR / f"{method}_classification_report.csv", index=False)
    np.savez_compressed(prediction_path, test_indices=data["test_ids"], y_true_codes=y_true, y_pred_codes=y_pred)
    test_eci = eci_from_rank_vectors(data["ranks"], selected_indices)
    elapsed = selection_seconds + fit_seconds + evaluation_seconds
    record = {
        "method": method,
        "selected_feature_count": int(len(selected_names)),
        "selected_features": selected_names,
        "metrics": metric_values,
        "classwise": report_rows,
        "confusion_matrix": cm.tolist(),
        "class_labels": [int(c) for c in data["classes"]],
        "ECI": float(test_eci),
        "runtime_seconds": elapsed,
        "runtime_components_seconds": {"selection": selection_seconds, "final_xgboost_fit": fit_seconds, "test_prediction_and_metrics": evaluation_seconds},
        "test_rows": int(len(y_true)),
        "train_rows": int(len(data["train_ids"])),
        "train_indices_sha256": sha256_array(data["train_ids"]),
        "test_indices_sha256": sha256_array(data["test_ids"]),
        "test_evaluated_once": True,
        "test_set_used_for_selection_or_tuning": False,
        "selection_details": selection_details,
        "final_model_configuration": data["model_config"] | {"num_class": 7},
        "prediction_artifact": str(prediction_path.relative_to(ROOT)),
    }
    write_json(method_path, record)
    if method == "CAMO_FS_5":
        write_json(OUT / "camo_fs_test_metrics.json", record)
        pd.DataFrame(report_rows).to_csv(OUT / "camo_fs_classification_report.csv", index=False)
        cm_df.to_csv(OUT / "camo_fs_confusion_matrix.csv")
    print(f"Completed final test evaluation: {method} | Macro-F1={metric_values['macro_f1']:.6f} | runtime={elapsed:.1f}s", flush=True)
    return record


def training_selection_sample(data: dict[str, Any], size: int = SELECTION_SAMPLE_SIZE) -> np.ndarray:
    path = OUT / "selector_training_sample_indices.npy"
    if path.exists():
        ids = np.load(path, allow_pickle=False)
        if len(ids) == size and set(ids).issubset(set(data["train_ids"])):
            return ids
    ids = train_test_split(data["train_ids"], test_size=size, random_state=SEED, stratify=data["y_codes"][data["train_ids"]])[1]
    np.save(path, ids, allow_pickle=False)
    return np.asarray(ids, dtype=np.int64)


def select_mi(data: dict[str, Any], sample_ids: np.ndarray) -> tuple[list[str], float, dict[str, Any]]:
    started = time.perf_counter()
    scores = mutual_info_classif(data["X"][sample_ids], data["y_codes"][sample_ids], discrete_features=False, n_neighbors=3, random_state=SEED, n_jobs=-1)
    indices = np.argsort(-scores, kind="stable")[:K]
    return [data["features"][int(i)] for i in indices], time.perf_counter() - started, {"selector": "mutual_info_classif", "sample_rows": int(len(sample_ids)), "n_neighbors": 3, "random_state": SEED, "selected_mi_scores": scores[indices].tolist(), "training_only": True}


def select_rf(data: dict[str, Any], sample_ids: np.ndarray) -> tuple[list[str], float, dict[str, Any]]:
    started = time.perf_counter()
    selector = RandomForestClassifier(n_estimators=80, max_features="sqrt", min_samples_leaf=2, n_jobs=-1, random_state=SEED, class_weight=None)
    selector.fit(data["X"][sample_ids], data["y_codes"][sample_ids])
    indices = np.argsort(-selector.feature_importances_, kind="stable")[:K]
    return [data["features"][int(i)] for i in indices], time.perf_counter() - started, {"selector": "RandomForestClassifier embedded feature_importances_", "sample_rows": int(len(sample_ids)), "n_estimators": 80, "max_features": "sqrt", "min_samples_leaf": 2, "random_state": SEED, "training_only": True}


def select_rfe(data: dict[str, Any], sample_ids: np.ndarray) -> tuple[list[str], float, dict[str, Any]]:
    # A deterministic tree ensemble RFE on a 20k stratified training-only sample.
    ids = train_test_split(sample_ids, test_size=20_000, random_state=SEED, stratify=data["y_codes"][sample_ids])[1]
    started = time.perf_counter()
    estimator = ExtraTreesClassifier(n_estimators=16, max_depth=12, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=SEED)
    selector = RFE(estimator=estimator, n_features_to_select=K, step=0.5)
    selector.fit(data["X"][ids], data["y_codes"][ids])
    indices = np.flatnonzero(selector.support_)
    return [data["features"][int(i)] for i in indices], time.perf_counter() - started, {"selector": "sklearn RFE with ExtraTreesClassifier", "sample_rows": int(len(ids)), "n_estimators": 16, "max_depth": 12, "min_samples_leaf": 5, "max_features": "sqrt", "step": 0.5, "random_state": SEED, "training_only": True}


def make_search_evaluator(data: dict[str, Any]) -> tuple[CachedFitnessEvaluator, dict[str, Any]]:
    path = CAMO_DIR / "search_sample_indices.npz"
    if path.exists():
        samples = np.load(path, allow_pickle=False)
        sample_ids = samples["sampled_training_ids"]
        fit_ids = samples["fit_ids"]
        validation_ids = samples["validation_ids"]
    else:
        sample_ids = train_test_split(data["train_ids"], test_size=SEARCH_SAMPLE_SIZE, random_state=SEED, stratify=data["y_codes"][data["train_ids"]])[1]
        fit_ids, validation_ids = train_test_split(sample_ids, test_size=0.2, random_state=SEED, stratify=data["y_codes"][sample_ids])
    if not set(sample_ids).issubset(set(data["train_ids"])) or np.intersect1d(sample_ids, data["test_ids"]).size:
        raise ValueError("Evolutionary selector sample contains non-training rows.")
    config = dict(data["model_config"])
    config["n_estimators"] = SEARCH_ESTIMATORS
    evaluator = CachedFitnessEvaluator(
        {"train": data["X"][fit_ids], "validation": data["X"][validation_ids]},
        {"train": data["y_codes"][fit_ids], "validation": data["y_codes"][validation_ids]},
        data["classes"], data["features"], data["ranks"], config, len(data["features"]),
    )
    return evaluator, {"search_sample_size": len(sample_ids), "fit_rows": len(fit_ids), "validation_rows": len(validation_ids), "n_estimators": SEARCH_ESTIMATORS, "same_seed42_search_sample_as_CAMO_FS": True, "test_rows_used": False}


def normalize_k(chromosome: np.ndarray, rng: np.random.Generator, k: int = K) -> np.ndarray:
    bits = np.asarray(chromosome, dtype=np.uint8).copy()
    selected = np.flatnonzero(bits)
    if len(selected) > k:
        bits[rng.choice(selected, len(selected) - k, replace=False)] = 0
    elif len(selected) < k:
        available = np.flatnonzero(bits == 0)
        bits[rng.choice(available, k - len(selected), replace=False)] = 1
    return bits


def random_k_population(size: int, n_features: int, rng: np.random.Generator) -> np.ndarray:
    pop = np.zeros((size, n_features), dtype=np.uint8)
    for i in range(size):
        pop[i, rng.choice(n_features, K, replace=False)] = 1
    return pop


def ga_select(data: dict[str, Any], evaluator: CachedFitnessEvaluator, sample_info: dict[str, Any]) -> tuple[list[str], float, dict[str, Any]]:
    started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    pop = random_k_population(GA_POPULATION, len(data["features"]), rng)
    best_score = -1.0
    best_chrom = pop[0].copy()
    for generation in range(1, GA_GENERATIONS + 1):
        scored = [(evaluator.evaluate(chrom).macro_f1, chrom.copy()) for chrom in pop]
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored[0][0] > best_score:
            best_score, best_chrom = scored[0][0], scored[0][1].copy()
        next_pop = [scored[0][1].copy(), scored[1][1].copy()]
        while len(next_pop) < GA_POPULATION:
            def parent() -> np.ndarray:
                a, b = rng.choice(len(scored), 2, replace=False)
                return scored[a if scored[a][0] >= scored[b][0] else b][1]
            child_a, child_b = uniform_crossover(parent(), parent(), rng, probability=0.9)
            for child in (child_a, child_b):
                child = normalize_k(child, rng)
                if rng.random() < 0.25:
                    one = rng.choice(np.flatnonzero(child))
                    zero = rng.choice(np.flatnonzero(child == 0))
                    child[one], child[zero] = 0, 1
                next_pop.append(normalize_k(child, rng))
                if len(next_pop) == GA_POPULATION:
                    break
        pop = np.asarray(next_pop, dtype=np.uint8)
        print(f"Single-objective GA generation {generation}/{GA_GENERATIONS} | best surrogate Macro-F1={best_score:.5f}", flush=True)
    # Use best cached K=5 chromosome, no test data involved.
    details = sample_info | {"population_size": GA_POPULATION, "generations": GA_GENERATIONS, "objective": "maximize Macro-F1 only", "k_constraint": K, "crossover": "uniform, p=0.9", "mutation": "uniform random one-feature swap, p=0.25", "random_state": SEED, "best_search_macro_f1": best_score, "unique_model_fits": evaluator.fit_count}
    return [data["features"][int(i)] for i in np.flatnonzero(best_chrom)], time.perf_counter() - started, details


def context_free_nsga_select(data: dict[str, Any], evaluator: CachedFitnessEvaluator, sample_info: dict[str, Any]) -> tuple[list[str], float, dict[str, Any]]:
    started = time.perf_counter()
    rng = np.random.default_rng(SEED + 1)
    population = random_k_population(NSGA_POPULATION, len(data["features"]), rng)
    archive: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for generation in range(1, NSGA_GENERATIONS + 1):
        current = [evaluator.evaluate(chrom) for chrom in population]
        objectives = np.asarray([item.objectives for item in current])
        archive = maintain_archive(archive, population, objectives, max_size=250)
        fronts = non_dominated_sort(objectives)
        ranks = np.empty(len(population), dtype=int)
        distances: dict[int, float] = {}
        for rank, front in enumerate(fronts):
            ranks[front] = rank
            distances.update(crowding_distance(objectives, front))

        def parent() -> np.ndarray:
            a, b = rng.integers(0, len(population), size=2)
            if ranks[a] != ranks[b]:
                return population[int(a if ranks[a] < ranks[b] else b)].copy()
            return population[int(a if distances[int(a)] >= distances[int(b)] else b)].copy()

        offspring = []
        while len(offspring) < NSGA_POPULATION:
            child_a, child_b = uniform_crossover(parent(), parent(), rng, probability=0.9)
            for child in (child_a, child_b):
                # Context-free mutation: independent uniform bit flips, followed by exact-K repair.
                flips = rng.random(len(child)) < 0.03
                child[flips] = 1 - child[flips]
                offspring.append(normalize_k(child, rng))
                if len(offspring) == NSGA_POPULATION:
                    break
        offspring_arr = np.asarray(offspring, dtype=np.uint8)
        child_fit = [evaluator.evaluate(chrom) for chrom in offspring_arr]
        child_obj = np.asarray([item.objectives for item in child_fit])
        archive = maintain_archive(archive, offspring_arr, child_obj, max_size=250)
        combined = np.vstack((population, offspring_arr))
        population, _ = environmental_selection(combined, np.vstack((objectives, child_obj)), NSGA_POPULATION)
        print(f"Context-free NSGA-II generation {generation}/{NSGA_GENERATIONS} | archive={len(archive)}", flush=True)

    items = list(archive.items())
    archive_chromosomes = np.vstack([value[0] for _, value in items])
    archive_objectives = np.vstack([value[1] for _, value in items])
    # This experiment maintains exactly K=5 features per chromosome; use the same weighted rule.
    pos, _, dist = weighted_distance_selection(archive_objectives, (0.5, 0.2, 0.3))
    selected = archive_chromosomes[pos]
    details = sample_info | {"population_size": NSGA_POPULATION, "generations": NSGA_GENERATIONS, "chromosome": "261-bit vector with exact K=5", "objectives": ["1-Macro-F1", "selected/261 (constant under K=5)", "1-ECI from stored fold rankings"], "initialization": "uniform random K=5 subsets", "crossover": "uniform, p=0.9 with random K repair", "mutation": "uniform bit mutation p=0.03 with random K repair; no SHAP prior or guidance", "selection": "weighted distance-to-ideal among context-free Pareto archive", "weighted_distance": float(dist[pos]), "archive_size": len(items), "unique_model_fits": evaluator.fit_count}
    return [data["features"][int(i)] for i in np.flatnonzero(selected)], time.perf_counter() - started, details


def bootstrap_camo(prediction_path: Path, seed: int = SEED, replicates: int = 1000) -> dict[str, Any]:
    saved = np.load(prediction_path, allow_pickle=False)
    y_true = saved["y_true_codes"]
    y_pred = saved["y_pred_codes"]
    rng = np.random.default_rng(seed)
    acc = np.empty(replicates, dtype=float)
    macro = np.empty(replicates, dtype=float)
    labels = np.arange(7)
    for i in range(replicates):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        acc[i] = accuracy_score(y_true[idx], y_pred[idx])
        macro[i] = f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0)
    return {
        "method": "nonparametric row bootstrap of the one saved test prediction vector; not independent experimental runs",
        "replicates": replicates,
        "seed": seed,
        "accuracy_95_percentile_ci": np.percentile(acc, [2.5, 97.5]).tolist(),
        "macro_f1_95_percentile_ci": np.percentile(macro, [2.5, 97.5]).tolist(),
    }


def create_comparison_outputs(data: dict[str, Any], records: dict[str, dict[str, Any]], baseline_record: dict[str, Any]) -> dict[str, Any]:
    summary_eci = json.loads((ROOT / "results/eci/eci_summary.json").read_text(encoding="utf-8"))
    baseline_metrics = baseline_record["metrics"]
    baseline_eci = float(summary_eci["overall_eci"])
    all_features = data["features"]
    rows = [{
        "method": "All_261", "selected_feature_count": 261, "feature_reduction_percent": 0.0,
        "accuracy": baseline_metrics["accuracy"], "macro_f1": baseline_metrics["macro_f1"], "weighted_f1": baseline_metrics["weighted_f1"],
        "kappa": baseline_metrics["cohen_kappa"], "eci": baseline_eci,
        "runtime": baseline_record["runtime_seconds"]["total"], "selected_features": ";".join(all_features),
        "runtime_includes": "existing full-feature baseline training and final test evaluation",
    }]
    for method in METHOD_ORDER[1:]:
        record = records[method]
        count = record["selected_feature_count"]
        rows.append({
            "method": method, "selected_feature_count": count, "feature_reduction_percent": (1 - count / 261) * 100,
            "accuracy": record["metrics"]["accuracy"], "macro_f1": record["metrics"]["macro_f1"],
            "weighted_f1": record["metrics"]["weighted_f1"], "kappa": record["metrics"]["kappa"], "eci": record["ECI"],
            "runtime": record["runtime_seconds"], "selected_features": ";".join(record["selected_features"]),
            "runtime_includes": "selection, final full-training XGBoost fit, and one saved-test evaluation; CAMO-FS includes prior search and full training CV",
        })
    main = pd.DataFrame(rows).set_index("method").loc[METHOD_ORDER].reset_index()
    main.to_csv(OUT / "main_comparison.csv", index=False)

    ablation = pd.read_csv(ROOT / "results/integrity/representation_ablation.csv")
    ablation = ablation[["experiment", "feature_count", "accuracy", "macro_f1", "weighted_f1", "cohen_kappa"]].rename(columns={"experiment": "representation", "cohen_kappa": "kappa"})
    ablation.to_csv(OUT / "representation_ablation.csv", index=False)

    selector_methods = ["Mutual_Information_5", "RFE_5", "RandomForest_5", "SingleObjectiveGA_5", "ContextFreeNSGAII_5"]
    strongest = max(selector_methods, key=lambda method: records[method]["metrics"]["macro_f1"])
    classwise = []
    for method in ["All_261", "CAMO_FS_5", strongest]:
        source = baseline_record["classwise"] if method == "All_261" else records[method]["classwise"]
        for item in source:
            classwise.append({"method": method, **item})
    pd.DataFrame(classwise).to_csv(OUT / "classwise_comparison.csv", index=False)

    interpretation = data["metadata"].copy()
    shap_table = data["shap_importance"].set_index("feature_name")
    interpretation["mean_abs_shap"] = interpretation.feature_name.map(shap_table.mean_abs_shap)
    interpretation["normalized_shap_prior"] = interpretation.feature_name.map(shap_table.normalized_shap_prior)
    selected_set = set(records["CAMO_FS_5"]["selected_features"])
    interpretation["selected"] = interpretation.feature_name.isin(selected_set)
    interpretation = interpretation.rename(columns={"channel_id": "channel", "measurement_name": "measurement"})
    interpretation.to_csv(OUT / "selected_feature_interpretation.csv", index=False)

    ablation_methods = ["All_261", "SingleObjectiveGA_5", "ContextFreeNSGAII_5", "CAMO_FS_5"]
    ablation_notes = {
        "All_261": "Reference without feature selection; describes change with feature reduction.",
        "SingleObjectiveGA_5": "Single-objective search; CAMO-FS contrast changes both objective structure and SHAP guidance.",
        "ContextFreeNSGAII_5": "No SHAP-guided initialization/mutation. Under fixed K=5, subset-size is constant and observed subset ECI was 1.0, so objectives 2 and 3 were degenerate and the archive collapsed to one solution; this is a limited context-free comparator.",
        "CAMO_FS_5": "SHAP-guided multi-objective method reference.",
    }
    ablation_rows = []
    for name in ablation_methods:
        row = main.loc[main.method == name].iloc[0]
        ablation_rows.append({"method": name, "selected_feature_count": int(row.selected_feature_count), "accuracy": row.accuracy, "macro_f1": row.macro_f1, "weighted_f1": row.weighted_f1, "kappa": row.kappa, "eci": row.eci, "interpretation": ablation_notes[name]})
    pd.DataFrame(ablation_rows).to_csv(OUT / "camo_fs_ablation.csv", index=False)

    front = pd.read_csv(CAMO_DIR / "pareto_front.csv")
    return {"main": main, "ablation": ablation, "classwise": pd.DataFrame(classwise), "interpretation": interpretation, "strongest_baseline_selector": strongest, "pareto_front": front}


def make_figures(outputs: dict[str, Any], records: dict[str, dict[str, Any]], data: dict[str, Any]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    main = outputs["main"]
    labels = [m.replace("_", " ") for m in main.method]
    for metric, filename, title in [
        ("macro_f1", "macro_f1_by_method.png", "Test Macro-F1 by method"),
        ("accuracy", "accuracy_by_method.png", "Test accuracy by method"),
        ("selected_feature_count", "feature_count_reduction_by_method.png", "Selected feature count by method"),
        ("eci", "eci_by_method.png", "Subset ECI by method"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        values = main[metric].to_numpy(dtype=float)
        bars = ax.bar(labels, values, color=["#777777", "#4C78A8", "#72B7B2", "#F58518", "#B279A2", "#E45756", "#54A24B"])
        ax.set_title(title)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.tick_params(axis="x", labelrotation=28)
        for index, (bar, value) in enumerate(zip(bars, values)):
            if metric == "selected_feature_count":
                reduction = float(main.iloc[index].feature_reduction_percent)
                label = f"{int(value)} ({reduction:.1f}% less)"
            else:
                label = f"{value:.3f}"
            ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=180)
        plt.close(fig)

    chosen = data["metadata"][data["metadata"].feature_name.isin(records["CAMO_FS_5"]["selected_features"])]
    rep_counts = chosen.representation.value_counts().reindex(["EARLY", "TRANSITION", "LATE"], fill_value=0)
    sensor_counts = chosen.sensor.value_counts().reindex(["radar", "optical"], fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].bar(rep_counts.index, rep_counts.values, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_title("CAMO-FS selected features by representation")
    axes[0].set_ylabel("Feature count")
    axes[1].bar(sensor_counts.index, sensor_counts.values, color=["#B279A2", "#E45756"])
    axes[1].set_title("CAMO-FS selected features by sensor")
    axes[1].set_ylabel("Feature count")
    fig.tight_layout()
    fig.savefig(FIGURES / "camo_fs_feature_distribution.png", dpi=180)
    plt.close(fig)

    cm = np.asarray(records["CAMO_FS_5"]["confusion_matrix"])
    class_names = [CROP_NAMES[int(c)] for c in data["classes"]]
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, label="Samples")
    ax.set(xticks=np.arange(7), yticks=np.arange(7), xticklabels=class_names, yticklabels=class_names, xlabel="Predicted class", ylabel="True class", title="CAMO-FS test confusion matrix")
    ax.tick_params(axis="x", labelrotation=35)
    threshold = cm.max() / 2
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", color="white" if cm[i,j] > threshold else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "camo_fs_confusion_matrix.png", dpi=180)
    plt.close(fig)

    front = outputs["pareto_front"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    scatter = ax.scatter(front.f2_subset_ratio, front.f1_classification_error, c=front.f3_explanation_instability, cmap="viridis", s=45, edgecolor="black", linewidth=0.35)
    fig.colorbar(scatter, ax=ax, label="Explanation instability (1 − ECI)")
    ax.set(xlabel="Relative subset size", ylabel="Search-surrogate classification error", title="CAMO-FS Pareto archive (search objectives)")
    fig.tight_layout()
    fig.savefig(FIGURES / "camo_fs_pareto_front.png", dpi=180)
    plt.close(fig)


def write_final_report(data: dict[str, Any], records: dict[str, dict[str, Any]], baseline: dict[str, Any], outputs: dict[str, Any], bootstrap: dict[str, Any]) -> None:
    eci_summary = json.loads((ROOT / "results/eci/eci_summary.json").read_text(encoding="utf-8"))
    shap_summary = pd.read_csv(SHAP_DIR / "representation_summary.csv")
    main = outputs["main"]
    camo = records["CAMO_FS_5"]
    strongest = outputs["strongest_baseline_selector"]
    methods_table = ["| Method | Features | Accuracy | Macro-F1 | Weighted-F1 | Kappa | ECI | Runtime (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in main.itertuples(index=False):
        methods_table.append(f"| {row.method} | {row.selected_feature_count} | {row.accuracy:.6f} | {row.macro_f1:.6f} | {row.weighted_f1:.6f} | {row.kappa:.6f} | {row.eci:.6f} | {row.runtime:.1f} |")
    class_rows = outputs["classwise"]
    minor = class_rows[class_rows.class_name.isin(["Peas", "Broadleaf"])][["method", "class_name", "f1", "support"]]
    comparison_rows = [f"| {r.method} | {r.class_name} | {r.f1:.6f} | {r.support} |" for r in minor.itertuples(index=False)]
    rep_rows = [f"| {r.representation} | {r.feature_count} | {r.accuracy:.6f} | {r.macro_f1:.6f} | {r.weighted_f1:.6f} | {r.kappa:.6f} |" for r in outputs["ablation"].itertuples(index=False)]
    stage_importance = "\n".join(f"- {r.representation}: {r.percentage_contribution:.3f}% of total mean absolute SHAP" for r in shap_summary.itertuples(index=False))
    camo_features = ", ".join(f"`{name}`" for name in camo["selected_features"])
    report = f"""# CAMO-FS Final Results

## 1. Experimental setup

The experiment uses UCI Crop Mapping using Fused Optical-Radar Dataset #525, with 325,834 rows, 261 derived predictors, and seven crop classes. The canonical representations are EARLY (July 5), TRANSITION (July 14 minus July 5), and LATE (July 14). These are operational temporal representations, not verified biological stages. The existing seed-42 stratified split was reused: {len(data['train_ids']):,} training rows and {len(data['test_ids']):,} test rows. XGBoost used the saved baseline configuration (300 estimators, depth 6, learning rate 0.05, subsample and column sample 0.8, seed 42, histogram tree method). Each new method selected features using training data only, then was evaluated once on the saved test rows.

## 2. Baseline

The existing all-feature baseline achieved Accuracy {baseline['metrics']['accuracy']:.6f}, Macro-F1 {baseline['metrics']['macro_f1']:.6f}, Weighted-F1 {baseline['metrics']['weighted_f1']:.6f}, and Kappa {baseline['metrics']['cohen_kappa']:.6f}. Its artifacts were reused; the baseline was not retrained.

## 3. Representation analysis

| Representation | Features | Accuracy | Macro-F1 | Weighted-F1 | Kappa |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rep_rows)}

These are the already completed ablation results; they were not rerun.

## 4. SHAP analysis

The saved training-only TreeSHAP analysis found representation contributions of:
{stage_importance}

The ten global leaders include TRANSITION_CH080_MeanPC2, LATE_CH051_G, TRANSITION_CH051_G, LATE_CH046_yamodd, and TRANSITION_CH055_NDVI. Values are model attributions, not causal effects. The saved analysis used a documented SHAP/XGBoost base-score compatibility shim and native TreeSHAP parity check; see `docs/shap_analysis.md`.

## 5. ECI

ECI from the existing five SHAP fold rankings was {eci_summary['overall_eci']:.6f} overall. Representation ECI was EARLY {eci_summary['representation_eci']['EARLY']:.6f}, TRANSITION {eci_summary['representation_eci']['TRANSITION']:.6f}, and LATE {eci_summary['representation_eci']['LATE']:.6f}. Subset ECI values use the same rankings restricted to each method's selected features.

## 6. CAMO-FS optimization

CAMO-FS used SHAP-guided initialization and mutation, NSGA-II with objectives 1−Macro-F1, subset ratio, and 1−ECI. The saved optimizer ran 60 individuals for 40 generations, with a 10,000-row training-only surrogate objective (8,000 fit / 2,000 validation, 30 boosting rounds), caching 2,169 unique candidates. The selected subset received full saved-training 5-fold CV at 300 rounds. Final archive selection was weighted distance-to-ideal with weights (0.5, 0.2, 0.3); this is not geometric knee detection. Only one deterministic optimizer run exists.

## 7. Feature selection results

CAMO-FS selected {camo['selected_feature_count']} of 261 features ({(1-camo['selected_feature_count']/261)*100:.3f}% reduction; subset ratio {camo['selected_feature_count']/261:.6f}): {camo_features}. Its full-training CV Macro-F1 was {json.loads((CAMO_DIR/'final_solution.json').read_text(encoding='utf-8'))['macro_f1']:.6f}. Its final saved-test Accuracy was {camo['metrics']['accuracy']:.6f}, Macro-F1 {camo['metrics']['macro_f1']:.6f}, Weighted-F1 {camo['metrics']['weighted_f1']:.6f}, and Kappa {camo['metrics']['kappa']:.6f}.

## 8. Baseline comparison

{chr(10).join(methods_table)}

Runtime includes selection, final-model training, and one test evaluation for newly run selectors. CAMO-FS includes its prior optimization and full training CV. The all-feature runtime is the existing baseline runtime. Selector runtimes therefore reflect their stated computational procedures and are not equal-cost implementations. MI and Random Forest used a deterministic 60,000-row stratified training sample; RFE used 20,000 rows. The GA and context-free NSGA-II used the same saved 10,000-row training-only search sample and 30-round search model. All final classifiers used the complete training partition and baseline 300-round classifier.

### CAMO-FS ablation interpretation

`results/final/camo_fs_ablation.csv` compares CAMO-FS with All_261, a single-objective GA, and a context-free NSGA-II. These deterministic comparisons describe the tested procedures; they do not isolate causal effects. The context-free NSGA-II retains ECI as an objective but uses no SHAP prior for initialization or mutation. Because all methods fix K=5, subset-size is constant; observed five-feature subset ECI was 1.0, and the context-free Pareto archive collapsed to one solution. Its second and third objectives were therefore degenerate in this comparison. Its search budget is also smaller than CAMO-FS's, so performance differences do not establish causal superiority of SHAP guidance.

## 9. Class-wise performance

The class-wise table includes all-feature, CAMO-FS, and the strongest non-CAMO selector ({strongest}), selected post hoc by its single test Macro-F1 for descriptive reporting only. This did not tune any selector.

| Method | Class | F1 | Test support |
|---|---|---:|---:|
{chr(10).join(comparison_rows)}

## 10. Limitations

- UCI provides two acquisition dates, not three verified phenological stages. EARLY/TRANSITION/LATE are operational temporal representations.
- The split is random stratified and IID-style, not spatially independent. Original rows are class-block ordered; field/spatial IDs are unavailable.
- Exact feature-vector overlaps with conflicting labels were found in the integrity audit; the current split cannot establish field-level generalization.
- Evolutionary search used a 10,000-row, one-holdout, 30-round surrogate for candidate fitness. The rare Broadleaf class had 35 rows in that sample, including 7 in its validation partition. The selected feature set is conditional on this approximation.
- The context-free NSGA-II archive contained one solution after each generation. With K fixed at five, subset-size is constant, and the observed subset ECI was 1.0, leaving effectively only classification error to distinguish candidates. This is a degenerate and limited multi-objective comparator.
- Only one deterministic optimization and one final test evaluation were performed. No Wilcoxon tests or claims of statistically significant superiority are made.
- The SHAP 0.48.0/XGBoost 3.2.0 compatibility shim substitutes a scalar only in SHAP's expected-value parser; feature contributions were checked against native TreeSHAP on a fixed sample. Fold ECI measures consistency of the saved rankings, not refit-model stability.
- A bootstrap interval below quantifies row-resampling uncertainty for this one test prediction vector; it is not uncertainty across independent experiments.
- The test set is one fixed holdout and was used for the final comparisons; no further method tuning should use these results.

## 11. Conclusion

On this fixed random holdout, CAMO-FS reduced the input from 261 to five features and achieved test Macro-F1 {camo['metrics']['macro_f1']:.6f}. The selected subset was dominated by TRANSITION and LATE operational representations. The deterministic comparison describes this dataset split and these implementation choices only; it does not establish universal superiority, biological-stage discovery, spatial generalization, or statistically significant differences.

### Bootstrap interval for CAMO-FS test predictions

Method: {bootstrap['method']}; {bootstrap['replicates']} replicates, seed {bootstrap['seed']}.

- Accuracy 95% percentile CI: [{bootstrap['accuracy_95_percentile_ci'][0]:.6f}, {bootstrap['accuracy_95_percentile_ci'][1]:.6f}]
- Macro-F1 95% percentile CI: [{bootstrap['macro_f1_95_percentile_ci'][0]:.6f}, {bootstrap['macro_f1_95_percentile_ci'][1]:.6f}]

## Artifacts

Machine-readable metrics, predictions, per-class reports, confusion matrices, selector configurations, and plots are under `results/final/`. The fixed split and raw dataset were not modified.
"""
    (ROOT / "docs/final_results.md").write_text(report, encoding="utf-8")


def validate_final_outputs(data: dict[str, Any], records: dict[str, dict[str, Any]], outputs: dict[str, Any]) -> None:
    if outputs["main"].method.tolist() != METHOD_ORDER:
        raise ValueError("Final comparison table is missing methods or has an unexpected order.")
    numeric = outputs["main"]["accuracy macro_f1 weighted_f1 kappa eci runtime".split()].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Final comparison has NaN or infinite metrics.")
    expected_test_digest = sha256_array(data["test_ids"])
    for method in METHOD_ORDER[1:]:
        record = records[method]
        if record["test_indices_sha256"] != expected_test_digest or not record["test_evaluated_once"]:
            raise ValueError(f"Test-index/evaluation provenance failed for {method}.")
        if record["test_set_used_for_selection_or_tuning"]:
            raise ValueError(f"Test-set selection leakage flag found for {method}.")
        if len(record["selected_features"]) != K:
            raise ValueError(f"{method} did not select exactly {K} features.")
        pred = np.load(METHOD_DIR / f"{method}_test_predictions.npz", allow_pickle=False)
        if not np.array_equal(pred["test_indices"], data["test_ids"]):
            raise ValueError(f"Prediction rows do not match saved test IDs for {method}.")
        if not np.isfinite(list(record["metrics"].values())).all():
            raise ValueError(f"Nonfinite metrics for {method}.")
    final = json.loads((CAMO_DIR / "final_solution.json").read_text(encoding="utf-8"))
    if records["CAMO_FS_5"]["selected_features"] != final["selected_feature_names"]:
        raise ValueError("CAMO-FS test evaluation selected-feature list differs from final_solution.json.")
    for path in [OUT / "main_comparison.csv", OUT / "representation_ablation.csv", OUT / "camo_fs_ablation.csv", OUT / "classwise_comparison.csv", OUT / "selected_feature_interpretation.csv"]:
        pd.read_csv(path)
    for path in [OUT / "final_summary.json", OUT / "camo_fs_test_metrics.json", OUT / "camo_fs_test_bootstrap_ci.json"]:
        json.loads(path.read_text(encoding="utf-8"))
    required_figures = ["macro_f1_by_method.png", "accuracy_by_method.png", "feature_count_reduction_by_method.png", "eci_by_method.png", "camo_fs_feature_distribution.png", "camo_fs_confusion_matrix.png", "camo_fs_pareto_front.png"]
    if any(not (FIGURES / name).is_file() or (FIGURES / name).stat().st_size == 0 for name in required_figures):
        raise ValueError("One or more required final figures are missing or empty.")
    audit = json.loads((ROOT / "docs/data_audit.json").read_text(encoding="utf-8"))
    if sha256_file(ROOT / "data/raw/crop+mapping+using+fused+optical+radar+data+set.zip") != audit["zip_sha256"]:
        raise ValueError("Raw dataset ZIP changed during final evaluation.")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    baseline_json = json.loads((BASELINE_DIR / "metrics.json").read_text(encoding="utf-8"))
    baseline_class = pd.read_csv(BASELINE_DIR / "classification_report.csv")
    baseline_classwise = [
        {"class_label": int(row["class_label"]), "class_name": CROP_NAMES[int(row["class_label"])], "precision": float(row["precision"]), "recall": float(row["recall"]), "f1": float(row["f1-score"]), "support": int(row["support"])}
        for row in baseline_class.to_dict(orient="records")
    ]
    baseline_record = {
        "metrics": baseline_json["metrics"],
        "classwise": baseline_classwise,
        "runtime_seconds": baseline_json["runtime_seconds"],
        "selected_feature_count": 261,
    }
    results: dict[str, dict[str, Any]] = {}

    # Critical first operation: test once for the already-finalized CAMO-FS subset.
    final_solution = json.loads((CAMO_DIR / "final_solution.json").read_text(encoding="utf-8"))
    camo_names = final_solution["selected_feature_names"]
    if camo_names != ["TRANSITION_CH051_G", "TRANSITION_CH055_NDVI", "LATE_CH003_sigVV", "LATE_CH015_Rvv", "LATE_CH027_l3"]:
        raise ValueError("CAMO-FS selected feature names do not match the approved saved final_solution.json.")
    camo_optimization_time = float(json.loads((CAMO_DIR / "runtime.json").read_text(encoding="utf-8"))["total_seconds"])
    results["CAMO_FS_5"] = evaluate_test_once("CAMO_FS_5", camo_names, data, camo_optimization_time, {"selection_source": "saved results/camo_fs/final_solution.json", "optimization_runtime_included": camo_optimization_time, "full_training_cv_macro_f1": final_solution["macro_f1"], "test_used_for_selection": False})

    # Shared training-only selector sample for classical selectors.
    selector_ids = training_selection_sample(data)
    selections: dict[str, tuple[list[str], float, dict[str, Any]]] = {}
    for method, function in [("Mutual_Information_5", select_mi), ("RandomForest_5", select_rf), ("RFE_5", select_rfe)]:
        cache_path = SELECTION_DIR / f"{method}_selection.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            selection = (cached["selected_features"], cached["selection_runtime_seconds"], cached["details"])
        else:
            selection = function(data, selector_ids)
            write_json(cache_path, {"selected_features": selection[0], "selection_runtime_seconds": selection[1], "details": selection[2]})
        selections[method] = selection
        results[method] = evaluate_test_once(method, selection[0], data, selection[1], selection[2])

    evaluator, search_info = make_search_evaluator(data)
    ga_cache = SELECTION_DIR / "SingleObjectiveGA_5_selection.json"
    if ga_cache.exists():
        cached = json.loads(ga_cache.read_text(encoding="utf-8"))
        ga_selection = (cached["selected_features"], cached["selection_runtime_seconds"], cached["details"])
    else:
        ga_selection = ga_select(data, evaluator, search_info)
        write_json(ga_cache, {"selected_features": ga_selection[0], "selection_runtime_seconds": ga_selection[1], "details": ga_selection[2]})
    results["SingleObjectiveGA_5"] = evaluate_test_once("SingleObjectiveGA_5", ga_selection[0], data, ga_selection[1], ga_selection[2])

    context_cache = SELECTION_DIR / "ContextFreeNSGAII_5_selection.json"
    if context_cache.exists():
        cached = json.loads(context_cache.read_text(encoding="utf-8"))
        context_selection = (cached["selected_features"], cached["selection_runtime_seconds"], cached["details"])
    else:
        context_evaluator, context_search_info = make_search_evaluator(data)
        context_selection = context_free_nsga_select(data, context_evaluator, context_search_info)
        write_json(context_cache, {"selected_features": context_selection[0], "selection_runtime_seconds": context_selection[1], "details": context_selection[2]})
    results["ContextFreeNSGAII_5"] = evaluate_test_once("ContextFreeNSGAII_5", context_selection[0], data, context_selection[1], context_selection[2])

    # Re-read method checkpoints so restart-skipped methods remain in the comparison.
    for method in METHOD_ORDER[1:]:
        path = METHOD_DIR / f"{method}.json"
        if path.exists():
            results[method] = json.loads(path.read_text(encoding="utf-8"))
    outputs = create_comparison_outputs(data, results, baseline_record)
    bootstrap = bootstrap_camo(METHOD_DIR / "CAMO_FS_5_test_predictions.npz")
    write_json(OUT / "camo_fs_test_bootstrap_ci.json", bootstrap)
    make_figures(outputs, results, data)

    chosen = outputs["interpretation"][outputs["interpretation"].selected]
    count_by_sensor = chosen.sensor.value_counts().to_dict()
    count_by_rep = chosen.representation.value_counts().reindex(["EARLY", "TRANSITION", "LATE"], fill_value=0).to_dict()
    final_solution = json.loads((CAMO_DIR / "final_solution.json").read_text(encoding="utf-8"))
    summary = {
        "dataset": {"source": "UCI Crop Mapping using Fused Optical-Radar Dataset #525", "rows": int(len(data["X"])), "feature_count": 261, "class_count": 7, "class_names": {str(k): v for k, v in CROP_NAMES.items()}, "representations": {"EARLY": "July 5", "TRANSITION": "July 14 - July 5", "LATE": "July 14"}, "biological_stage_mapping_verified": False},
        "split": {"method": data["split"]["split_method"], "seed": SEED, "train_rows": len(data["train_ids"]), "test_rows": len(data["test_ids"]), "class_distributions": {key: data["split"][key] for key in ("full_class_distribution", "train_class_distribution", "test_class_distribution")}, "train_indices_sha256": sha256_array(data["train_ids"]), "test_indices_sha256": sha256_array(data["test_ids"]), "test_touched_only_for_final_evaluation": True},
        "baseline_metrics": baseline_record,
        "camo_fs_metrics": results["CAMO_FS_5"],
        "all_comparison_metrics": {row.method: {"selected_feature_count": int(row.selected_feature_count), "feature_reduction_percent": float(row.feature_reduction_percent), "accuracy": float(row.accuracy), "macro_f1": float(row.macro_f1), "weighted_f1": float(row.weighted_f1), "kappa": float(row.kappa), "eci": float(row.eci), "runtime_seconds": float(row.runtime), "selected_features": row.selected_features.split(";") if row.selected_features else []} for row in outputs["main"].itertuples(index=False)},
        "selected_features": final_solution["selected_feature_names"],
        "selected_feature_count": 5,
        "representation_counts": {key: int(value) for key, value in count_by_rep.items()},
        "sensor_counts": {key: int(value) for key, value in count_by_sensor.items()},
        "ECI": {"overall": json.loads((ROOT / "results/eci/eci_summary.json").read_text(encoding="utf-8"))["overall_eci"], "selected_subset": results["CAMO_FS_5"]["ECI"]},
        "feature_reduction_percent": (1 - 5 / 261) * 100,
        "runtimes_seconds": {method: float(row.runtime) for method, row in outputs["main"].set_index("method").iterrows()},
        "bootstrap_ci": bootstrap,
        "strongest_baseline_selector_by_single_test_macro_f1": outputs["strongest_baseline_selector"],
        "random_seed": SEED,
        "limitations": [
            "The source contains two acquisition dates, not three verified phenological stages; EARLY/TRANSITION/LATE are operational temporal representations.",
            "Random stratified split is IID-style, not spatially independent; source rows are class-block ordered and spatial/field IDs are unavailable.",
            "Exact feature-vector overlaps with conflicting labels exist.",
            "CAMO-FS evolutionary search used a 10,000-row, one-holdout, 30-round surrogate objective; Broadleaf had 35 rows in that sample.",
            "The fixed-K context-free NSGA-II archive collapsed to one solution; subset size was constant and observed subset ECI was 1.0, so its multi-objective comparison was degenerate.",
            "Only one deterministic optimization run was performed; no statistical significance test is claimed.",
            "The SHAP/XGBoost compatibility shim affects expected-value bookkeeping; feature contributions were checked against native TreeSHAP on a deterministic sample.",
            "Bootstrap intervals describe row resampling of one test prediction vector, not independent runs.",
            "No spatial generalization or universal superiority claim is supported.",
        ],
        "software": {"python": platform.python_version(), "numpy": pkg_version("numpy"), "pandas": pkg_version("pandas"), "scikit_learn": pkg_version("scikit-learn"), "xgboost": pkg_version("xgboost"), "matplotlib": pkg_version("matplotlib")},
    }
    write_json(OUT / "final_summary.json", summary)
    write_final_report(data, results, baseline_record, outputs, bootstrap)
    validate_final_outputs(data, results, outputs)
    print("Final CAMO-FS comparison complete")
    print(json.dumps({"CAMO_FS_test": results["CAMO_FS_5"]["metrics"], "selected_features": camo_names, "comparison": outputs["main"].to_dict(orient="records"), "bootstrap_ci": bootstrap, "strongest_baseline_selector": outputs["strongest_baseline_selector"], "output_dir": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
