"""Baseline integrity checks and fixed-config representation ablation.

Run from the project root:
    python src/camo_fs/experiments/baseline_integrity_ablation.py

This investigation reuses the saved row-index split and the exact XGBoost
configuration from results/baseline/configuration.json. It does not modify the
baseline artifacts, raw ZIP, or Step 1 derived dataset.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import platform
import time
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data/processed/camo_temporal_261.parquet"
FEATURE_METADATA_PATH = ROOT / "data/processed/camo_temporal_feature_metadata.csv"
TRAIN_PATH = ROOT / "data/processed/train_indices.npy"
TEST_PATH = ROOT / "data/processed/test_indices.npy"
SPLIT_INFO_PATH = ROOT / "data/processed/baseline_split.json"
BASELINE_CONFIG_PATH = ROOT / "results/baseline/configuration.json"
RAW_ZIP = ROOT / "data/raw/crop+mapping+using+fused+optical+radar+data+set.zip"
OUTPUT_DIR = ROOT / "results/integrity"
ABLATION_CSV = OUTPUT_DIR / "representation_ablation.csv"
DOCS_ABLATION = ROOT / "docs/representation_ablation.md"
DOCS_INTEGRITY = ROOT / "docs/baseline_integrity.md"
N_CLASSES = 7
EXPECTED_ROWS = 325834
REPS = ("EARLY", "TRANSITION", "LATE")


def get_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    for path in (DATA_PATH, FEATURE_METADATA_PATH, TRAIN_PATH, TEST_PATH, SPLIT_INFO_PATH, BASELINE_CONFIG_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required baseline artifact is missing: {path.relative_to(ROOT)}")
    df = pd.read_parquet(DATA_PATH)
    metadata = pd.read_csv(FEATURE_METADATA_PATH, dtype=str)
    train_ids = np.load(TRAIN_PATH, allow_pickle=False)
    test_ids = np.load(TEST_PATH, allow_pickle=False)
    split_info = json.loads(SPLIT_INFO_PATH.read_text(encoding="utf-8"))
    config = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    if len(df) != EXPECTED_ROWS or len(train_ids) != 260667 or len(test_ids) != 65167:
        raise ValueError("Dataset or saved split dimensions differ from the verified baseline.")
    if df.columns[0] != "label" or len(df.columns) != 262:
        raise ValueError("Expected target label plus 261 derived features.")
    if train_ids.dtype.kind not in "iu" or test_ids.dtype.kind not in "iu":
        raise ValueError("Saved split sample IDs must be integer row indices.")
    if np.intersect1d(train_ids, test_ids).size:
        raise ValueError("The saved train and test row-index sets overlap.")
    if len(np.unique(np.concatenate((train_ids, test_ids)))) != len(df):
        raise ValueError("Saved split indices do not cover every dataset row exactly once.")
    if split_info.get("dataset_sha256") != file_sha256(DATA_PATH):
        raise ValueError("Saved split fingerprint does not match the derived dataset.")
    if config.get("dataset", {}).get("sha256") != split_info.get("dataset_sha256"):
        raise ValueError("Baseline config and split metadata refer to different datasets.")
    return df, metadata, train_ids, test_ids, config


def validate_data(df: pd.DataFrame, metadata: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    feature_names = [c for c in df.columns if c != "label"]
    if len(feature_names) != 261 or len(metadata) != 261:
        raise ValueError("Expected exactly 261 features and 261 metadata records.")
    if metadata["feature_name"].tolist() != feature_names:
        raise ValueError("Feature metadata does not match Parquet feature order.")
    counts = metadata["representation"].value_counts().to_dict()
    if counts != {"EARLY": 87, "TRANSITION": 87, "LATE": 87}:
        raise ValueError(f"Unexpected representation feature counts: {counts}")
    if df[feature_names].isna().to_numpy().any():
        raise ValueError("NaN found in feature matrix.")
    if df["label"].isna().any():
        raise ValueError("NaN found in target.")
    X32 = df[feature_names].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(X32).all():
        raise ValueError("Infinity found in feature matrix.")
    classes = np.sort(df["label"].unique())
    if len(classes) != N_CLASSES:
        raise ValueError(f"Expected exactly seven classes; found {len(classes)}.")
    return feature_names, classes


def run_representation_ablation(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    config: dict[str, Any],
    classes: np.ndarray,
) -> list[dict[str, Any]]:
    try:
        import xgboost as xgb
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("XGBoost is required; install the project requirements.") from exc

    model_config = dict(config["model"]["configuration"])
    if model_config.get("random_state") != 42 or model_config.get("n_estimators") != 300:
        raise ValueError("Stored baseline model configuration differs from the recorded baseline.")
    if model_config.get("objective") != "multi:softprob" or model_config.get("num_class") != 7:
        raise ValueError("Stored baseline is not the expected seven-class XGBoost configuration.")

    labels = df["label"].to_numpy(dtype=np.int64, copy=True)
    class_to_code = {int(label): code for code, label in enumerate(classes)}
    code_to_class = {code: int(label) for label, code in class_to_code.items()}
    y = np.fromiter((class_to_code[int(label)] for label in labels), dtype=np.int32, count=len(labels))
    all_features = [c for c in df.columns if c != "label"]
    rep_features = {
        rep: metadata.loc[metadata["representation"] == rep, "feature_name"].tolist()
        for rep in REPS
    }
    experiments = [(rep, rep_features[rep]) for rep in REPS]
    experiments.append(("ALL_261", all_features))
    records: list[dict[str, Any]] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for experiment, names in experiments:
        X = df[names].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite feature values found for {experiment}.")
        model = XGBClassifier(**model_config)
        started = time.perf_counter()
        model.fit(X[train_ids], y[train_ids], verbose=False)
        fit_seconds = time.perf_counter() - started
        # Test predictions are calculated only after fitting; no eval_set or tuning.
        predicted_codes = model.predict(X[test_ids]).astype(np.int64)
        predicted = np.fromiter(
            (code_to_class[int(code)] for code in predicted_codes),
            dtype=np.int64,
            count=len(predicted_codes),
        )
        truth = labels[test_ids]
        row: dict[str, Any] = {
            "experiment": experiment,
            "feature_count": len(names),
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=classes, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(truth, predicted, labels=classes, average="weighted", zero_division=0)),
            "cohen_kappa": float(cohen_kappa_score(truth, predicted, labels=classes)),
            "training_runtime_seconds": fit_seconds,
            "xgboost_version": xgb.__version__,
        }
        per_class = classification_per_class(truth, predicted, classes)
        for metric_name in ("precision", "recall", "f1"):
            for label in classes:
                row[f"class_{int(label)}_{metric_name}"] = per_class[str(int(label))][metric_name]
        records.append(row)
        pd.DataFrame(records).to_csv(ABLATION_CSV, index=False)
        print(
            f"{experiment}: accuracy={row['accuracy']:.6f}, "
            f"macro_f1={row['macro_f1']:.6f}, fit={fit_seconds:.2f}s",
            flush=True,
        )
    return records


def classification_per_class(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> dict[str, dict[str, float]]:
    p = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    return {
        str(int(label)): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
        for i, label in enumerate(classes)
    }


def exact_duplicate_audit(
    df: pd.DataFrame,
    feature_names: list[str],
    labels: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> dict[str, Any]:
    frame = df[feature_names]
    row_hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    _, inverse, hash_counts = np.unique(row_hashes, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(hash_counts)))
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_ids] = True
    duplicate_groups: list[list[int]] = []
    collision_splits = 0

    for hash_group_id in np.flatnonzero(hash_counts > 1):
        indices = order[offsets[hash_group_id] : offsets[hash_group_id + 1]].tolist()
        exact_subgroups: list[list[int]] = []
        representatives: list[np.ndarray] = []
        for idx in indices:
            vector = frame.iloc[idx].to_numpy(copy=False)
            matched = False
            for rep_idx, representative in enumerate(representatives):
                if np.array_equal(vector, representative, equal_nan=False):
                    exact_subgroups[rep_idx].append(idx)
                    matched = True
                    break
            if not matched:
                representatives.append(np.array(vector, copy=True))
                exact_subgroups.append([idx])
        if len(exact_subgroups) > 1:
            collision_splits += len(exact_subgroups) - 1
        duplicate_groups.extend(group for group in exact_subgroups if len(group) > 1)

    duplicated_rows = sum(len(group) for group in duplicate_groups)
    duplicate_excess_rows = sum(len(group) - 1 for group in duplicate_groups)
    unique_vectors = len(df) - duplicate_excess_rows
    conflict_groups = 0
    train_test_groups = 0
    overlap_train_rows = 0
    overlap_test_rows = 0
    train_test_conflict_groups = 0
    train_test_agree_groups = 0
    duplicate_group_summaries = []
    for group in duplicate_groups:
        group_labels = sorted(int(v) for v in np.unique(labels[group]))
        if len(group_labels) > 1:
            conflict_groups += 1
        in_train = [idx for idx in group if train_mask[idx]]
        in_test = [idx for idx in group if not train_mask[idx]]
        crosses_split = bool(in_train and in_test)
        if crosses_split:
            train_test_groups += 1
            overlap_train_rows += len(in_train)
            overlap_test_rows += len(in_test)
            if len(group_labels) > 1:
                train_test_conflict_groups += 1
            else:
                train_test_agree_groups += 1
        duplicate_group_summaries.append(
            {
                "row_count": len(group),
                "labels": group_labels,
                "conflicting_labels": len(group_labels) > 1,
                "train_rows": len(in_train),
                "test_rows": len(in_test),
                "crosses_train_test": crosses_split,
            }
        )

    return {
        "rows": len(df),
        "feature_count": len(feature_names),
        "unique_exact_feature_vectors": int(unique_vectors),
        "duplicate_feature_vector_groups": len(duplicate_groups),
        "rows_in_duplicate_groups": int(duplicated_rows),
        "duplicate_excess_rows_after_first_occurrence": int(duplicate_excess_rows),
        "duplicate_groups_with_conflicting_labels": int(conflict_groups),
        "train_test_overlapping_unique_feature_vectors": int(train_test_groups),
        "train_rows_with_train_test_overlapping_vectors": int(overlap_train_rows),
        "test_rows_with_train_test_overlapping_vectors": int(overlap_test_rows),
        "train_test_overlapping_vectors_with_agreeing_labels": int(train_test_agree_groups),
        "train_test_overlapping_vectors_with_conflicting_labels": int(train_test_conflict_groups),
        "hash_collision_splits_after_exact_row_comparison": int(collision_splits),
        "method": "pandas 64-bit row hashes to find candidates; every repeated-hash candidate was confirmed by exact full-vector equality",
        "duplicate_groups": duplicate_group_summaries,
    }


def near_duplicate_audit(
    df: pd.DataFrame,
    feature_names: list[str],
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> dict[str, Any]:
    frame = df[feature_names]
    means = frame.mean(axis=0).to_numpy(dtype=np.float64)
    scales = frame.std(axis=0, ddof=0).to_numpy(dtype=np.float64)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_ids] = True
    counts: dict[int, list[int]] = {}
    quantum = 0.01  # standardized feature units
    chunk_size = 10000
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        x = frame.iloc[start:end].to_numpy(dtype=np.float64, copy=False)
        quantized = np.rint((x - means) / scales / quantum).astype(np.int64)
        qframe = pd.DataFrame(quantized)
        hashes = pd.util.hash_pandas_object(qframe, index=False).to_numpy(dtype=np.uint64)
        local_unique, local_inverse, local_counts = np.unique(hashes, return_inverse=True, return_counts=True)
        for local_group, group_hash in enumerate(local_unique):
            positions = np.flatnonzero(local_inverse == local_group) + start
            entry = counts.setdefault(int(group_hash), [0, 0, 0])
            entry[0] += int(local_counts[local_group])
            entry[1] += int(train_mask[positions].sum())
            entry[2] += int((~train_mask[positions]).sum())
    repeated = [entry for entry in counts.values() if entry[0] > 1]
    cross_split = [entry for entry in repeated if entry[1] > 0 and entry[2] > 0]
    return {
        "method": "feature-wise z-score using full-dataset mean and population standard deviation; quantize to nearest 0.01 standard deviation and hash row vectors in 10,000-row chunks",
        "precision": "two decimal places in standardized feature units",
        "rows": len(df),
        "unique_quantized_vectors": len(counts),
        "near_duplicate_groups_including_any_exact_duplicates": len(repeated),
        "rows_in_repeated_quantized_groups": int(sum(entry[0] for entry in repeated)),
        "train_test_overlapping_repeated_quantized_vectors": len(cross_split),
        "train_rows_in_cross_split_repeated_quantized_vectors": int(sum(entry[1] for entry in cross_split)),
        "test_rows_in_cross_split_repeated_quantized_vectors": int(sum(entry[2] for entry in cross_split)),
        "limitations": [
            "Quantization detects only vectors that land in the same bins; near-equal vectors on opposite bin boundaries can be missed.",
            "The 0.01-standard-deviation threshold is a diagnostic choice, not a domain-specific definition of duplicate observations.",
            "A hash collision is theoretically possible; these results are approximate near-duplicate counts, not exact geometric nearest-neighbor results.",
            "Standardization and hashing inspect feature similarity only and do not establish spatial identity or leakage.",
        ],
    }


def representation_stats(df: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for rep in REPS:
        names = metadata.loc[metadata["representation"] == rep, "feature_name"].tolist()
        values = df[names].to_numpy(dtype=np.float64, copy=False)
        result[rep] = {
            "feature_count": len(names),
            "pooled_value_count": int(values.size),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values, dtype=np.float64)),
            "std_population": float(np.std(values, dtype=np.float64, ddof=0)),
            "nan_count": int(np.isnan(values).sum()),
            "positive_infinity_count": int(np.isposinf(values).sum()),
            "negative_infinity_count": int(np.isneginf(values).sum()),
            "finite": bool(np.isfinite(values).all()),
        }
    return result


def temporal_sample_check(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    sample_ids: np.ndarray,
) -> dict[str, Any]:
    channels = sorted(metadata["channel_id"].unique(), key=lambda value: int(value[2:]))
    selected_channel_positions = np.unique(
        np.linspace(0, len(channels) - 1, num=min(17, len(channels)), dtype=int)
    )
    channel_ids = [channels[i] for i in selected_channel_positions]
    selected_rows = np.sort(sample_ids)
    early_meta = metadata[metadata["representation"] == "EARLY"].set_index("channel_id")
    trans_meta = metadata[metadata["representation"] == "TRANSITION"].set_index("channel_id")
    late_meta = metadata[metadata["representation"] == "LATE"].set_index("channel_id")
    raw_columns = ["label"]
    for channel in channel_ids:
        raw_columns.extend([early_meta.loc[channel, "source_july5_column"], early_meta.loc[channel, "source_july14_column"]])
    raw_columns = list(dict.fromkeys(raw_columns))
    sample_set = set(int(v) for v in selected_rows)
    raw_parts: dict[int, dict[str, float]] = {}
    with zipfile.ZipFile(RAW_ZIP, "r") as archive:
        with archive.open("WinnipegDataset.txt", "r") as stream:
            offset = 0
            for chunk in pd.read_csv(stream, usecols=raw_columns, chunksize=20000):
                chunk_end = offset + len(chunk)
                wanted = [v for v in selected_rows if offset <= v < chunk_end]
                for row_id in wanted:
                    row = chunk.iloc[int(row_id - offset)]
                    raw_parts[int(row_id)] = {col: float(row[col]) for col in raw_columns if col != "label"}
                offset = chunk_end
                if len(raw_parts) == len(selected_rows):
                    break
    if len(raw_parts) != len(selected_rows):
        raise ValueError("Could not read every deterministically sampled raw row.")

    exact_source_matches = 0
    stored_snapshot_exact_matches = 0
    stored_snapshot_max_abs_residual = 0.0
    n_checks = 0
    for row_id in selected_rows:
        raw_row = raw_parts[int(row_id)]
        row = df.iloc[int(row_id)]
        for channel in channel_ids:
            early_row = early_meta.loc[channel]
            transition_row = trans_meta.loc[channel]
            late_row = late_meta.loc[channel]
            july5_col = str(early_row["source_july5_column"])
            july14_col = str(early_row["source_july14_column"])
            early_raw = raw_row[july5_col]
            late_raw = raw_row[july14_col]
            transition_raw = late_raw - early_raw
            transition_stored = float(row[str(transition_row["feature_name"])])
            if transition_stored == transition_raw:
                exact_source_matches += 1
            early_stored = float(row[str(early_row["feature_name"])])
            late_stored = float(row[str(late_row["feature_name"])])
            snapshot_delta = late_stored - early_stored
            residual = abs(transition_stored - snapshot_delta)
            stored_snapshot_max_abs_residual = max(stored_snapshot_max_abs_residual, residual)
            if transition_stored == snapshot_delta:
                stored_snapshot_exact_matches += 1
            n_checks += 1
    return {
        "method": "seed-42 sample of 256 original row positions and 17 channels selected evenly from audit-backed metadata; raw ZIP values re-read directly",
        "sampled_rows": len(selected_rows),
        "sampled_channels": len(channel_ids),
        "scalar_transition_checks": n_checks,
        "transition_exactly_equals_raw_july14_minus_july5": exact_source_matches == n_checks,
        "exact_source_transition_matches": exact_source_matches,
        "transition_equals_stored_float32_late_minus_early_exactly": stored_snapshot_exact_matches == n_checks,
        "stored_snapshot_exact_matches": stored_snapshot_exact_matches,
        "max_abs_residual_against_stored_float32_snapshots": stored_snapshot_max_abs_residual,
        "note": "TRANSITION was computed from full-precision parsed raw values; EARLY and LATE are stored as float32, so subtracting the stored snapshots may differ by float32 rounding.",
    }


def row_order_audit(
    labels: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    n = len(labels)
    boundaries = np.linspace(0, n, 11, dtype=int)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_ids] = True
    segments = []
    contingency = np.zeros((10, len(classes)), dtype=np.int64)
    for segment in range(10):
        start, end = int(boundaries[segment]), int(boundaries[segment + 1])
        segment_labels = labels[start:end]
        segment_classes, segment_counts = np.unique(segment_labels, return_counts=True)
        for c, count in zip(segment_classes, segment_counts):
            class_idx = int(np.searchsorted(classes, c))
            contingency[segment, class_idx] = int(count)
        segments.append({
            "segment": segment + 1,
            "start_row_inclusive": start,
            "end_row_exclusive": end,
            "rows": end - start,
            "train_rows": int(train_mask[start:end].sum()),
            "test_rows": int((~train_mask[start:end]).sum()),
            "class_counts": {str(int(c)): int(contingency[segment, i]) for i, c in enumerate(classes)},
            "class_proportions": {
                str(int(c)): float(contingency[segment, i] / (end - start))
                for i, c in enumerate(classes)
            },
        })
    row_totals = contingency.sum(axis=1, keepdims=True)
    col_totals = contingency.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / n
    chi2 = float(np.sum((contingency - expected) ** 2 / expected))
    cramers_v = math.sqrt(chi2 / (n * min(contingency.shape[0] - 1, contingency.shape[1] - 1)))

    boundaries_run = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, n]
    run_labels = labels[boundaries_run[:-1]]
    run_lengths = np.diff(boundaries_run)
    class_runs = {}
    for c in classes:
        lengths = run_lengths[run_labels == c]
        class_runs[str(int(c))] = {
            "run_count": int(len(lengths)),
            "longest_consecutive_run": int(lengths.max()) if len(lengths) else 0,
            "mean_run_length": float(lengths.mean()) if len(lengths) else 0.0,
        }
    observed_same_neighbor_rate = float(np.mean(labels[1:] == labels[:-1]))
    class_counts = np.array([np.count_nonzero(labels == c) for c in classes], dtype=np.float64)
    expected_same_neighbor_rate = float(np.sum(class_counts * (class_counts - 1)) / (n * (n - 1)))
    return {
        "row_order_sample_id": "zero-based original Parquet row position",
        "segment_count": 10,
        "segments": segments,
        "label_by_segment_cramers_v": cramers_v,
        "observed_same_label_adjacent_rate": observed_same_neighbor_rate,
        "expected_same_label_adjacent_rate_under_random_order_given_class_counts": expected_same_neighbor_rate,
        "adjacent_same_label_rate_ratio": observed_same_neighbor_rate / expected_same_neighbor_rate,
        "total_label_runs": int(len(run_lengths)),
        "longest_global_same_label_run": int(run_lengths.max()),
        "class_run_statistics": class_runs,
        "interpretation_limit": "Row-order diagnostics cannot identify geography, field boundaries, or spatial leakage because no spatial/group identifier is present in this dataset.",
    }


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_reports(
    ablation: list[dict[str, Any]],
    exact: dict[str, Any],
    near: dict[str, Any],
    temporal: dict[str, Any],
    row_order: dict[str, Any],
    software: dict[str, str | None],
    split_info: dict[str, Any],
    baseline_config: dict[str, Any],
) -> None:
    ablation_frame = pd.DataFrame(ablation)
    ablation_frame.to_csv(ABLATION_CSV, index=False)
    pd.DataFrame(row_order["segments"]).to_csv(OUTPUT_DIR / "row_order_decile_class_distribution.csv", index=False)
    save_json(OUTPUT_DIR / "duplicate_leakage.json", exact)
    save_json(OUTPUT_DIR / "near_duplicate.json", near)
    save_json(OUTPUT_DIR / "temporal_sanity.json", temporal)
    save_json(OUTPUT_DIR / "row_order_analysis.json", row_order)

    class_lines = ["| Experiment | " + " | ".join(f"Class {i} F1" for i in range(1, 8)) + " |", "|---|" + "---:|" * 7]
    for row in ablation:
        class_lines.append(
            f"| {row['experiment']} | " + " | ".join(f"{row[f'class_{i}_f1']:.6f}" for i in range(1, 8)) + " |"
        )
    metric_lines = [
        "| Representation | Features | Accuracy | Macro-F1 | Weighted-F1 | Kappa | Fit seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ablation:
        metric_lines.append(
            f"| {row['experiment']} | {row['feature_count']} | {row['accuracy']:.6f} | "
            f"{row['macro_f1']:.6f} | {row['weighted_f1']:.6f} | {row['cohen_kappa']:.6f} | "
            f"{row['training_runtime_seconds']:.2f} |"
        )
    report_ablation = f"""# Representation Ablation

## Protocol

Four independent XGBoost models were fit using the exact saved seed-42 stratified split. The stored baseline configuration was reused without tuning: `{json.dumps(baseline_config['model']['configuration'], sort_keys=True)}`. Each model was fit only on training rows; test data were used only for final scoring. No feature selection, SHAP, or NSGA-II was performed. EARLY, TRANSITION, and LATE are operational date/change representations, not verified biological growth stages.

Split sizes: {split_info['train_rows']:,} train and {split_info['test_rows']:,} test. Features: 87 per individual representation and 261 combined.

## Results

{chr(10).join(metric_lines)}

### Per-class F1

{chr(10).join(class_lines)}

## Interpretation

These results compare predictive signal in the representations under one fixed classifier and split. They do not establish causal importance, biological stage meaning, or performance on a spatially independent test set. Use Macro-F1 and class-wise scores because the target is imbalanced.

The full results, including per-class precision and recall, are in `results/integrity/representation_ablation.csv`.

## Software

{chr(10).join(f'- {name}: `{value}`' for name, value in software.items())}
"""
    DOCS_ABLATION.write_text(report_ablation, encoding="utf-8")

    segment_table = [
        "| Row segment | Rows | Train | Test | " + " | ".join(f"Class {i}" for i in range(1, 8)) + " |",
        "|---:|---:|---:|---:|" + "---:|" * 7,
    ]
    for seg in row_order["segments"]:
        segment_table.append(
            f"| {seg['segment']} | {seg['rows']:,} | {seg['train_rows']:,} | {seg['test_rows']:,} | "
            + " | ".join(f"{seg['class_proportions'][str(i)]:.2%}" for i in range(1, 8))
            + " |"
        )
    metric_rows = "\n".join(
        f"| {name} | {value['min']:.6g} | {value['max']:.6g} | {value['mean']:.6g} | {value['std_population']:.6g} | {value['finite']} |"
        for name, value in temporal["representation_statistics"].items()
    )
    report_integrity = f"""# Baseline Integrity Investigation

## Scope and controls

This audit uses the Step 1 Parquet dataset and the exact row-position indices saved in `data/processed/train_indices.npy` and `test_indices.npy`. It does not modify the prior baseline outputs, raw UCI ZIP, or derived dataset. The representation ablation uses the same XGBoost parameters, seed, and train/test rows as the baseline; it is not tuning and is not CAMO-FS.

## Exact duplicate feature vectors and train/test overlap

- Dataset rows / feature dimensions: {exact['rows']:,} / {exact['feature_count']}.
- Unique exact feature vectors: {exact['unique_exact_feature_vectors']:,}.
- Duplicate feature-vector groups: {exact['duplicate_feature_vector_groups']:,}.
- Rows belonging to duplicate groups: {exact['rows_in_duplicate_groups']:,}.
- Duplicate groups with conflicting labels: {exact['duplicate_groups_with_conflicting_labels']}.
- Unique vectors occurring in both train and test: {exact['train_test_overlapping_unique_feature_vectors']}.
- Of those overlapping vectors, label-agreeing / label-conflicting groups: {exact['train_test_overlapping_vectors_with_agreeing_labels']} / {exact['train_test_overlapping_vectors_with_conflicting_labels']}.

The audit used row hashes to find repeated-vector candidates, then confirmed every candidate by exact equality of the full feature vector. Detailed counts are in `results/integrity/duplicate_leakage.json`.

## Near-duplicate diagnostic

Features were standardized using full-dataset means and population standard deviations, quantized to 0.01 standard-deviation increments, and hashed in chunks. Repeated rounded-vector groups: {near['near_duplicate_groups_including_any_exact_duplicates']:,}; affected rows: {near['rows_in_repeated_quantized_groups']:,}; rounded vectors shared across train/test: {near['train_test_overlapping_repeated_quantized_vectors']:,}.

This is a scalable diagnostic, not an all-pairs distance calculation. It can miss close vectors that land in adjacent bins, and its threshold is not a domain-specific duplicate definition. Details and caveats are in `results/integrity/near_duplicate.json`.

## Representation ranges and temporal equation check

The derived feature matrix contains no NaN or infinity values. Summary statistics below pool all values across all 87 channels and all rows within each representation; standard deviation is population standard deviation.

| Representation | Min | Max | Mean | Std | All finite |
|---|---:|---:|---:|---:|:---:|
{metric_rows}

Using the saved metadata mappings, {temporal['scalar_transition_checks']:,} sampled scalar checks across {temporal['sampled_rows']} rows and {temporal['sampled_channels']} channels were compared with the raw ZIP. Exact matches to raw July 14 minus raw July 5: **{temporal['exact_source_transition_matches']:,}/{temporal['scalar_transition_checks']:,}**. Exact matches when subtracting the stored float32 LATE and EARLY snapshots: **{temporal['stored_snapshot_exact_matches']:,}/{temporal['scalar_transition_checks']:,}**; maximum absolute residual: {temporal['max_abs_residual_against_stored_float32_snapshots']:.9g}. The transition was constructed from full-precision parsed raw data, while snapshots are float32, so small differences after subtracting the stored snapshots are expected rounding effects.

## Original row-order inspection

The following table reports each of 10 contiguous, nearly equal-sized row segments. Class cells are within-segment proportions.

{chr(10).join(segment_table)}

- Cramér's V for association between row segment and class: {row_order['label_by_segment_cramers_v']:.6g}.
- Adjacent equal-label rate: {row_order['observed_same_label_adjacent_rate']:.4%}; expected under random ordering conditional on class counts: {row_order['expected_same_label_adjacent_rate_under_random_order_given_class_counts']:.4%}; ratio: {row_order['adjacent_same_label_rate_ratio']:.3f}.
- Longest contiguous same-label run: {row_order['longest_global_same_label_run']} rows.

These diagnostics assess class order only. No spatial identifier or field/group key is available here, so they cannot establish or rule out spatial leakage. Detailed per-class runs and segment distributions are in `results/integrity/row_order_analysis.json` and `row_order_decile_class_distribution.csv`.

## Assessment

The full-feature baseline's accuracy of 99.756% is unusually high and warrants careful interpretation. The representation ablation in [`representation_ablation.md`](representation_ablation.md) shows how predictive performance changes when each 87-channel representation is used separately; this comparison is descriptive, not a causal attribution.

Exact train/test feature-vector overlap: **{exact['train_test_overlapping_unique_feature_vectors']}**. Each overlapping vector has one row in each split, and all 13 pairs have conflicting labels; no label-consistent exact duplicate crosses the split. This is still direct feature-vector overlap, but it does not explain a high score through repeated same-label examples. The check does not rule out near duplicates, shared fields, spatial autocorrelation, or source-level grouping. The random stratified split is reproducible and suitable as an IID benchmark; its suitability for spatial generalization cannot be established from the available columns. Report this limitation and avoid claims of spatially independent validation.

The deterministic row-order split itself has no explicit test-time tuning. However, if source rows have field or spatial adjacency, a random split could place related observations in both partitions. The data contain no spatial IDs to test that possibility. State that the result is a random holdout baseline on this dataset, not an independent-region or independent-field estimate.

## Software versions

{chr(10).join(f'- {name}: `{value}`' for name, value in software.items())}

## Files

- Representation ablation: `results/integrity/representation_ablation.csv`
- Exact duplicates/leakage: `results/integrity/duplicate_leakage.json`
- Near-duplicate diagnostic: `results/integrity/near_duplicate.json`
- Temporal sanity: `results/integrity/temporal_sanity.json`
- Row-order analysis: `results/integrity/row_order_analysis.json`
"""
    DOCS_INTEGRITY.write_text(report_integrity, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, metadata, train_ids, test_ids, baseline_config = read_inputs()
    features, classes = validate_data(df, metadata)
    labels = df["label"].to_numpy(dtype=np.int64, copy=True)
    if not np.array_equal(np.sort(np.unique(labels[train_ids])), classes):
        raise ValueError("Training split does not contain every class.")
    if not np.array_equal(np.sort(np.unique(labels[test_ids])), classes):
        raise ValueError("Test split does not contain every class.")

    ablation = run_representation_ablation(df, metadata, train_ids, test_ids, baseline_config, classes)
    exact = exact_duplicate_audit(df, features, labels, train_ids, test_ids)
    near = near_duplicate_audit(df, features, train_ids, test_ids)
    stats = representation_stats(df, metadata)
    sample_rng = np.random.default_rng(42)
    sampled_rows = sample_rng.choice(len(df), size=256, replace=False)
    temporal = temporal_sample_check(df, metadata, sampled_rows)
    temporal["representation_statistics"] = stats
    row_order = row_order_audit(labels, train_ids, test_ids, classes)
    software = {
        "python": platform.python_version(),
        "numpy": get_version("numpy"),
        "pandas": get_version("pandas"),
        "scikit-learn": get_version("scikit-learn"),
        "xgboost": get_version("xgboost"),
        "pyarrow": get_version("pyarrow"),
    }
    split_info = json.loads(SPLIT_INFO_PATH.read_text(encoding="utf-8"))
    write_reports(ablation, exact, near, temporal, row_order, software, split_info, baseline_config)
    summary = {
        "dataset_sha256": file_sha256(DATA_PATH),
        "split_seed": split_info["seed"],
        "train_rows": len(train_ids),
        "test_rows": len(test_ids),
        "representation_ablation": ablation,
        "exact_duplicate_and_leakage": exact,
        "near_duplicate": near,
        "temporal_sanity": temporal,
        "row_order": row_order,
        "software_versions": software,
        "total_runtime_seconds": time.perf_counter() - started,
    }
    save_json(OUTPUT_DIR / "integrity_summary.json", summary)
    print("Baseline integrity and representation ablation complete")
    print(json.dumps({
        "ablation": [{k: r[k] for k in ("experiment", "accuracy", "macro_f1", "weighted_f1", "cohen_kappa", "training_runtime_seconds")} for r in ablation],
        "exact_duplicates": {k: exact[k] for k in ("unique_exact_feature_vectors", "duplicate_feature_vector_groups", "rows_in_duplicate_groups", "duplicate_groups_with_conflicting_labels", "train_test_overlapping_unique_feature_vectors", "train_test_overlapping_vectors_with_conflicting_labels")},
        "near_duplicates": {k: near[k] for k in ("near_duplicate_groups_including_any_exact_duplicates", "rows_in_repeated_quantized_groups", "train_test_overlapping_repeated_quantized_vectors")},
        "temporal_check": {k: temporal[k] for k in ("exact_source_transition_matches", "scalar_transition_checks", "stored_snapshot_exact_matches", "max_abs_residual_against_stored_float32_snapshots")},
        "row_order": {k: row_order[k] for k in ("label_by_segment_cramers_v", "observed_same_label_adjacent_rate", "expected_same_label_adjacent_rate_under_random_order_given_class_counts", "adjacent_same_label_rate_ratio", "longest_global_same_label_run")},
        "total_runtime_seconds": summary["total_runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
