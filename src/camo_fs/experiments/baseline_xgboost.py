"""Reproducible full-feature XGBoost baseline for the temporal dataset.

Run from the repository root:
    python src/camo_fs/experiments/baseline_xgboost.py

The first run creates a seed-42 stratified 80/20 split using original row-order
sample IDs. Later runs reuse the saved indices and refuse to regenerate them.
The test set is not passed to model.fit and is evaluated only after fitting.
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data/processed/camo_temporal_261.parquet"
FEATURE_METADATA_PATH = PROJECT_ROOT / "data/processed/camo_temporal_feature_metadata.csv"
TRAIN_INDICES_PATH = PROJECT_ROOT / "data/processed/train_indices.npy"
TEST_INDICES_PATH = PROJECT_ROOT / "data/processed/test_indices.npy"
SPLIT_INFO_PATH = PROJECT_ROOT / "data/processed/baseline_split.json"
RESULTS_DIR = PROJECT_ROOT / "results/baseline"
REPORT_PATH = PROJECT_ROOT / "docs/baseline_results.md"
SEED = 42
TEST_SIZE = 0.20
EXPECTED_ROWS = 325834
EXPECTED_FEATURES = 261
EXPECTED_CLASSES = 7

MODEL_CONFIG: dict[str, Any] = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "n_jobs": -1,
    "tree_method": "hist",
    "verbosity": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def library_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def distribution(labels: np.ndarray, classes: np.ndarray) -> dict[str, dict[str, float | int]]:
    total = len(labels)
    counts = {int(c): int(np.count_nonzero(labels == c)) for c in classes}
    return {
        str(c): {"count": counts[int(c)], "proportion": counts[int(c)] / total}
        for c in classes
    }


def create_or_load_split(
    labels: np.ndarray, dataset_sha256: str, classes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    index_files_exist = (TRAIN_INDICES_PATH.exists(), TEST_INDICES_PATH.exists())
    if any(index_files_exist) and not all(index_files_exist):
        raise FileNotFoundError(
            "Only one saved split index file exists. Restore the matching train/test pair; "
            "the script will not silently create a replacement split."
        )

    if all(index_files_exist):
        if not SPLIT_INFO_PATH.exists():
            raise FileNotFoundError("Saved indices exist but baseline_split.json is missing.")
        info = json.loads(SPLIT_INFO_PATH.read_text(encoding="utf-8"))
        if info.get("dataset_sha256") != dataset_sha256:
            raise ValueError("Dataset hash differs from the dataset used to create saved indices.")
        if info.get("seed") != SEED or info.get("test_size") != TEST_SIZE:
            raise ValueError("Saved split metadata does not match the requested fixed split config.")
        train_ids = np.load(TRAIN_INDICES_PATH, allow_pickle=False)
        test_ids = np.load(TEST_INDICES_PATH, allow_pickle=False)
        if info.get("train_rows") != len(train_ids) or info.get("test_rows") != len(test_ids):
            raise ValueError("Saved split index lengths differ from baseline_split.json.")
        if info.get("sample_id_definition") != "zero-based original Parquet row position":
            raise ValueError("Saved split sample ID definition is not recognized.")
        created_new = False
    else:
        sample_ids = np.arange(len(labels), dtype=np.int64)
        train_ids, test_ids = train_test_split(
            sample_ids,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=labels,
        )
        train_ids = np.asarray(train_ids, dtype=np.int64)
        test_ids = np.asarray(test_ids, dtype=np.int64)
        created_new = True
        TRAIN_INDICES_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(TRAIN_INDICES_PATH, train_ids, allow_pickle=False)
        np.save(TEST_INDICES_PATH, test_ids, allow_pickle=False)

    if train_ids.ndim != 1 or test_ids.ndim != 1:
        raise ValueError("Saved split indices must be one-dimensional.")
    if len(train_ids) + len(test_ids) != len(labels):
        raise ValueError("Train and test indices do not cover the complete dataset.")
    if (
        np.any(train_ids < 0)
        or np.any(test_ids < 0)
        or np.any(train_ids >= len(labels))
        or np.any(test_ids >= len(labels))
    ):
        raise ValueError("Saved split contains an out-of-range sample ID.")
    if len(np.unique(train_ids)) != len(train_ids) or len(np.unique(test_ids)) != len(test_ids):
        raise ValueError("Saved train or test indices contain repeated sample IDs.")
    if np.intersect1d(train_ids, test_ids).size:
        raise ValueError("Train and test sample IDs overlap.")

    train_labels = labels[train_ids]
    test_labels = labels[test_ids]
    full_dist = distribution(labels, classes)
    train_dist = distribution(train_labels, classes)
    test_dist = distribution(test_labels, classes)
    for c in classes:
        expected_test = len(test_ids) / len(labels) * int(np.count_nonzero(labels == c))
        actual_test = int(np.count_nonzero(test_labels == c))
        if abs(actual_test - expected_test) > 1.0:
            raise ValueError(f"Test split does not preserve class {int(c)} proportions.")
    info = {
        "dataset_path": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "dataset_sha256": dataset_sha256,
        "split_method": "sklearn.model_selection.train_test_split with stratify=label",
        "seed": SEED,
        "test_size": TEST_SIZE,
        "train_rows": int(len(train_ids)),
        "test_rows": int(len(test_ids)),
        "sample_id_definition": "zero-based original Parquet row position",
        "sample_id_column_added_to_dataset": False,
        "indices_are_sample_ids": True,
        "train_indices_file": str(TRAIN_INDICES_PATH.relative_to(PROJECT_ROOT)),
        "test_indices_file": str(TEST_INDICES_PATH.relative_to(PROJECT_ROOT)),
        "full_class_distribution": full_dist,
        "train_class_distribution": train_dist,
        "test_class_distribution": test_dist,
        "created_on_this_run": created_new,
    }
    SPLIT_INFO_PATH.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return train_ids, test_ids, info


def validate_dataset(df: pd.DataFrame, metadata: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows; found {len(df)}.")
    if "label" not in df.columns:
        raise ValueError("Target column 'label' is missing.")
    features = [column for column in df.columns if column != "label"]
    if len(features) != EXPECTED_FEATURES:
        raise ValueError(f"Expected 261 features; found {len(features)}.")
    if len(metadata) != EXPECTED_FEATURES or metadata["feature_name"].duplicated().any():
        raise ValueError("Feature metadata must contain 261 unique feature records.")
    if features != metadata["feature_name"].tolist():
        raise ValueError("Parquet feature order differs from the feature metadata order.")
    counts = metadata["representation"].value_counts().to_dict()
    if counts != {"EARLY": 87, "TRANSITION": 87, "LATE": 87}:
        raise ValueError(f"Expected 87 features per representation; found {counts}.")
    if not pd.api.types.is_integer_dtype(df["label"].dtype):
        raise ValueError("Target label must have an integer dtype.")
    if df["label"].isna().any():
        raise ValueError("Target contains missing labels.")
    classes = np.sort(df["label"].unique())
    if len(classes) != EXPECTED_CLASSES:
        raise ValueError(f"Expected 7 target classes; found {len(classes)}.")
    if df[features].isna().to_numpy().any():
        raise ValueError("Feature matrix contains NaN values.")
    X = df[features].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains positive or negative infinity.")
    return features, classes


def write_markdown_report(metrics: dict[str, Any], class_report: pd.DataFrame) -> None:
    ds = metrics["dataset"]
    split = metrics["split"]
    model = metrics["model"]
    scores = metrics["metrics"]
    class_rows = class_report[class_report["class_label"].notna()]
    table_lines = [
        "| Class label | Precision | Recall | F1 | Support |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in class_rows.iterrows():
        table_lines.append(
            f"| {int(row['class_label'])} | {row['precision']:.6f} | "
            f"{row['recall']:.6f} | {row['f1-score']:.6f} | {int(row['support'])} |"
        )
    report = f"""# Full-Feature XGBoost Baseline

This is a reproducible full-feature baseline for the operational two-date temporal representation. EARLY, TRANSITION, and LATE are representation names, not verified biological growth stages. This is not a CAMO-FS feature-selection result.

## Dataset and split

- Dataset: `{ds['path']}`
- Dataset SHA-256: `{ds['sha256']}`
- Shape: {ds['rows']:,} rows × {ds['feature_count']} features plus integer target `label`
- Representation counts: EARLY={ds['representation_counts']['EARLY']}, TRANSITION={ds['representation_counts']['TRANSITION']}, LATE={ds['representation_counts']['LATE']}
- Sample IDs: zero-based original Parquet row positions; no ID column was added to the dataset.
- Split: stratified {100 * (1 - split['test_size']):.0f}% train / {100 * split['test_size']:.0f}% test, seed {split['seed']}; exact row positions are saved under `data/processed/` and reused on later runs.
- Sizes: {split['train_rows']:,} train and {split['test_rows']:,} test.
- Test observations were not supplied to `fit`; they were used only for final evaluation.

### Class distributions

| Label | Full count | Full share | Train count | Train share | Test count | Test share |
|---:|---:|---:|---:|---:|---:|---:|
{_distribution_markdown(split)}

## Model configuration

```json
{json.dumps(model['configuration'], indent=2)}
```

## Test metrics

- Accuracy: **{scores['accuracy']:.6f}**
- Macro-F1: **{scores['macro_f1']:.6f}**
- Weighted-F1: **{scores['weighted_f1']:.6f}**
- Cohen's Kappa: **{scores['cohen_kappa']:.6f}**
- Training runtime: {metrics['runtime_seconds']['fit']:.2f} seconds
- Total experiment runtime: {metrics['runtime_seconds']['total']:.2f} seconds

### Class-wise test performance

{chr(10).join(table_lines)}

The dataset is imbalanced, so Macro-F1 and the per-class results should be interpreted alongside accuracy. No hyperparameter search or feature selection was performed.

## Software versions

{chr(10).join(f'- {name}: `{value}`' for name, value in metrics['software_versions'].items())}

## Reproduction artifacts

- Split indices: `data/processed/train_indices.npy`, `data/processed/test_indices.npy`
- Split metadata: `data/processed/baseline_split.json`
- Configuration and data/split fingerprints: `results/baseline/configuration.json`
- Metrics: `results/baseline/metrics.json`
- Per-class metrics: `results/baseline/classification_report.csv`
- Confusion matrix: `results/baseline/confusion_matrix.csv`
- Trained XGBoost model: `results/baseline/xgboost_model.json`

Re-run with `python src/camo_fs/experiments/baseline_xgboost.py`. Existing saved indices are reused; the script refuses to silently replace them if they are incomplete or do not match the data fingerprint.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def _distribution_markdown(split: dict[str, Any]) -> str:
    full = split["full_class_distribution"]
    train = split["train_class_distribution"]
    test = split["test_class_distribution"]
    return "\n".join(
        f"| {label} | {full[label]['count']:,} | {full[label]['proportion']:.4%} | "
        f"{train[label]['count']:,} | {train[label]['proportion']:.4%} | "
        f"{test[label]['count']:,} | {test[label]['proportion']:.4%} |"
        for label in full
    )


def main() -> None:
    start_total = time.perf_counter()
    if not DATASET_PATH.is_file() or not FEATURE_METADATA_PATH.is_file():
        raise FileNotFoundError("Step 1 Parquet and feature metadata must exist before this baseline.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset_hash = sha256_file(DATASET_PATH)
    df = pd.read_parquet(DATASET_PATH)
    metadata = pd.read_csv(FEATURE_METADATA_PATH, dtype=str)
    feature_columns, classes = validate_dataset(df, metadata)
    labels = df["label"].to_numpy(dtype=np.int64, copy=True)
    train_ids, test_ids, split_info = create_or_load_split(labels, dataset_hash, classes)

    # Import only after schema and split checks so a missing XGBoost installation
    # still leaves the canonical split saved for all future experiments.
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is required to run this baseline. Install project requirements with "
            "`python -m pip install -r requirements.txt`. The seed-42 split has been saved."
        ) from exc
    from xgboost import XGBClassifier

    X = df[feature_columns].to_numpy(dtype=np.float32, copy=True)
    class_to_code = {int(label): code for code, label in enumerate(classes)}
    code_to_class = {code: int(label) for label, code in class_to_code.items()}
    y_encoded = np.fromiter((class_to_code[int(v)] for v in labels), dtype=np.int32, count=len(labels))

    model = XGBClassifier(num_class=EXPECTED_CLASSES, **MODEL_CONFIG)
    fit_start = time.perf_counter()
    # No eval_set is supplied: the test set is never used during training.
    model.fit(X[train_ids], y_encoded[train_ids], verbose=False)
    fit_seconds = time.perf_counter() - fit_start

    evaluation_start = time.perf_counter()
    predicted_codes = model.predict(X[test_ids]).astype(np.int64)
    y_pred = np.fromiter((code_to_class[int(v)] for v in predicted_codes), dtype=np.int64, count=len(predicted_codes))
    y_true = labels[test_ids]
    metric_values = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=classes)),
    }
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    report_rows = []
    for label in classes:
        row = report_dict[str(int(label))]
        report_rows.append(
            {
                "class_label": int(label),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1-score": float(row["f1-score"]),
                "support": int(row["support"]),
            }
        )
    class_report = pd.DataFrame(report_rows)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{int(c)}" for c in classes],
        columns=[f"pred_{int(c)}" for c in classes],
    )
    evaluation_seconds = time.perf_counter() - evaluation_start
    total_seconds = time.perf_counter() - start_total

    model_path = RESULTS_DIR / "xgboost_model.json"
    model.save_model(model_path)
    class_report.to_csv(RESULTS_DIR / "classification_report.csv", index=False)
    cm_df.to_csv(RESULTS_DIR / "confusion_matrix.csv", index=True, index_label="true_label")

    software = {
        "python": platform.python_version(),
        "numpy": library_version("numpy"),
        "pandas": library_version("pandas"),
        "scikit-learn": library_version("scikit-learn"),
        "xgboost": xgb.__version__,
        "pyarrow": library_version("pyarrow"),
    }
    config = {
        "experiment": "full_feature_xgboost_baseline",
        "not_camo_fs_result": True,
        "dataset": {
            "path": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
            "sha256": dataset_hash,
            "rows": int(len(df)),
            "feature_count": len(feature_columns),
            "feature_names": feature_columns,
            "target": "label",
            "classes": [int(c) for c in classes],
            "representation_counts": metadata["representation"].value_counts().to_dict(),
        },
        "split": split_info,
        "label_encoding_for_xgboost": {str(k): v for k, v in class_to_code.items()},
        "model": {"class": "xgboost.XGBClassifier", "configuration": MODEL_CONFIG | {"num_class": EXPECTED_CLASSES}},
        "test_set_used_only_for_final_evaluation": True,
        "software_versions": software,
    }
    (RESULTS_DIR / "configuration.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "experiment": "full_feature_xgboost_baseline",
        "not_camo_fs_result": True,
        "dataset": config["dataset"],
        "split": split_info,
        "model": config["model"],
        "metrics": metric_values,
        "class_wise": report_rows,
        "confusion_matrix": {
            "labels": [int(c) for c in classes],
            "counts_true_rows_predicted_columns": cm.tolist(),
        },
        "runtime_seconds": {
            "fit": fit_seconds,
            "final_evaluation": evaluation_seconds,
            "total": total_seconds,
        },
        "software_versions": software,
        "artifacts": {
            "model": str(model_path.relative_to(PROJECT_ROOT)),
            "classification_report": "results/baseline/classification_report.csv",
            "confusion_matrix": "results/baseline/confusion_matrix.csv",
            "configuration": "results/baseline/configuration.json",
            "split_info": "data/processed/baseline_split.json",
            "train_indices": "data/processed/train_indices.npy",
            "test_indices": "data/processed/test_indices.npy",
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(metrics, class_report)

    print("Full-feature XGBoost baseline complete")
    print(json.dumps({
        "train_rows": len(train_ids),
        "test_rows": len(test_ids),
        "feature_count": len(feature_columns),
        "metrics": metric_values,
        "class_wise_f1": {str(row["class_label"]): row["f1-score"] for row in report_rows},
        "runtime_seconds": metrics["runtime_seconds"],
        "xgboost_version": xgb.__version__,
        "saved_split_reused": not split_info["created_on_this_run"],
    }, indent=2))


if __name__ == "__main__":
    main()
